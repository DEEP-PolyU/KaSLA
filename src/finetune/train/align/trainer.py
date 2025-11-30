import math
import os
import sys
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

from ...extras.callbacks import FixValueHeadModelCallback, LogCallback
from ...extras.logging import get_logger
from ...extras.misc import AverageMeter, count_parameters, get_logits_processor

from .utils import dump_layernorm, restore_layernorm

from .ppo_trainer_ import PPOTrainer

from ...extras.constants import IGNORE_INDEX
import numpy as np
import inspect
import json
import sqlite3
import multiprocessing as mp
from func_timeout import func_timeout, FunctionTimedOut
import time
import re
import warnings

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""
    Inherits PPOTrainer.
    """
    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: List["TrainerCallback"],
        **kwargs,
    ):
        PPOTrainer.__init__(self, **kwargs)

        self.args = training_args
        self.model_args = model_args
        self.generating_args = generating_args
        self.finetuning_args = finetuning_args

        # self.label_pad_token_id = self.tokenizer.pad_token_id
        self.label_pad_token_id = IGNORE_INDEX
        self.ppo_ftx = finetuning_args.ppo_ftx
        logger.info("finetuning_args.ppo_epochs: {}".format(self.config.ppo_epochs))

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(self.save_callback, FixValueHeadModelCallback)

        if self.args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")




    def get_batch_logps_ber_token(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != self.label_pad_token_id
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps = per_token_logps * loss_mask
        return per_token_logps.sum(-1) / loss_mask.sum(-1), per_token_logps


    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = True,
    ) -> torch.FloatTensor:

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        loss_mask = labels != self.label_pad_token_id
        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0
        # 计算预测的候选词概率中label token的概率，越大越好
        # 如果仅仅是softmax，得到的结果差距很小很小，拉不开差距
        # 我有点不想尝试这个代码了，因为motivation根本没想好，现在就是换一种套路的sft，完全没有insight，
        # 我应该去思考sft那边应该怎么搞。放上去一个llama3先训练起来看看
        # 给每个token各自的reward是合理的，不采用KL也是合理的，SQL这里完全没必要采用KL。想象不出来有什么切实意义
        # 当然，也许保留是有价值的。
        # 并且我也不觉得sft model作为critic model有什么用， 我还没想清楚这个点
        print("logits.size() {}".format(logits.size()))
        print("logits {}".format(logits))
        print("logits.log_softmax(-1).size() {}".format(logits.log_softmax(-1).size()))
        print("logits.log_softmax(-1) {}".format(logits.log_softmax(-1)))
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        print("per_token_logps.size() {}".format(per_token_logps.size()))
        print("per_token_logps {}".format(per_token_logps))
        print("loss_mask {}".format(loss_mask))
        print("per_token_logps * loss_mask.size() {}".format((per_token_logps * loss_mask).size()))
        print("per_token_logps * loss_mask {}".format(per_token_logps * loss_mask))

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _step_safety_checker_query_labels(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        labels: List[torch.LongTensor]
    ):
        for name, tensor_list in zip(["queries", "labels"], [queries, labels]):
            if not isinstance(tensor_list, list):
                raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
                )
        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        labels = [tensor.to(self.current_device) for tensor in labels]

        return queries, labels


    def prepare_model_inputs_queries_labels(self, queries: [torch.Tensor], labels: [torch.Tensor]):
        # if not self.is_encoder_decoder:
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, labels)]
        input_data = self.data_collator(
            [{"input_ids": ids, "attention_mask": torch.ones_like(ids)} for ids in input_ids]
        ).to(self.current_device)
        input_data.pop("labels", None)
        (bs, max_length) = input_data["input_ids"].size()
        padded_labels = self.label_pad_token_id * torch.ones((bs, max_length), dtype=torch.long).to(self.current_device)
        for i in range(bs):
            padded_labels[i][queries[i].size(-1): queries[i].size(-1) + labels[i].size(-1)] = labels[i]
        padded_labels = padded_labels[:, 1:]
        return input_data, padded_labels
    @torch.no_grad()
    def inverse_sft_loss_as_reward(
        self,
        queries: List[torch.Tensor],
        labels: List[torch.Tensor]
    ) -> Tuple[List[torch.FloatTensor], torch.FloatTensor]:
        bs = len(queries)
        queries, labels = self._step_safety_checker_query_labels(bs, queries, labels)
        model_inputs, padded_labels = self.prepare_model_inputs_queries_labels(queries, labels)
        with torch.no_grad():
            all_logprobs, logits, values, masks = self.batched_forward_pass(
                self.model,
                queries,
                labels,
                model_inputs,
                return_logits=True
            )
        avg_token_logps, per_token_logps = self.get_batch_logps_ber_token(logits, padded_labels)
        # inverse_sft_loss = self.get_batch_logps(logits, padded_labels, average_log_prob=True)

        # if self.ppo_ftx > 1e-6:
        #     inverse_sft_loss = self.ppo_ftx * inverse_sft_loss
        return list(avg_token_logps), per_token_logps

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {}".format(num_examples))
            logger.info("  Num Epochs = {}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {}".format(self.args.per_device_train_batch_size))
            logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {}".format(
                    total_train_batch_size
                )
            )
            logger.info("  Gradient Accumulation steps = {}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {}".format(self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {}".format(max_steps))
            logger.info("  Number of trainable parameters = {}".format(count_parameters(self.model)[0]))

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)
        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            batch_labels = batch["labels"]
            del batch["labels"]

            # Cast to inference mode
            unwrapped_model.gradient_checkpointing_disable()
            unwrapped_model.config.use_cache = True
            self.model.eval()

            # Get inputs
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, scores, labels = [], [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                (mini_batch_queries, mini_batch_responses, mini_batch_labels) \
                    = self.get_inputs(batch[idx: idx + self.config.mini_batch_size], batch_labels[idx: idx + self.config.mini_batch_size])
                # execution
                mini_batch_scores, per_token_logps = self.inverse_sft_loss_as_reward(mini_batch_queries, mini_batch_labels)
                # mini_batch_scores = self.execute_in_DB(mini_batch_queries, mini_batch_responses, mini_batch_labels)

                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                labels.extend(mini_batch_labels)
                scores.extend(mini_batch_scores)

            scores_list = []
            for t_i in scores:
                scores_list.append(t_i.item())
            logger.info(" max scores: {:.2f}, min scores: {:.2f}, mean scores: {:.2f}".format(
                 max(scores_list), min(scores_list), np.mean(scores_list)))
            # Cast to training mode
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
            self.model.train()

            # Run PPO step
            stats = self.step(queries, labels, scores)
            # stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(scores))
            reward_meter.update(torch.stack(scores).mean().item(), n=len(scores))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, scores)
                except Exception:
                    logger.warning("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.log_callback.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, self.control)

                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step))
                )
                self.save_callback.on_save(
                    self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
                )

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.log_callback.on_train_end(self.args, self.state, self.control)
        self.save_callback.on_train_end(
            self.args, self.state, self.control, model=self.accelerator.unwrap_model(self.model)
        )

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, torch.Tensor], labels_batch: torch.LongTensor) \
            -> Tuple[List[torch.LongTensor], List[torch.LongTensor], List[torch.LongTensor]]:
        r"""
        Generates model's responses given queries.
        """
        if self.model_args.upcast_layernorm:
            layernorm_params = dump_layernorm(self.model)

        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            logger.info("handle llama2 ppo with gradient accumulation > 1")
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(self.model)

        generate_output: torch.Tensor = unwrapped_model.generate(
            generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
        )
        if self.model_args.upcast_layernorm:
            restore_layernorm(self.model, layernorm_params)
        query = batch["input_ids"].detach().cpu()
        label = labels_batch.detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1):].detach().cpu()

        queries, responses, labels = [], [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            # labels_start_index = (label[i] != IGNORE_INDEX).nonzero()[0].item()

            labels_no_pad_index = (label[i] != IGNORE_INDEX).nonzero()
            labels_start_index, labels_end_index = labels_no_pad_index[0].item(), labels_no_pad_index[-1].item()

            response_index = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_index) == 0:
                response_length = 1  # allow empty response
            else:
                response_length = response_index[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            # labels.append(label[i, labels_start_index:])  # remove padding from left
            labels.append(label[i, labels_start_index: labels_end_index + 1])  # remove padding from left and right
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses, labels


    def execute_one_sql(self, sql, db_path):
        conn = sqlite3.connect(db_path)
        # Connect to the database
        cursor = conn.cursor()
        cursor.execute(sql)
        res = cursor.fetchall()
        return res


    @torch.no_grad()
    def execute_in_DB(
        self,
        mini_batch_queries: List[torch.Tensor],
        mini_batch_responses: List[torch.Tensor],
        mini_batch_labels: List[torch.Tensor]
    ) -> List[torch.FloatTensor]:

        batch_query = self.tokenizer.batch_decode(mini_batch_queries, skip_special_tokens=True)
        batch_response = self.tokenizer.batch_decode(mini_batch_responses, skip_special_tokens=True)
        batch_label = self.tokenizer.batch_decode(mini_batch_labels, skip_special_tokens=True)
        mini_batch_rewards = []
        for i in range(len(batch_query)):
            a_query = batch_query[i]
            a_response = batch_response[i]
            a_label = batch_label[i]
            pred_sql = "SELECT" + a_response
            gold_sql = "SELECT" + a_label
            regex = r"%s(.*?)%s" % ("Database: <", ">")
            db_id = re.findall(regex, a_query)[0]
            db_path = self.generating_args.database_path + "/" + db_id + "/" + db_id + '.sqlite'
            reward = 0.0
            gold_results, pred_results = None, None
            try:
                gold_results = func_timeout(timeout=30.0, func=self.execute_one_sql, args=(gold_sql, db_path))
            except Exception as e:
                mini_batch_rewards.append(0.0)  # relevant database is error, don't consider this instance, reward = 0.0
                continue
            try:
                pred_results = func_timeout(timeout=30.0, func=self.execute_one_sql, args=(pred_sql, db_path))
                if len(set(pred_results)) == 0:
                    reward -= 1.0  # Executable but NoResults: reward = -1.0
                else:
                    reward += 1.0  # "Executable": reward = 1.0
                    if gold_results:
                        if set(gold_results) == set(pred_results):
                            reward += 2.0  # # "Correct": reward = 3.0
            except Exception as e:
                reward -= 2.0  # Un-executable: reward = -2.0
                # pred_report = "Execution Error: {}".format(e)
            mini_batch_rewards.append(reward)
        mini_batch_rewards = torch.Tensor(mini_batch_rewards)
        mini_batch_rewards = list(mini_batch_rewards)
        return mini_batch_rewards


    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            try:
                self._save(output_dir, state_dict=self.accelerator.get_state_dict(self.model))
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                self._save(output_dir, state_dict={})
                remove_dummy_checkpoint(True, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

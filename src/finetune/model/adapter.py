import inspect
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras.logging import get_logger
from .utils import find_all_linear_modules


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def init_adapter(
    model: "PreTrainedModel", model_args: "ModelArguments", finetuning_args: "FinetuningArguments", is_trainable: bool
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """

    if (not is_trainable) and model_args.adapter_name_or_path is None:
        logger.info("Adapter is not found at evaluation, load the base model.")
        return model

    if finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze" and is_trainable:
        logger.info("Fine-tuning method: Freeze")
        num_layers = (
            getattr(model.config, "num_hidden_layers", None)
            or getattr(model.config, "num_layers", None)
            or getattr(model.config, "n_layer", None)
        )
        if not num_layers:
            raise ValueError("Current model does not support freeze tuning.")

        if finetuning_args.num_layer_trainable > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [num_layers - k - 1 for k in range(finetuning_args.num_layer_trainable)]
        else:  # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-finetuning_args.num_layer_trainable)]  # noqa: C416

        trainable_layers = []
        for module_name in finetuning_args.name_module_trainable:
            for idx in trainable_layer_ids:
                trainable_layers.append("{:d}.{}".format(idx, module_name))

        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    # finetuning_type默认是lora，说明eval的时候不需要传也是lora
    if finetuning_args.finetuning_type == "lora":
        logger.info("Fine-tuning method: LoRA")
        adapter_to_resume = None

        if model_args.adapter_name_or_path is not None:
            is_mergeable = True
            if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
                assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
                is_mergeable = False

            if is_deepspeed_zero3_enabled():
                assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
                is_mergeable = False

            # 这里提供了类似断点续train的功能，接着之前的adapter训练
            # 神奇，那也说明了我在sft的基础上训练ppo的时候，其实可以选择接着sft的adapter来训练，也可以选择额外新建一个adapter
            # 两种不同做法的差异是什么
            if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
                adapter_to_merge = model_args.adapter_name_or_path[:-1]
                adapter_to_resume = model_args.adapter_name_or_path[-1]
            else:
                adapter_to_merge = model_args.adapter_name_or_path  # 平时fintuning走的是这个

            # 这里很关键，需要看懂是怎么运作的，后面的disable_adapter是否会影响sft的adapter（理论上应该不会）
            # 需要清楚在eval的时候为什么仍然需要传入2个，找到eval时候的代码
            # 其实不管是eval还是train都会走这一步，因此这就是eval时候的代码，如果之前采用的是resume则eval也只需要传一个adapter path
            # https://blog.csdn.net/liuqixuan1994/article/details/130664198
            # https://github.com/huggingface/peft/pull/263

            """
            在加载第一个适配器时，可以通过 PeftModel.from_pretrained 方法并指定 adapter_name 参数来给它命名。否则，将使用默认的适配器名称 default。
            要加载另一个适配器，请使用 PeftModel 的 load_adapter() 方法，例如：model.load_adapter(peft_model_path, adapter_name)
            要切换适配器，请使用 PeftModel 的 set_adapter() 方法，例如：model.set_adapter(adapter_name)
            要禁用适配器，请使用上下文管理器 disable_adapter()，例如：with model.disable_adapter()
            特别适用于LoRA方法：要合并和卸载当前活动的适配器，以便将LoRA权重添加到基础模型权重中，并将注入的LoRA模型删除以恢复具有添加了LoRA权重的Transformers基础模型的模型，
            请使用 merge_and_unload()方法，例如：model = model.merge_and_unload()
            """
            for adapter in adapter_to_merge:
                model = PeftModel.from_pretrained(model, adapter)
                model = model.merge_and_unload()

            if len(adapter_to_merge) > 0:
                logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))

            if adapter_to_resume is not None:  # resume lora training
                model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable)

        # 前面是load之前的lora adapter，这里新建adapter。
        # 根据blog的了解，disable-adapter应该只对ppo lora adapter起作用，因为前阶段的sft lora adapter已经和base model 融合merge了
        # 也就是此时base model已经是融合sft之后的样子。神奇，上面那篇lora的介绍可以帮助我们后面实现多个lora互相协作
        if is_trainable and adapter_to_resume is None:  # create new lora weights while training
            if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
                target_modules = find_all_linear_modules(model)
            else:
                target_modules = finetuning_args.lora_target

            peft_kwargs = {
                "r": finetuning_args.lora_rank,
                "target_modules": target_modules,
                "lora_alpha": finetuning_args.lora_alpha,
                "lora_dropout": finetuning_args.lora_dropout,
            }
            # Whether or not to use unsloth's optimization for the LoRA training.
            # 默认是false
            if model_args.use_unsloth:
                from unsloth import FastLlamaModel, FastMistralModel  # type: ignore

                unsloth_peft_kwargs = {"model": model, "max_seq_length": model_args.model_max_length}
                if "loftq_config" in inspect.signature(FastLlamaModel.get_peft_model).parameters:
                    unsloth_peft_kwargs["loftq_config"] = {}

                if getattr(model.config, "model_type", None) == "llama":
                    model = FastLlamaModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
                elif getattr(model.config, "model_type", None) == "mistral":
                    model = FastMistralModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
                else:
                    raise NotImplementedError

            else:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    modules_to_save=finetuning_args.additional_target,
                    **peft_kwargs,
                )
                model = get_peft_model(model, lora_config)

        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.bfloat16 if finetuning_args.lora_bf16_mode else torch.float32)

    if model_args.adapter_name_or_path is not None:
        logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    return model

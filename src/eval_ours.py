import argparse
import os
import torch
import json
import time
import numpy as np
import dill
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.db_utils import check_sql_executability, detect_special_char, get_matched_content_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from simcse import SimCSE
from peft import PeftModel
import random
from transformers.trainer_utils import set_seed
from utils.schema_item_filter import SchemaItemClassifierInference


from utils_data_processing.prepare_inputs import (prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema, \
    prepare_sequence_t2s_i_1schema_o_1_sql, prepare_fewshot_input_seq, prepare_sequence_t2s_codesStyle_json,prepare_sequence_t2s_codesStyle_json_GPT,
   prepare_sequence_t2s_codesStyle_str, prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike,
                                                  prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike)
#
from utils_data_processing.prepare_schema import (get_db_schema_sequence_noType_noValue, get_db_schema_sequence_noType,
                                                  get_db_schema_sequence_all, get_db_schema_sequence_codesStyle, get_db_schema_sequence_codesStyle_noV)
from utils_data_processing.prepare_data_filter import (filter_schema_ideal, filter_schema_full, filter_schema_codesStyle, filter_schema_codesStyle_ranking_cut,
                                                       filter_schema_onlySLM, filter_schema_DTS, filter_schema_codesStyle_KSL_filterByProb,
                                                       filter_schema_onlySLM_DTSSQL, filter_schema_onlySLM_TASQL)
from utils_data_processing.utils import (prepare_inputs, text2sql_func,  extract_skeleton)

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_path', type = str, default = None)
    parser.add_argument('--sic_path', type = str, default = None)
    parser.add_argument('--table_num', type = int, default = None)
    parser.add_argument('--column_num', type = int, default = None)
    parser.add_argument('--t_padding', type = int, default = None)
    parser.add_argument('--c_padding', type = int, default = None)

    parser.add_argument('--dataset_path', type = str)
    parser.add_argument('--mode', type = str)

    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 256)

    parser.add_argument('--predicted_filename', type = str)
    parser.add_argument('--adapter_name_or_path', type = str, default=None)
    parser.add_argument('--schema_linking_file', type = str, default = None)
    parser.add_argument('--num_beams', type = int, default = 2)
    parser.add_argument('--others', type = str, default=None)

    parser.add_argument('--recall_perT_file', type = str, default = None)

    parser.add_argument('--schema_linking_file_demonstration_set', type = str, default=None)
    parser.add_argument('--demonstration_set_path', type = str, default=None)
    parser.add_argument('--num_of_demonstrations', type = int, default=None)


    parser.add_argument('--masked_question_train', type = str, default=None)
    parser.add_argument('--masked_question_eval', type = str, default=None)


    parser.add_argument('--continuing', type = int, default = 0)

    opt = parser.parse_args()

    return opt



def post_process(sql, schema_items):
    sql = sql.replace("\n", " ")
    for table in schema_items:
        for column_name in table["column_names"]:
            if detect_special_char(column_name) and column_name in sql:
                sql = sql.replace(column_name, "`"+column_name+"`")

    while "``" in sql:
        sql = sql.replace("``", "`")

    return sql


class SFTSQLGenerationDataset_eval(Dataset):
    def __init__(self, text2sql_data_dir, tokenizer, max_tokens, mode, table_num, column_num,
                 sic_path, schema_linking_file, schema_linking_file_demonstration_set, num_of_demonstrations, eval_set_questions, eval_set_question_skeletons,
                 demonstration_set, demonstration_set_questions, demonstration_set_question_skeletons,
                 t_padding, c_padding, masked_demonstration_set_questions, masked_demonstration_set_question_skeletons,
         masked_eval_set_questions, masked_eval_set_question_skeletons, recall_perT_file):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))
        self.all_lens = []
        print("apply filtering strategies...")

        sic = None
        if "fewshot" not in mode:
            if "sg" in mode:
                if "KSL" in mode:
                    dataset = filter_schema_codesStyle_KSL_filterByProb(dataset, "eval", recall_perT_file=recall_perT_file,
                                                       num_top_k_tables=table_num, num_top_k_columns=column_num)
                    for data in dataset:
                        data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                        data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
                else:
                    dataset = filter_schema_ideal(dataset)
                    if "codesStyle" in mode:
                        for data in dataset:
                            data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["schema"])
                    if "-wT" in mode:
                        for data in dataset:
                            data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])
                    else:
                        for data in dataset:
                            data["schema_sequence"] = get_db_schema_sequence_noType(data["schema"])
            elif "codesStyle" in mode:
                if "full" in mode:
                    dataset = dataset
                elif "ideal" in mode:
                    dataset = filter_schema_ideal(dataset)
                elif "DTS-SQL" in mode:
                    dataset = filter_schema_onlySLM_DTSSQL(dataset, schema_linking_file, "eval")
                elif "TA-SQL" in mode:
                    dataset = filter_schema_onlySLM_TASQL(dataset, schema_linking_file, "eval")
                elif "onlySLM" in mode:
                    dataset = filter_schema_onlySLM(dataset, schema_linking_file)
                else:
                    sic = SchemaItemClassifierInference(sic_path)
                    # if "ideal" in mode:
                    #     dataset = filter_schema_codesStyle_ideal(dataset, "eval", sic=sic, num_top_k_tables=table_num,
                    #                                       num_top_k_columns=column_num)
                    if "ranking-cut" in mode:
                        dataset = filter_schema_codesStyle_ranking_cut(dataset, "eval", sic=sic, num_top_k_tables=table_num,
                                                          num_top_k_columns=column_num, cut=0.5)

                    else:
                        dataset = filter_schema_codesStyle(dataset, "eval", sic=sic, num_top_k_tables=table_num,
                                                          num_top_k_columns=column_num)

                if "full" in mode:
                    if "noV" in mode:
                        for data in dataset:
                            data["schema_sequence"] = get_db_schema_sequence_codesStyle_noV(data["schema"])
                    else:
                        for data in dataset:
                            data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["schema"])
                else:
                    for data in dataset:
                        data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                        data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
                    if "GPT" in mode:
                        for data in dataset:
                            data["full_schema_sequence"] = get_db_schema_sequence_codesStyle_noV(data["schema"])
            elif "brief" in mode:
                dataset = filter_schema_ideal(dataset)
                for data in dataset:
                    data["schema_sequence"] = get_db_schema_sequence_noType_noValue(data["schema"])
            elif "ideal" in mode:
                dataset = filter_schema_ideal(dataset)
                for data in dataset:
                    data["schema_sequence"] = get_db_schema_sequence_noType(data["filtered_schema"])
                    data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
            elif "full" in mode:
                for data in dataset:
                    # data["schema_sequence"] = get_db_schema_sequence_noType(data["schema"])
                    data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])
        else:
            eval_set = json.load(open(text2sql_data_dir))
            if "codesStyle" in mode:
                sic = SchemaItemClassifierInference(sic_path)

                if "onlySLM" in mode:
                    eval_set = filter_schema_onlySLM(eval_set, schema_linking_file)
                    demonstration_set = filter_schema_onlySLM(demonstration_set, schema_linking_file_demonstration_set)
                elif "ideal" in mode:
                    eval_set = filter_schema_ideal(eval_set)
                    demonstration_set = filter_schema_ideal(demonstration_set)
                elif "full" in mode:
                    eval_set = filter_schema_full(eval_set)
                    demonstration_set = filter_schema_full(demonstration_set)
                elif "DTS" in mode:
                    eval_set = filter_schema_DTS(eval_set, schema_linking_file, dataset_type="eval")
                    demonstration_set = filter_schema_DTS(eval_set, schema_linking_file_demonstration_set, dataset_type="train")
                elif "ranking-cut" in mode:
                    eval_set = filter_schema_codesStyle_ranking_cut(eval_set, "eval", sic=sic, num_top_k_tables=table_num,
                                                      num_top_k_columns=column_num, cut=0.5)
                    demonstration_set = filter_schema_codesStyle_ranking_cut(eval_set, "train", sic=sic, num_top_k_tables=table_num,
                                                      num_top_k_columns=column_num, cut=0.5)
                elif "ranking" in mode:
                    eval_set = filter_schema_codesStyle(eval_set, "eval", sic=sic, num_top_k_tables=table_num,
                                                        num_top_k_columns=column_num)
                    demonstration_set = filter_schema_codesStyle(demonstration_set,  "train", sic=sic, num_top_k_tables=table_num,
                                                      num_top_k_columns=column_num)
                else:
                    sic = SchemaItemClassifierInference(sic_path)
                    eval_set = filter_schema_codesStyle(eval_set, "eval", sic=sic, num_top_k_tables=table_num,
                                                      num_top_k_columns=column_num)
                    demonstration_set = filter_schema_codesStyle(demonstration_set,  "train", sic=sic, num_top_k_tables=table_num,
                                                      num_top_k_columns=column_num)
                for demonstration_sample in demonstration_set:
                    demonstration_sample["schema_sequence"] = get_db_schema_sequence_codesStyle(demonstration_sample["filtered_schema"])
                    demonstration_sample["content_sequence"] = get_matched_content_sequence(demonstration_sample["matched_contents"])
                for eval_sample in eval_set:
                    eval_sample["schema_sequence"] = get_db_schema_sequence_codesStyle(eval_sample["filtered_schema"])
                    eval_sample["content_sequence"] = get_matched_content_sequence(eval_sample["matched_contents"])
            else:
                eval_set = filter_schema_ideal(eval_set)
                demonstration_set = filter_schema_ideal(demonstration_set)

                for demonstration_sample in demonstration_set:
                    demonstration_sample["schema_sequence"] = get_db_schema_sequence_noType_noValue(demonstration_sample["schema"])
                for eval_sample in eval_set:
                    eval_sample["schema_sequence"] = get_db_schema_sequence_noType_noValue(eval_sample["schema"])

            simsce_model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
            question_similarities = simsce_model.similarity(eval_set_questions, demonstration_set_questions)
            question_skeleton_similarities = simsce_model.similarity(eval_set_question_skeletons,
                                                                     demonstration_set_question_skeletons)
            this_similarities = np.maximum(question_similarities, question_skeleton_similarities)
            print("this_similarities {}".format(len(this_similarities)))
            print(this_similarities)
            if "DAIL" in mode:
                masked_question_similarities = simsce_model.similarity(masked_eval_set_questions, masked_demonstration_set_questions)
                masked_question_skeleton_similarities = simsce_model.similarity(masked_eval_set_question_skeletons,
                                                                     masked_demonstration_set_question_skeletons)
                this_masked_similarities = np.maximum(masked_question_similarities, masked_question_skeleton_similarities)
                print("this_masked_similarities {}".format(len(this_masked_similarities)))
                print(this_masked_similarities)
                similarities = this_similarities * 0.5 + this_masked_similarities * 0.5
            else:
                similarities = this_similarities
            print("similarities {}".format(len(similarities)))
            print(similarities)


            del simsce_model

            self.eval_set = eval_set
            self.demonstration_set = demonstration_set
            self.similarities = similarities

        if sic is not None:
            del sic
        torch.cuda.empty_cache()

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.num_of_demonstrations = num_of_demonstrations

    def __getitem__(self, index):
        data = self.dataset[index]
        if "fewshot" in self.mode:
            eval_sample = self.eval_set[index]
            prefix_seq = prepare_fewshot_input_seq(self.tokenizer, self.max_tokens, self.mode, self.num_of_demonstrations,
                                                       eval_sample, self.demonstration_set, self.similarities[index])[0]
        elif "t2s-codesStyle" in self.mode:
            if "GPT" in self.mode:
                prefix_seq = prepare_sequence_t2s_codesStyle_json_GPT(data)[0]
            else:
                prefix_seq = prepare_sequence_t2s_codesStyle_json(data)[0]
        elif "sg" in self.mode:
            if "Dsl" in self.mode:
                prefix_seq = prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike(data)[0]
            elif "t2sTsl" in self.mode:
                prefix_seq = prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike(data)[0]
        else:
            if self.mode == "t2s_fullS-d" or self.mode == "t2s_idealS-d":
                prefix_seq = prepare_sequence_t2s_i_1schema_o_1_sql(data)[0]
        if index < 2:
            print(prefix_seq)

        inputs, lens = prepare_inputs(prefix_seq, self.tokenizer, self.max_tokens)
        self.all_lens.append(lens)
        print("\ninput lens {}\n".format(lens))

        return inputs

    def __len__(self):
        return len(self.dataset)



if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens

    print("max_tokens:", max_tokens)
    print("max_new_tokens:", max_new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
    raw_dataset = json.load(open(opt.dataset_path))



    if opt.num_of_demonstrations is not None:
        print("few shot setting")

        print(opt.demonstration_set_path)
        print(opt.num_of_demonstrations)
        print(opt.schema_linking_file_demonstration_set)
        eval_set_questions = [data["question"] for data in raw_dataset]
        eval_set_question_skeletons = [extract_skeleton(question) for question in eval_set_questions]
        print("length of evaluation set:", len(raw_dataset))
        demonstration_set = json.load(open(opt.demonstration_set_path))
        demonstration_set_questions = [data["question"] for data in demonstration_set]
        demonstration_set_question_skeletons = [extract_skeleton(question) for question in demonstration_set_questions]
        print("length of demonstration set:", len(demonstration_set))
        if opt.masked_question_train is not None and opt.masked_question_eval is not None and "DAIL" in opt.mode:
            print("DAIL-SQL")
            print(opt.masked_question_train)
            masked_question_train_json = json.load(open(opt.masked_question_train))
            print("length of masked_question_train_json:", len(masked_question_train_json))
            masked_demonstration_set_questions = []
            for i in range(len(masked_question_train_json)):
                masked_demonstration_set_questions.append(masked_question_train_json[str(i)][0])
            print("length of masked_demonstration_set_questions:", len(masked_demonstration_set_questions))
            masked_demonstration_set_question_skeletons = [extract_skeleton(question) for question in masked_demonstration_set_questions]

            masked_question_eval_json = json.load(open(opt.masked_question_eval))
            print("length of masked_question_eval_json:", len(masked_question_eval_json))
            masked_eval_set_questions = []
            for i in range(len(masked_question_eval_json)):
                masked_eval_set_questions.append(masked_question_eval_json[str(i)][0])
            print("length of masked_eval_set_questions:", len(masked_eval_set_questions))
            masked_eval_set_question_skeletons = [extract_skeleton(question) for question in masked_eval_set_questions]

        else:
            (masked_demonstration_set_questions, masked_demonstration_set_question_skeletons,
             masked_eval_set_questions, masked_eval_set_question_skeletons) = None, None, None, None

    else:
        (eval_set_questions, eval_set_question_skeletons,
         demonstration_set, demonstration_set_questions,
         demonstration_set_question_skeletons) = None, None, None, None, None
        (masked_demonstration_set_questions, masked_demonstration_set_question_skeletons,
         masked_eval_set_questions, masked_eval_set_question_skeletons) = None, None, None, None



    model = AutoModelForCausalLM.from_pretrained(opt.llm_path,
                                                 device_map = "auto",
                                                 torch_dtype = torch.float16)
    print("LLM path: {}".format(opt.llm_path))

    # if "codes" not in opt.mode:
    #     print("update eos token id of the tokenizer and the model to support early stop SQL generation")
    #     token_ids_of_example_sql = tokenizer("SELECT * FROM tables ;")["input_ids"]
    #     print(token_ids_of_example_sql)
    #     if token_ids_of_example_sql[-1] == tokenizer.eos_token_id:
    #         new_eos_token_id = token_ids_of_example_sql[-2]
    #     else:
    #         new_eos_token_id = token_ids_of_example_sql[-1]
    #     model.config.eos_token_id = new_eos_token_id
    #     tokenizer.eos_token_id = new_eos_token_id
    #     print("new_eos_token_id:", new_eos_token_id)
    #     print("tokenizer.decode(new_eos_token_id): '{}'".format(tokenizer.decode(new_eos_token_id)))
    # else:
    new_eos_token_id = None


    adapter_to_merge = opt.adapter_name_or_path
    if adapter_to_merge is not None:  # support merging multiple lora weights
        adapter_to_merge = [path.strip() for path in adapter_to_merge.split(",")]
        for adapter in adapter_to_merge:
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        print("Lora path: {}".format(opt.adapter_name_or_path))
    model.eval()


    print(opt.mode)
    eval_set = SFTSQLGenerationDataset_eval(
        opt.dataset_path,
        tokenizer,
        max_tokens - max_new_tokens,
        opt.mode,
        opt.table_num,
        opt.column_num,
        opt.sic_path,
        opt.schema_linking_file,
        opt.schema_linking_file_demonstration_set,
        opt.num_of_demonstrations,
        eval_set_questions, eval_set_question_skeletons,
        demonstration_set, demonstration_set_questions, demonstration_set_question_skeletons,
    opt.t_padding, opt.c_padding,masked_demonstration_set_questions, masked_demonstration_set_question_skeletons,
         masked_eval_set_questions, masked_eval_set_question_skeletons, opt.recall_perT_file
    )
    # TODO: current, we only support batch size = 1
    dataloader = DataLoader(eval_set, batch_size=1)

    start_time = time.time()
    if "sg-" in opt.mode:
        if "t2sTsl" in opt.mode:
            predicted_sqls = []
            # predicted_relevantTables = []
            predicted_relevantColumns = []
            predicted_error_reports = []
            the_index = -1
            for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
                the_index += 1
                if the_index < opt.continuing:
                    print("Index {}, jump to {}".format(str(the_index), str(opt.continuing)))
                    continue
                for key in batch_data:
                    batch_data[key] = batch_data[key].to(model.device)
                generated_strs = text2sql_func(model, batch_data, tokenizer, max_new_tokens, opt.num_beams, new_eos_token_id)

                generated_sqls = []
                # generated_relevantTables_list = []
                generated_relevantColumns_list = []
                for generated_str in generated_strs:
                    try:
                        if "}}" in generated_str:
                            generated_str = generated_str.split("}}")[0] + "}}"
                        generated_json = json.loads(generated_str)
                        this_sql = generated_json["SQL"]
                        if ";" in this_sql:
                            this_sql = this_sql.split(";")[0]
                        generated_sqls.append(this_sql)
                        generated_relevantColumns_list.append(generated_json["Relevant columns"])
                    except:
                        generated_sqls.append("Type error:\n" + generated_str)
                        generated_relevantColumns_list.append("Type error:\n" + generated_str)
                generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in
                                  generated_sqls]

                final_generated_sql = None
                final_generated_relevantColumns = None
                final_error_report = dict()
                for index in range(len(generated_sqls)):
                    generated_sql = generated_sqls[index]
                    generated_relevantColumns = generated_relevantColumns_list[index]
                    execution_error = check_sql_executability(generated_sql, raw_data["db_path"])

                    if execution_error is None:  # the generated sql has no execution errors, we will return it as the final generated sql
                        final_generated_sql = generated_sql
                        final_generated_relevantColumns = generated_relevantColumns
                        break
                    else:
                        this_dict = dict()
                        this_dict["execution_error"] = execution_error
                        this_dict["generated_sql"] = generated_sql
                        this_dict["generated_relevantColumns"] = generated_relevantColumns
                        final_error_report[str(index)] = this_dict

                if final_generated_sql is None:
                    final_generated_relevantColumns = generated_relevantColumns_list[0]
                    if generated_sqls[0].strip() != "":
                        final_generated_sql = generated_sqls[0]
                    else:
                        final_generated_sql = "SQL placeholder"
                print("Index {}".format(the_index))
                print(final_generated_sql)
                print(final_generated_relevantColumns)
                # print(list(final_generated_relevantColumns.keys()))
                print(final_error_report)
                predicted_sqls.append(final_generated_sql)
                predicted_relevantColumns.append(final_generated_relevantColumns)
                # predicted_relevantTables.append(list(final_generated_relevantColumns.keys()))
                predicted_error_reports.append(final_error_report)
            end_time = time.time()
            print("LLM name: {} | Total time: {}s | Example number: {} | Average time: {}s".format(
                opt.llm_path,
                end_time - start_time,
                len(raw_dataset),
                (end_time - start_time) / len(raw_dataset)
            )
            )

            print("LLM name:", opt.llm_path)
            if "bird" in opt.dataset_path:
                predict_relevantColumns_file = "Results_json_bird/Results_" + opt.predicted_filename + "_relevant_columns.json"
                json.dump(predicted_relevantColumns, open(predict_relevantColumns_file, 'w'), indent=4)
                predicted_error_reports_file = "Results_error_reports/Results_" + opt.predicted_filename + "_error_reports.json"
                json.dump(predicted_error_reports, open(predicted_error_reports_file, 'w'), indent=4)

                bird_results_dict = dict()
                for idx, (data, predicted_sql) in enumerate(zip(raw_dataset, predicted_sqls)):
                    bird_results_dict[idx] = predicted_sql + "\t----- bird -----\t" + data["db_id"]
                predict_dev_file = "Results_json_bird/Results_" + opt.predicted_filename + "_sql.json"
                with open(predict_dev_file, "w", encoding='utf-8') as f:
                    f.write(json.dumps(bird_results_dict, indent=2, ensure_ascii=False))
                os.system("sh evaluation/run_evaluation_bird.sh " + predict_dev_file)

            elif "spider" in opt.dataset_path:
                predict_relevantColumns_file = "Results_json_spider/Results_" + opt.predicted_filename + "_relevant_columns.json"
                json.dump(predicted_relevantColumns, open(predict_relevantColumns_file, 'w'), indent=4)
                predicted_error_reports_file = "Results_error_reports/Results_" + opt.predicted_filename + "_error_reports.json"
                json.dump(predicted_error_reports, open(predicted_error_reports_file, 'w'), indent=4)

                predict_dev_file = "Results_txt_spider/Results_" + opt.predicted_filename + "_sql.txt"
                with open(predict_dev_file, "w", encoding='utf-8') as f:
                    for sql in predicted_sqls:
                        f.write(sql + "\n")
                print("Execution accuracy:")
                os.system(
                    'python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred {} --db ./data/sft_data_collections/spider/database --etype exec'.format(
                        predict_dev_file))
                # print("Exact Match:")
                # os.system(
                #     'python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred {} --db ./data/sft_data_collections/spider/database --etype match'.format(
                #         predict_dev_file))
        elif "Dsl" in opt.mode:
            # predicted_relevantTables = []
            predicted_relevantColumns = []
            predicted_error_reports = []
            the_index = -1
            for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
                the_index += 1
                for key in batch_data:
                    batch_data[key] = batch_data[key].to(model.device)
                generated_strs = text2sql_func(model, batch_data, tokenizer, max_new_tokens, opt.num_beams, new_eos_token_id)
                # generated_relevantTables_list = []
                generated_relevantColumns_list = []
                final_error_report = []
                final_generated_relevantColumns = dict()
                for generated_str in generated_strs:
                    try:
                        if "}}" in generated_str:
                            generated_str = generated_str.split("}}")[0] + "}}"
                        generated_json = json.loads(generated_str)
                        final_generated_relevantColumns = generated_json["Relevant columns"]
                        break
                    except:
                        final_error_report.append("Type error:\n" + generated_str)

                print("Index {}".format(the_index))
                print(final_generated_relevantColumns)
                # print(list(final_generated_relevantColumns.keys()))
                print(final_error_report)
                predicted_relevantColumns.append(final_generated_relevantColumns)
                # predicted_relevantTables.append(list(final_generated_relevantColumns.keys()))
                predicted_error_reports.append(final_error_report)
            end_time = time.time()
            print("LLM name: {} | Total time: {}s | Example number: {} | Average time: {}s".format(
                opt.llm_path,
                end_time - start_time,
                len(raw_dataset),
                (end_time - start_time) / len(raw_dataset)
            )
            )

            print("LLM name:", opt.llm_path)
            if "bird" in opt.dataset_path:
                predict_relevantColumns_file = "Results_json_bird/Results_" + opt.predicted_filename + "_relevant_columns.json"
                json.dump(predicted_relevantColumns, open(predict_relevantColumns_file, 'w'), indent=4)
                predicted_error_reports_file = "Results_error_reports/Results_" + opt.predicted_filename + "_error_reports.json"
                json.dump(predicted_error_reports, open(predicted_error_reports_file, 'w'), indent=4)
            elif "spider" in opt.dataset_path:
                predict_relevantColumns_file = "Results_json_spider/Results_" + opt.predicted_filename + "_relevant_columns.json"
                json.dump(predicted_relevantColumns, open(predict_relevantColumns_file, 'w'), indent=4)
                predicted_error_reports_file = "Results_error_reports/Results_" + opt.predicted_filename + "_error_reports.json"
                json.dump(predicted_error_reports, open(predicted_error_reports_file, 'w'), indent=4)
    else:
        predicted_sqls = []
        predicted_error_reports = []
        the_index = -1
        for raw_data, batch_data in tqdm(zip(raw_dataset, dataloader)):
            the_index += 1
            for key in batch_data:
                batch_data[key] = batch_data[key].to(model.device)
            generated_strs = text2sql_func(model, batch_data, tokenizer, max_new_tokens, opt.num_beams, new_eos_token_id)
            generated_sqls = []
            for generated_str in generated_strs:
                if "SQL" in generated_str and "}" in generated_str:
                    try:
                        generated_str = generated_str.split("}")[0] + "}"
                        generated_json = json.loads(generated_str)
                        this_sql = generated_json["SQL"]
                        if ";" in this_sql:
                            this_sql = this_sql.split(";")[0]
                        generated_sqls.append(this_sql)
                    except:
                        generated_sqls.append("Type error:\n" + generated_str)
                else:
                    this_sql = generated_str
                    if ";" in this_sql:
                        this_sql = this_sql.split(";")[0]
                    generated_sqls.append(this_sql)

                generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in
                                  generated_sqls]

            final_generated_sql = None
            final_error_report = dict()
            for index in range(len(generated_sqls)):
                generated_sql = generated_sqls[index]
                execution_error = check_sql_executability(generated_sql, raw_data["db_path"])
                if execution_error is None:  # the generated sql has no execution errors, we will return it as the final generated sql
                    final_generated_sql = generated_sql
                    break
                else:
                    this_dict = dict()
                    this_dict["execution_error"] = execution_error
                    this_dict["generated_sql"] = generated_sql
                    final_error_report[str(index)] = this_dict

            if final_generated_sql is None:
                if generated_sqls[0].strip() != "":
                    final_generated_sql = generated_sqls[0]
                else:
                    final_generated_sql = "SQL placeholder"
            print("Index {}".format(the_index))
            print(final_generated_sql)
            print(final_error_report)
            predicted_sqls.append(final_generated_sql)
            predicted_error_reports.append(final_error_report)
        end_time = time.time()
        print("LLM name: {} | Total time: {}s | Example number: {} | Average time: {}s".format(
            opt.llm_path,
            end_time - start_time,
            len(raw_dataset),
            (end_time - start_time) / len(raw_dataset)
        )
        )

        print("LLM name:", opt.llm_path)
        if "bird" in opt.dataset_path:
            bird_results_dict = dict()
            for idx, (data, predicted_sql) in enumerate(zip(raw_dataset, predicted_sqls)):
                bird_results_dict[idx] = predicted_sql + "\t----- bird -----\t" + data["db_id"]
            predict_dev_file = "Results_json_bird/Results_" + opt.predicted_filename + ".json"
            with open(predict_dev_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(bird_results_dict, indent=2, ensure_ascii=False))
            os.system("sh evaluation/run_evaluation_bird.sh " + predict_dev_file)

            predicted_error_reports_file = "Results_error_reports/Results_" + opt.predicted_filename + "_error_reports.json"
            json.dump(predicted_error_reports, open(predicted_error_reports_file, 'w'), indent=4)

        elif "spider" in opt.dataset_path:
            spider_results_dict = dict()
            for idx, (data, predicted_sql) in enumerate(zip(raw_dataset, predicted_sqls)):
                spider_results_dict[idx] = predicted_sql + "\t----- spider -----\t" + data["db_id"]
            predict_dev_file = "Results_json_spider/Results_" + opt.predicted_filename + ".json"
            with open(predict_dev_file, "w", encoding='utf-8') as f:
                f.write(json.dumps(spider_results_dict, indent=2, ensure_ascii=False))
            os.system("sh evaluation/run_evaluation_spider.sh " + predict_dev_file)

            predicted_error_reports_file = "Results_error_reports/Results_" + opt.predicted_filename + "_error_reports.json"
            json.dump(predicted_error_reports, open(predicted_error_reports_file, 'w'), indent=4)

            predict_dev_file = "Results_txt_spider/Results_" + opt.predicted_filename + ".txt"
            with open(predict_dev_file, "w", encoding='utf-8') as f:
                for sql in predicted_sqls:
                    f.write(sql + "\n")
            print("Execution accuracy:")
            os.system(
                'python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred {} --db ./data/sft_data_collections/spider/database --etype exec'.format(
                    predict_dev_file))
            # print("Test suit execution accuracy:")
            # os.system(
            #     'python -u test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred {} --db test_suite_sql_eval/test_suite_database --etype exec'.format(
            #         predict_dev_file))

    print("max(eval_set.all_lens)")
    print(max(eval_set.all_lens))



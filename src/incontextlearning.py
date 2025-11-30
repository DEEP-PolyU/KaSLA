import argparse
import os
import torch
import json
import time
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.db_utils import check_sql_executability, detect_special_char, get_matched_content_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel
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
    parser.add_argument('--linking_result', type = str, default = None)
    parser.add_argument('--num_beams', type = int, default = 2)
    parser.add_argument('--others', type = str, default=None)


    parser.add_argument('--schema_linking_file_demonstration_set', type = str, default=None)
    parser.add_argument('--demonstration_set_path', type = str, default=None)
    parser.add_argument('--num_of_demonstrations', type = int, default=None)


    parser.add_argument('--masked_question_train', type = str, default=None)
    parser.add_argument('--masked_question_eval', type = str, default=None)

    parser.add_argument('--id_start', type = int, default=None)
    parser.add_argument('--id_end', type = int, default=None)

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
                 sic_path, linking_result):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))
        self.all_lens = []
        sic = None
        if "t2sTsl" in mode or "onlysl" in mode :
            dataset = filter_schema_ideal(dataset)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])
        elif "text2sql" in mode:
            if "full" in mode:
                dataset = dataset
            elif "ideal" in mode:
                dataset = filter_schema_ideal(dataset)
            elif "load" in mode:
                dataset = filter_schema_onlySLM(dataset, linking_result)
            elif "DTS-SQL" in mode:
                dataset = filter_schema_onlySLM_DTSSQL(dataset, linking_result, "eval")
            elif "TA-SQL" in mode:
                dataset = filter_schema_onlySLM_TASQL(dataset, linking_result, "eval")
            else:
                sic = SchemaItemClassifierInference(sic_path)
                if "ranking-cut" in mode:
                    dataset = filter_schema_codesStyle_ranking_cut(dataset, "eval", sic=sic, num_top_k_tables=table_num,
                                                      num_top_k_columns=column_num, cut=0.5)

                else:
                    dataset = filter_schema_codesStyle(dataset, "eval", sic=sic, num_top_k_tables=table_num,
                                                      num_top_k_columns=column_num)

            if "full" in mode:
                for data in dataset:
                    data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["schema"])
            else:
                for data in dataset:
                    data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                    data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])

        if sic is not None:
            del sic
        torch.cuda.empty_cache()

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __getitem__(self, index):
        data = self.dataset[index]
        if "onlysl" in self.mode:
            prefix_seq = prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike(data)[0]
        elif "t2sTsl" in self.mode:
            prefix_seq = prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike(data)[0]
        elif "text2sql" in self.mode:
            prefix_seq = prepare_sequence_t2s_codesStyle_json(data)[0]
        if index < 2:
            print(prefix_seq)

        inputs, lens = prepare_inputs(prefix_seq, self.tokenizer, self.max_tokens)
        self.all_lens.append(lens)
        print("\ninput lens {}\n".format(lens))

        return inputs

    def __len__(self):
        return len(self.dataset)


def execute_sql_bird(predicted_sql,gold_sql, db_path):
    db_path = db_path.replace("./data/sft_data_collections/bird/dev/dev_databases/", "/home/chen/data/zheng/datasets_SQL/bird/dev/databases/")
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    cursor.execute(gold_sql)
    gold_sql_res = cursor.fetchall()
    res = 0
    if set(predicted_res) == set(gold_sql_res):
        res = 1
    return res

def check_correctness(raw_data, predicted_sql):
    db_path = raw_data["db_path"]
    gold_sql = raw_data["sql"]
    predicted_sql = predicted_sql.replace('\n', ' ').replace('"', "`").replace('\"', "`")
    try:
        res = func_timeout(30.0, execute_sql_bird, args=(predicted_sql, gold_sql, db_path))
        error = "incorrect answer" if res == 0 else "--"
    except FunctionTimedOut:
        error = "timeout"
        res = 0
    except Exception as e:
        error = str(e)
        res = 0
    return {'exec_res': res, 'exec_err': error}

def update_results_dict(results_file_path, output_dict):
    if os.path.exists(results_file_path):
        contents = json.load(open(results_file_path, 'r'))
    else:
        contents = {}
    contents.update(output_dict)
    json.dump(contents, open(results_file_path, 'w'), indent=4)

def update_results_list(results_file_path, output_dict):
    if os.path.exists(results_file_path):
        contents = json.load(open(results_file_path, 'r'))
        contents.append(output_dict)
        json.dump(contents, open(results_file_path, 'w'), indent=4)
    else:
        contents = [output_dict]
        json.dump(contents, open(results_file_path, 'w'), indent=4)

def main():
    opt = parse_option()
    print(opt)
    max_tokens = opt.max_tokens
    max_new_tokens = opt.max_new_tokens
    print("max_tokens:", max_tokens)
    print("max_new_tokens:", max_new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(opt.llm_path)
    raw_dataset = json.load(open(opt.dataset_path))

    model = AutoModelForCausalLM.from_pretrained(opt.llm_path,
                                                 device_map = "auto",
                                                 torch_dtype = torch.float16)
    print("LLM path: {}".format(opt.llm_path))

    new_eos_token_id = None
    adapter_to_merge = opt.adapter_name_or_path
    if adapter_to_merge is not None:
        adapter_to_merge = [path.strip() for path in adapter_to_merge.split(",")]
        for adapter in adapter_to_merge:
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        print("Lora path: {}".format(opt.adapter_name_or_path))
    model.eval()


    print(opt.mode)
    eval_set = SFTSQLGenerationDataset_eval(
        opt.dataset_path, tokenizer,  max_tokens - max_new_tokens, opt.mode,  opt.table_num,
        opt.column_num, opt.sic_path, opt.linking_result
    )
    dataloader = DataLoader(eval_set, batch_size=1)

    if not os.path.exists(opt.predicted_filename):
        os.makedirs(opt.predicted_filename)
    start_time = time.time()
    predict_relevantColumns_file = opt.predicted_filename + "/relevant_columns.json"
    predicted_reports_file = opt.predicted_filename + "/reports.json"
    predict_dev_file = opt.predicted_filename + "/predict_dev.json"

    pre_exp_question_ids = []
    if os.path.isfile(predict_dev_file):
        pre_contents = json.load(open(predict_dev_file, 'r'))
        if len(pre_contents)!=0:
            for k, v in pre_contents.items():
                if v and "error" not in v:
                    pre_exp_question_ids.append(int(k))
            print(f"Original number of tasks: {len(raw_dataset)}")
            print(f"Pre exp number of tasks: {len(pre_exp_question_ids)}")
            print(f"Total number of tasks: {len(raw_dataset) - len(pre_exp_question_ids)}")
    if opt.id_start:
        print(f"Processing questions from question-{opt.id_start} to {opt.id_end}")
    predicted_sqls = []
    for ind, (raw_data, batch_data) in tqdm(enumerate(zip(raw_dataset, dataloader))):
        if opt.id_start:
            if ind < opt.id_start or ind > opt.id_end:
                print(f"Question {ind} is not belong to this task, jump to the next question\n")
                continue
        if ind in pre_exp_question_ids:
            print(f"Question {ind} has been processed, jump to the next question\n")
            continue


        print("\ninputs\n")
        print(batch_data)
        print("\ninputs\n")
        for key in batch_data:
            batch_data[key] = batch_data[key].to(model.device)
        generated_strs = text2sql_func(model, batch_data, tokenizer, max_new_tokens, opt.num_beams, new_eos_token_id)

        if "text2sql" in opt.mode:
            generated_sqls = []
        elif "onlysl" in opt.mode:
            generated_relevantColumns_list = []
        else:
            generated_sqls = []
            generated_relevantColumns_list = []

        for generated_str in generated_strs:
            try:
                if "codes-15b-bird-with-evidence" in opt.llm_path or "codes-15b-spider" in opt.llm_path:
                    this_sql = generated_str.strip()
                    generated_sqls.append(this_sql)
                else:
                    if "text2sql" in opt.mode:
                        if "}" in generated_str:
                            generated_str = generated_str.split("}")[0] + "}"
                        generated_json = json.loads(generated_str)
                        this_sql = generated_json["SQL"]
                        if ";" in this_sql:
                            this_sql = this_sql.split(";")[0]
                        generated_sqls.append(this_sql)
                    elif "onlysl" in opt.mode:
                        generated_relevantColumns_list.append(generated_json["Relevant columns"])
                    else:
                        if "}}" in generated_str:
                            generated_str = generated_str.split("}}")[0] + "}}"
                        generated_json = json.loads(generated_str)
                        this_sql = generated_json["SQL"]
                        if ";" in this_sql:
                            this_sql = this_sql.split(";")[0]
                        generated_sqls.append(this_sql)
                        generated_relevantColumns_list.append(generated_json["Relevant columns"])
            except:
                if "text2sql" in opt.mode:
                    generated_sqls.append("[JSON error] " + generated_str)
                elif "onlysl" in opt.mode:
                    generated_relevantColumns_list.append("[JSON error] " + generated_str)
                else:
                    generated_sqls.append("[JSON error] " + generated_str)
                    generated_relevantColumns_list.append("[JSON error] " + generated_str)


        if "text2sql" in opt.mode:
            generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in
                              generated_sqls]
            final_generated_sql = None
        elif "onlysl" in opt.mode:
            final_generated_relevantColumns = None
        else:
            generated_sqls = [post_process(generated_sql, raw_data["schema"]["schema_items"]) for generated_sql in
                              generated_sqls]
            final_generated_sql = None
            final_generated_relevantColumns = None


        if "text2sql" in opt.mode:
            for index in range(len(generated_sqls)):
                generated_sql = generated_sqls[index]

                db_path = raw_data["db_path"].replace("./data/sft_data_collections/bird/dev/dev_databases/", "/home/chen/data/zheng/datasets_SQL/bird/dev/databases/")
                execution_error = check_sql_executability(generated_sql, db_path)
                if execution_error is None:
                    final_generated_sql = generated_sql
                    break
            if final_generated_sql is None:
                if generated_sqls[0].strip() != "":
                    final_generated_sql = generated_sqls[0]
                else:
                    final_generated_sql = "SQL placeholder error"
        elif "onlysl" in opt.mode:
            for index in range(len(generated_relevantColumns_list)):
                if isinstance(generated_relevantColumns_list[index], list):
                    final_generated_relevantColumns = generated_relevantColumns_list[index]
                break
            if final_generated_relevantColumns is None:
                final_generated_relevantColumns = generated_relevantColumns_list[0]
        else:
            for index in range(len(generated_sqls)):
                generated_sql = generated_sqls[index]
                try:
                    generated_relevantColumns = generated_relevantColumns_list[index]
                except:
                    print("generated_relevantColumns_list error")
                    generated_relevantColumns = generated_relevantColumns_list[0]

                db_path = raw_data["db_path"].replace("./data/sft_data_collections/bird/dev/dev_databases/", "/home/chen/data/zheng/datasets_SQL/bird/dev/databases/")
                execution_error = check_sql_executability(generated_sql, db_path)
                if execution_error is None:
                    final_generated_sql = generated_sql
                    final_generated_relevantColumns = generated_relevantColumns
                    break
            if final_generated_sql is None:
                final_generated_relevantColumns = generated_relevantColumns_list[0]
                if generated_sqls[0].strip() != "":
                    final_generated_sql = generated_sqls[0]
                else:
                    final_generated_sql = "SQL placeholder error"




        # Compare predicted and ground truth sqls
        t2s_object_prediction = dict()
        t2s_object_prediction["question_id"] = str(ind)
        t2s_object_prediction["db_id"] = raw_data["db_id"]
        t2s_object_prediction["question"] = raw_data["question"]
        t2s_object_prediction["evidence"] = raw_data["evidence"]
        t2s_object_prediction["gold_sql"] = raw_data["sql"]

        if "text2sql" in opt.mode:
            compare_results = check_correctness(raw_data, final_generated_sql)
            t2s_object_prediction["pred_sql"] = final_generated_sql
            t2s_object_prediction['results'] = compare_results
        elif "onlysl" in opt.mode:
            t2s_object_prediction["pred_linking"] = final_generated_relevantColumns
        else:
            compare_results = check_correctness(raw_data, final_generated_sql)
            t2s_object_prediction["pred_sql"] = final_generated_sql
            t2s_object_prediction['results'] = compare_results
            t2s_object_prediction["pred_linking"] = final_generated_relevantColumns


        update_results_list(predicted_reports_file, t2s_object_prediction)


        if "text2sql" in opt.mode:
            output_dict = dict()
            if "bird" in opt.dataset_path:
                output_dict[str(ind)] = final_generated_sql + '\t----- bird -----\t' + raw_data["db_id"]
            else:
                output_dict[str(ind)] = final_generated_sql + '\t----- spider -----\t' + raw_data["db_id"]
            predicted_sqls.append(final_generated_sql)
            update_results_dict(predict_dev_file, output_dict)
            print(final_generated_sql)
            print(f"Question with {ind} is processed. Correctness: {compare_results['exec_res']} ")
        elif "onlysl" in opt.mode:
            update_results_list(predict_relevantColumns_file, final_generated_relevantColumns)
            print(final_generated_relevantColumns)
            print(f"Question with {ind} is processed.")
        else:
            output_dict = dict()
            if "bird" in opt.dataset_path:
                output_dict[str(ind)] = final_generated_sql + '\t----- bird -----\t' + raw_data["db_id"]
            else:
                output_dict[str(ind)] = final_generated_sql + '\t----- spider -----\t' + raw_data["db_id"]
            update_results_dict(predict_dev_file, output_dict)
            update_results_list(predict_relevantColumns_file, final_generated_relevantColumns)
            print(final_generated_sql)
            print(final_generated_relevantColumns)
            print(f"Question with {ind} is processed. Correctness: {compare_results['exec_res']} ")

    end_time = time.time()
    print("LLM name: {} | Total time: {}s | Example number: {} | Average time: {}s".format(
        opt.llm_path,
        end_time - start_time,
        len(raw_dataset),
        (end_time - start_time) / len(raw_dataset)
    )
    )
    print("LLM name:", opt.llm_path)

    if "onlysl" not in opt.mode:
        if "bird" in opt.dataset_path:
            os.system("sh evaluation/run_evaluation_bird.sh " + predict_dev_file)
        elif "spider" in opt.dataset_path:

            predict_dev_file = "Results_txt_spider/Results_" + opt.predicted_filename + "_sql.txt"
            with open(predict_dev_file, "w", encoding='utf-8') as f:
                for sql in predicted_sqls:
                    f.write(sql + "\n")
            os.system(
                'python -u src/test_suite_sql_eval/evaluation.py --gold ./data/sft_data_collections/spider/dev_gold.sql --pred {} --db ./data/sft_data_collections/spider/database --etype exec'.format(
                    predict_dev_file))

    print("max(eval_set.all_lens)")
    print(max(eval_set.all_lens))


if __name__ == "__main__":
    main()

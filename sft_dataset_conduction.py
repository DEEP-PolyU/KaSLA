import argparse
import os
import torch
import json
import time
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.utils_data_processing.prepare_inputs import (prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike,prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike,
                                                  prepare_sequence_t2s_dataset, prepare_sequence_t2s_codesStyle_json_fewshot, prepare_sequence_t2s_codesStyle_json, prepare_sequence_t2s_codesStyle_json_GPT)
from src.utils_data_processing.prepare_schema import (get_matched_content_sequence, get_db_schema_sequence_noType,
                                                  get_db_schema_sequence_all, get_db_schema_sequence_codesStyle, get_db_schema_sequence_codesStyle_noV)
from src.utils_data_processing.prepare_data_filter import (filter_schema_ideal, filter_schema_onlySLM, filter_schema_onlySLM_rankGPT, filter_schema_codesStyle, filter_schema_codesStyle_ranking_cut,
                                                       filter_schema_codesStyle_ranking_cut_KSL, filter_schema_codesStyle_KSL_filterByProb,
                                                       filter_schema_onlySLM_DTSSQL, filter_schema_onlySLM_TASQL,
                                                       filter_schema_onlySLM_DTSSQL_Gold, filter_schema_onlySLM_TASQL_Gold)

class SchemaFilterDataset_filtered(Dataset):
    def __init__(self, text2sql_data_dir, mode, schema_linking_file,
                 num_top_k_tables, num_top_k_columns, t_padding, c_padding, recall_perT_file):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir, encoding = 'utf-8'))
        self.mode = mode
        self.all_lens = []
        self.all_lens_output = []
        self.tokenizer = AutoTokenizer.from_pretrained("src/sic_ckpts/sic_bird_with_evidence")

        if "onlysl" in mode:
            dataset = filter_schema_ideal(dataset)
            if "pure" in mode:
                for data in dataset:
                    data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])
            else:
                for data in dataset:
                    data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["schema"])

        elif "t2sTsl" in mode:
            dataset = filter_schema_ideal(dataset)
            if "pure" in mode:
                for data in dataset:
                    data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])
            else:
                for data in dataset:
                    data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["schema"])

        elif "t2sTsl" in mode and "KSL" in mode:
            dataset = filter_schema_codesStyle_KSL_filterByProb(dataset, "train", recall_perT_file=recall_perT_file,
                                               num_top_k_tables=num_top_k_tables, num_top_k_columns=num_top_k_columns)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif "DTS-SQL" in mode:
            if "Gold" in mode:
                dataset = filter_schema_onlySLM_DTSSQL_Gold(dataset, "train")
            else:
                dataset = filter_schema_onlySLM_DTSSQL(dataset, schema_linking_file, "train")
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif "TA-SQL" in mode:
            if "Gold" in mode:
                dataset = filter_schema_onlySLM_TASQL_Gold(dataset, "train")
            else:
                dataset = filter_schema_onlySLM_TASQL(dataset, schema_linking_file, "train")
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif "onlySLM" in mode:
            dataset = filter_schema_onlySLM(dataset, schema_linking_file)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
            if "GPT" in self.mode:
                if "rank" in self.mode:
                    dataset = filter_schema_onlySLM_rankGPT(dataset, schema_linking_file)
                for data in dataset:
                    data["full_schema_sequence"] = get_db_schema_sequence_codesStyle_noV(data["schema"])

        elif "ideal" in mode:
            dataset = filter_schema_ideal(dataset)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif mode == "t2s-codesStyle-json-full":
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["schema"])
        elif mode == "t2s-codesStyle-json-full-noV":
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle_noV(data["schema"])
        elif "t2s-codesStyle-json" in mode:
            if "ranking-cut" in mode:
                if "KSL" in mode:
                    dataset = filter_schema_codesStyle_ranking_cut_KSL(dataset, "train", sic=None, cut=0.5)
                else:
                    dataset = filter_schema_codesStyle_ranking_cut(dataset, "train", sic=None, num_top_k_tables=num_top_k_tables,
                                                   num_top_k_columns=num_top_k_columns, cut = 0.5)
            else:
                dataset = filter_schema_codesStyle(dataset, "train", sic=None,
                                                   num_top_k_tables=num_top_k_tables, num_top_k_columns=num_top_k_columns)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif mode == "t2s_fullS-d":
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_noType(data["schema"])
        elif mode == "t2s_fullS-d-wT":
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])


        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]

        if "t2sDatasetForGPT" in self.mode:
            prefix_seq = prepare_sequence_t2s_dataset(data)
            self.all_lens.append(len(self.tokenizer(
                prefix_seq[0] + prefix_seq[1]+ prefix_seq[2], truncation=False)["input_ids"]))
            return prefix_seq
        elif "onlysl" in self.mode:
            prefix_seq = prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike(data)

        elif "t2sTsl" in self.mode:
            prefix_seq = prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike(data)
        elif "t2s" in self.mode:
            prefix_seq = prepare_sequence_t2s_codesStyle_json(data)
            if "GPT" in self.mode:
                prefix_seq = prepare_sequence_t2s_codesStyle_json_GPT(data)
        elif "fewshot" in self.mode:
            prefix_seq = prepare_sequence_t2s_codesStyle_json_fewshot(data)
            return prefix_seq
        elif "codesStyle" in self.mode:
            if "GPT" in self.mode:
                prefix_seq = prepare_sequence_t2s_codesStyle_json_GPT(data)
            else:
                prefix_seq = prepare_sequence_t2s_codesStyle_json(data)
            
        self.all_lens.append(len(self.tokenizer(prefix_seq[0] + prefix_seq[1], truncation = False)["input_ids"]))
        return prefix_seq

    def __len__(self):
        return len(self.dataset)


def parse_option():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type = str)
    parser.add_argument('--target_dataset_path', type = str)
    parser.add_argument('--mode', type = str)
    parser.add_argument('--schema_linking_file', type = str, default = None)
    parser.add_argument('--recall_perT_file', type = str, default = None)

    parser.add_argument('--table_num', type = int, default = None)
    parser.add_argument('--column_num', type = int, default = None)
    parser.add_argument('--t_padding', type = int, default = None)
    parser.add_argument('--c_padding', type = int, default = None)

    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_option()
    print(opt)

    raw_dataset = json.load(open(opt.dataset_path, encoding = 'utf-8'))

    eval_set = SchemaFilterDataset_filtered(
        opt.dataset_path,
        opt.mode,
        opt.schema_linking_file,
        opt.table_num,
        opt.column_num,
        opt.t_padding,
        opt.c_padding,
        opt.recall_perT_file
    )

    # TODO: current, we only support batch size = 1
    dataloader = DataLoader(eval_set, batch_size=1)

    start_time = time.time()
    new_dataset = []
    for raw_data, (batch_input, batch_output) in tqdm(zip(raw_dataset, dataloader)):
        db_id = raw_data["db_id"]
        question = raw_data["question"]
        evidence = raw_data["evidence"]
        gold_sql = raw_data["sql"]
        dataset = raw_data["source"]
        new_instance = {
            "instruction": db_id,
            "input": batch_input[0],
            "output": batch_output[0]
        }
        new_dataset.append(new_instance)

    print("mean {}".format(sum(eval_set.all_lens)/len(eval_set.all_lens)))
    print("max {}".format(max(eval_set.all_lens)))
    print()
    json.dump(new_dataset, open(opt.target_dataset_path, 'w'), indent=4)
    print(opt.target_dataset_path)

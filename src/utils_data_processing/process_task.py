import argparse
import os
import torch
import json
import time
import numpy as np
import random
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# from schema_item_filter import SchemaItemClassifierInference

from prepare_inputs import (prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema, \
    prepare_sequence_t2s_i_1schema_o_1_sql, prepare_sequence_t2s_codesStyle_json
, prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike, prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike)

from prepare_schema import get_matched_content_sequence, \
    get_db_schema_sequence_all, get_db_schema_sequence_noType, \
    get_db_schema_sequence_noType_noValue, get_db_schema_sequence_codesStyle
from prepare_data_filter import (filter_schema_ideal, filter_schema_rankingSL,
                                 filter_schema_codesStyle, filter_schema_filtered_and_rankingSL, filter_schema_filtered_and_rankingSL_train)



class SchemaFilterDataset_filtered(Dataset):
    def __init__(self, text2sql_data_dir, mode, sic_path, schema_linking_file):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))
        self.mode = mode
        self.all_lens = []
        self.all_lens_output = []
        self.tokenizer = AutoTokenizer.from_pretrained(sic_path)

        print("apply filtering strategies...")
        if mode == "t2s_rankingSL-mo":
            dataset = filter_schema_rankingSL(dataset, "train", num_top_k_tables=6, num_top_k_columns=10)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_all(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif mode == "t2s-codesStyle-json":
            dataset = filter_schema_codesStyle(dataset, "train",sic=None, num_top_k_tables=6, num_top_k_columns=10)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif mode == "t2s-codesStyle-FAR-json":
            from schema_item_filter import SchemaItemClassifierInference
            sic = SchemaItemClassifierInference(sic_path)
            dataset = filter_schema_filtered_and_rankingSL_train(dataset, schema_linking_file, sic, num_top_k_tables=6,
                                              num_top_k_columns=10)
            dataset = filter_schema_codesStyle(dataset, "train",sic=None, num_top_k_tables=6, num_top_k_columns=10)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif mode == "t2s_idealS-d":
            dataset = filter_schema_ideal(dataset)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_noType(data["filtered_schema"])
                data["content_sequence"] = get_matched_content_sequence(data["matched_contents"])
        elif mode == "t2s_fullS-d":
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_noType(data["schema"])
        elif mode == "t2s_fullS-d-wT":
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])
        elif mode == "sg-t2sTsl-fullS-d-wT":
            dataset = filter_schema_ideal(dataset)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])
        elif mode == "sg-Dsl-fullS-d-wT":
            dataset = filter_schema_ideal(dataset)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_all(data["schema"])
        elif mode == "sg-t2sTsl-fullS-d":
            dataset = filter_schema_ideal(dataset)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_noType(data["schema"])
        elif mode == "sg-t2sTsl-fullS-d-codesStyle":
            dataset = filter_schema_ideal(dataset)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_codesStyle(data["schema"])
        elif mode == "sg_t2s-sl_briefS":
            dataset = filter_schema_ideal(dataset)
            for data in dataset:
                data["schema_sequence"] = get_db_schema_sequence_noType_noValue(data["schema"])


        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        if "t2s-codesStyle" in self.mode:
            prefix_seq = prepare_sequence_t2s_codesStyle_json(data)
        elif "sg-t2sTsl" in self.mode:
            prefix_seq = prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike(data)
        elif "sg-Dsl" in self.mode:
            prefix_seq = prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike(data)
        elif "t2s" in self.mode:
            prefix_seq = prepare_sequence_t2s_i_1schema_o_1_sql(data)
            
        self.all_lens.append(len(self.tokenizer(prefix_seq[0] + prefix_seq[1], truncation = False)["input_ids"]))
        return prefix_seq

    def __len__(self):
        return len(self.dataset)


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sic_path', type = str)

    parser.add_argument('--dataset_path', type = str)
    parser.add_argument('--target_dataset_path', type = str, default = None)
    parser.add_argument('--mode', type = str)
    parser.add_argument('--schema_linking_file', type = str, default = None)

    # parser.add_argument('--max_tokens', type = int, default = 4096)
    # parser.add_argument('--max_new_tokens', type = int, default = 256)
    
    opt = parser.parse_args()

    return opt


if __name__ == "__main__":
    opt = parse_option()
    print(opt)

    raw_dataset = json.load(open(opt.dataset_path))

    eval_set = SchemaFilterDataset_filtered(
        opt.dataset_path,
        opt.mode,
        opt.sic_path,
        opt.schema_linking_file
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
    print(max(eval_set.all_lens))
    if opt.target_dataset_path is not None:
        json.dump(new_dataset, open(opt.target_dataset_path, 'w'), indent=4)
        print(opt.dataset_path)

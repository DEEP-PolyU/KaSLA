
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


def filter_schema_filtered_and_rankingSL_train_add(dataset, schema_linking_file, sic,
                                                   num_top_k_tables, num_top_k_columns, mode, t_padding, c_padding):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")


    if t_padding is None and c_padding is None:
        t_padding = 1
        c_padding = 2
    print(mode)
    print("num_top_k_tables {}".format(num_top_k_tables))
    print("num_top_k_columns {}".format(num_top_k_columns))
    print("t_padding {}".format(t_padding))
    print("c_padding {}".format(c_padding))
    print("filtering schema items for the dataset")


    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]
        print("predicted_schemalinking {}".format(predicted_schemalinking))

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]


        table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]

        print("table_indices {}".format(table_indices))
        pred_column_names_dict = dict()
        SLM_match_table_indices = []
        sic_pred_results = sic.predict(data)

        if len(table_indices) + t_padding < num_top_k_tables:
            this_num_top_k_tables = len(table_indices) + t_padding
        else:
            this_num_top_k_tables = num_top_k_tables

        if len(table_indices) < this_num_top_k_tables:
            print("Table padding with SIM")
            for table_idx, table_name in enumerate(table_names):
                try:
                    pred_column_names = predicted_schemalinking[table_name]
                    if pred_column_names == None:
                        continue
                    else:
                        if table_idx not in table_indices:
                            table_indices.append(table_idx)
                        SLM_match_table_indices.append(table_idx)
                        pred_column_names_dict[table_idx] = pred_column_names
                        if len(table_indices) >= this_num_top_k_tables:
                            break
                except:
                    continue
            print("table_indices {}".format(table_indices))

        if len(table_indices) < this_num_top_k_tables:
            print("Table padding with SIC")
            sic_table_probs = [pred_result["table_prob"] for pred_result in sic_pred_results]
            sic_table_indices = np.argsort(-np.array(sic_table_probs), kind="stable").tolist()
            if len(table_indices) < this_num_top_k_tables:
                for i in range(len(sic_table_indices)):
                    this_sic_table_indice = sic_table_indices[i]
                    if this_sic_table_indice not in table_indices:
                        table_indices.append(this_sic_table_indice)
                    if len(table_indices) >= this_num_top_k_tables:
                        break
            else:
                print("No table padding")
            print("table_indices {}".format(table_indices))


        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]

        print("ground_truth_schema_linking {}".format(ground_truth_schema_linking))

        random.shuffle(table_indices)
        print(len(table_indices))
        for table_idx in table_indices:
            if table_idx in ground_truth_table_indices:
                print("Using groundtruth\n")
                column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                  if column_label == 1]
                print("column_indices {}: {}".format(table_names[table_idx],
                                      [column_names[table_idx][column_idx] for column_idx in column_indices]))

                if len(column_indices) + c_padding < num_top_k_columns:
                    this_num_top_k_columns = len(column_indices) + c_padding
                else:
                    this_num_top_k_columns = num_top_k_columns


                if len(column_indices) < this_num_top_k_columns and table_idx in SLM_match_table_indices:
                    SLM_column_names = pred_column_names_dict[table_idx]
                    if type(SLM_column_names) == list:
                        for column_idx, column_name in enumerate(column_names[table_idx]):
                            if column_name in SLM_column_names and column_idx not in column_indices:
                                column_indices.append(column_idx)
                            if len(column_indices) >= this_num_top_k_columns:
                                break
                    print("Padding with SIM\n")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))
                else:
                    print("No need SIM column padding")
                if len(column_indices) < this_num_top_k_columns:
                    SIC_column_probs = sic_pred_results[table_idx]["column_probs"]
                    SIC_column_indices = np.argsort(-np.array(SIC_column_probs), kind="stable").tolist()
                    for j in range(len(SIC_column_indices)):
                        this_SIC_column_indice = SIC_column_indices[j]
                        if this_SIC_column_indice not in column_indices:
                            column_indices.append(this_SIC_column_indice)
                        if len(column_indices) >= this_num_top_k_columns:
                            break
                    print("Padding with SIC\n")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))
                else:
                    print("No need SIC column padding")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))

            elif table_idx in SLM_match_table_indices:
                print("Using SIM\n")
                column_indices = []
                SLM_column_names = pred_column_names_dict[table_idx]
                if type(SLM_column_names) == list:
                    for column_idx, column_name in enumerate(column_names[table_idx]):
                        if column_name in SLM_column_names and column_idx not in column_indices:
                            column_indices.append(column_idx)
                        if len(column_indices) >= num_top_k_columns:
                            break
                if len(column_indices) + c_padding < num_top_k_columns:
                    this_num_top_k_columns = len(column_indices) + c_padding
                else:
                    this_num_top_k_columns = num_top_k_columns

                if len(column_indices) < this_num_top_k_columns:
                    SIC_column_probs = sic_pred_results[table_idx]["column_probs"]
                    SIC_column_indices = np.argsort(-np.array(SIC_column_probs), kind="stable").tolist()
                    for j in range(len(SIC_column_indices)):
                        this_SIC_column_indice = SIC_column_indices[j]
                        if this_SIC_column_indice not in column_indices:
                            column_indices.append(this_SIC_column_indice)
                        if len(column_indices) >= this_num_top_k_columns:
                            break
                    print("Padding with SIC\n")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))
                else:
                    print("No need SIC column padding")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))

            else:
                print("Using SIC\n")
                sic_column_probs = sic_pred_results[table_idx]["column_probs"]
                column_indices = np.argsort(-np.array(sic_column_probs), kind="stable")[:c_padding].tolist()
                print("Generated by SIC\n")
                print("column_indices {}: {}".format(table_names[table_idx],
                                      [column_names[table_idx][column_idx] for column_idx in column_indices]))
            print("CCcc")
            print(len(column_names[table_idx]))
            print(len(column_indices))
            print("The End of this Table\n")
            random.shuffle(column_indices)

            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        # replace the old matched contents with the filtered matched contents
        data["matched_contents"] = filtered_matched_contents
        print("\n")
    return dataset

def filter_schema_filtered_and_rankingSL_train(dataset, schema_linking_file, sic, num_top_k_tables, num_top_k_columns):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]
        print("predicted_schemalinking {}".format(predicted_schemalinking))

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]


        table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]

        print("table_indices {}".format(table_indices))
        pred_column_names_dict = dict()
        SLM_match_table_indices = []
        sic_pred_results = sic.predict(data)

        if len(table_indices) < num_top_k_tables:
            print("Table padding with SIM")
            for table_idx, table_name in enumerate(table_names):
                try:
                    pred_column_names = predicted_schemalinking[table_name]
                    if pred_column_names == None:
                        continue
                    else:
                        if table_idx not in table_indices:
                            table_indices.append(table_idx)
                        SLM_match_table_indices.append(table_idx)
                        pred_column_names_dict[table_idx] = pred_column_names
                        if len(table_indices) >= num_top_k_tables:
                            break
                except:
                    continue
            print("table_indices {}".format(table_indices))

        if len(table_indices) < num_top_k_tables:

            sic_table_probs = [pred_result["table_prob"] for pred_result in sic_pred_results]
            sic_table_indices = np.argsort(-np.array(sic_table_probs), kind="stable").tolist()  # [:num_top_k_tables]

            if num_top_k_tables is not None:
                print("Table padding with SIC")
                if len(table_indices) < num_top_k_tables:
                    for i in range(len(sic_table_indices)):
                        this_sic_table_indice = sic_table_indices[i]
                        if this_sic_table_indice not in table_indices:
                            table_indices.append(this_sic_table_indice)
                        if len(table_indices) >= num_top_k_tables:
                            break
            else:
                print("No table padding")
            print("table_indices {}".format(table_indices))


        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]

        print("ground_truth_schema_linking {}".format(ground_truth_schema_linking))

        random.shuffle(table_indices)
        print(len(table_indices))
        for table_idx in table_indices:
            if table_idx in ground_truth_table_indices:
                print("Using groundtruth\n")
                column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                  if column_label == 1]
                print("column_indices {}: {}".format(table_names[table_idx],
                                      [column_names[table_idx][column_idx] for column_idx in column_indices]))
                if len(column_indices) < num_top_k_columns and table_idx in SLM_match_table_indices:
                    SLM_column_names = pred_column_names_dict[table_idx]
                    if type(SLM_column_names) == list:
                        for column_idx, column_name in enumerate(column_names[table_idx]):
                            if column_name in SLM_column_names and column_idx not in column_indices:
                                column_indices.append(column_idx)
                            if len(column_indices) >= num_top_k_columns:
                                break
                    print("Padding with SIM\n")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))
                else:
                    print("No need SIM column padding")
                if len(column_indices) < num_top_k_columns:
                    SIC_column_probs = sic_pred_results[table_idx]["column_probs"]
                    SIC_column_indices = np.argsort(-np.array(SIC_column_probs), kind="stable").tolist()  # [:num_top_k_columns]
                    for j in range(len(SIC_column_indices)):
                        this_SIC_column_indice = SIC_column_indices[j]
                        if this_SIC_column_indice not in column_indices:
                            column_indices.append(this_SIC_column_indice)
                        if len(column_indices) >= num_top_k_columns:
                            break
                    print("Padding with SIC\n")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))
                else:
                    print("No need SIC column padding")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))

            elif table_idx in SLM_match_table_indices:
                print("Using SIM\n")
                column_indices = []
                SLM_column_names = pred_column_names_dict[table_idx]
                if type(SLM_column_names) == list:
                    for column_idx, column_name in enumerate(column_names[table_idx]):
                        if column_name in SLM_column_names and column_idx not in column_indices:
                            column_indices.append(column_idx)
                        if len(column_indices) >= num_top_k_columns:
                            break
                if len(column_indices) < num_top_k_columns:
                    SIC_column_probs = sic_pred_results[table_idx]["column_probs"]
                    SIC_column_indices = np.argsort(-np.array(SIC_column_probs), kind="stable").tolist()  # [:num_top_k_columns]
                    for j in range(len(SIC_column_indices)):
                        this_SIC_column_indice = SIC_column_indices[j]
                        if this_SIC_column_indice not in column_indices:
                            column_indices.append(this_SIC_column_indice)
                        if len(column_indices) >= num_top_k_columns:
                            break
                    print("Padding with SIC\n")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))
                else:
                    print("No need SIC column padding")
                    print("column_indices {}: {}".format(table_names[table_idx],
                                          [column_names[table_idx][column_idx] for column_idx in column_indices]))

            else:
                print("Using SIC\n")
                sic_column_probs = sic_pred_results[table_idx]["column_probs"]
                column_indices = np.argsort(-np.array(sic_column_probs), kind="stable")[:num_top_k_columns].tolist()
                print("Generated by SIC\n")
                print("column_indices {}: {}".format(table_names[table_idx],
                                      [column_names[table_idx][column_idx] for column_idx in column_indices]))
            print("CCcc")
            print(len(column_names[table_idx]))
            print(len(column_indices))
            print("The End of this Table\n")
            random.shuffle(column_indices)

            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        # replace the old matched contents with the filtered matched contents
        data["matched_contents"] = filtered_matched_contents
        print("\n")
    return dataset


def filter_schema_filtered_and_rankingSL(dataset, schema_linking_file, sic, num_top_k_tables, num_top_k_columns, mode, t_padding, c_padding):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print(mode)
    print("t_padding {}".format(t_padding))
    print("c_padding {}".format(c_padding))
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        pred_column_names_dict = dict()

        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]

        # print(ground_truth_schema_linking)


        SLM_match_table_indices = []
        table_indices = []
        for table_idx, table_name in enumerate(table_names):
            try:
                pred_column_names = predicted_schemalinking[table_name]
                if pred_column_names == None:
                    continue
                else:
                    table_indices.append(table_idx)
                    SLM_match_table_indices.append(table_idx)
                    pred_column_names_dict[table_idx] = pred_column_names
            except:
                continue


        sic_pred_results = sic.predict(data)
        sic_table_probs = [pred_result["table_prob"] for pred_result in sic_pred_results]
        sic_table_indices = np.argsort(-np.array(sic_table_probs), kind="stable").tolist()

        if num_top_k_tables is not None:
            # print("Table padding")
            if "add" in mode:
                if len(table_indices) + t_padding < num_top_k_tables:
                    this_num_top_k_tables = len(table_indices) + t_padding
                else:
                    this_num_top_k_tables = num_top_k_tables
            else:
                this_num_top_k_tables = num_top_k_tables
            if len(table_indices) < this_num_top_k_tables:
                for i in range(len(sic_table_indices)):
                    this_sic_table_indice = sic_table_indices[i]
                    if this_sic_table_indice not in table_indices:
                        table_indices.append(this_sic_table_indice)
                    if len(table_indices)>=this_num_top_k_tables:
                        break
        # else:
        #     print("No table padding")


        for table_idx in table_indices:
            if table_idx in SLM_match_table_indices:
                pred_column_names = pred_column_names_dict[table_idx]
                column_indices = []
                if type(pred_column_names) == list:
                    for column_idx, column_name in enumerate(column_names[table_idx]):
                        try:
                            if column_name in pred_column_names:
                                column_indices.append(column_idx)
                        except:
                            continue
                # print("Linking by filteredSL\n")
                # print("{}: {}".format(table_names[table_idx],
                #                       [column_names[table_idx][column_idx] for column_idx in column_indices]))
                if num_top_k_columns is not None:

                    if "add" in mode:
                        if len(column_indices) + c_padding < num_top_k_columns:
                            this_num_top_k_columns = len(column_indices) + c_padding
                        else:
                            this_num_top_k_columns = num_top_k_columns
                    else:
                        this_num_top_k_columns = num_top_k_columns
                    if len(column_indices) < this_num_top_k_columns:
                        sic_column_probs = sic_pred_results[table_idx]["column_probs"]
                        sic_column_indices = np.argsort(-np.array(sic_column_probs), kind="stable").tolist()
                        for j in range(len(sic_column_indices)):
                            this_sic_column_indice = sic_column_indices[j]
                            if this_sic_column_indice not in column_indices:
                                column_indices.append(this_sic_column_indice)
                            if len(column_indices) >= this_num_top_k_columns:
                                break
                #         print("Padding with rankingSL\n")
                #         print("{}: {}".format(table_names[table_idx],
                #                               [column_names[table_idx][column_idx] for column_idx in column_indices]))
                # else:
                #     print("No column padding")
            else:
                sic_column_probs = sic_pred_results[table_idx]["column_probs"]

                # 这种情况下 t4c4_p1c2实际上是第二行好，但45c6_p2c3第一行好，并是目前最高值
                # 所以我加入了and num_top_k_columns>4的判断条件
                if "add" in mode and num_top_k_columns>4:
                    column_indices = np.argsort(-np.array(sic_column_probs), kind="stable")[:c_padding].tolist()
                else:
                    column_indices = np.argsort(-np.array(sic_column_probs), kind="stable")[:num_top_k_columns].tolist()
                # print("Generated by rankingSL\n")
                # print("{}: {}".format(table_names[table_idx],
                #                       [column_names[table_idx][column_idx] for column_idx in column_indices]))
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        # replace the old matched contents with the filtered matched contents
        data["matched_contents"] = filtered_matched_contents
        # print("\n")
    return dataset
def filter_schema_DTS(dataset, schema_linking_file, dataset_type):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        table_indices = []
        match_table_names = []
        pred_column_names_dict = dict()

        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]
        print(index)
        print(ground_truth_schema_linking)
        if dataset_type == "eval":
            predicted_schemalinking = schema_linking_results[index]
            for table_idx, table_name in enumerate(table_names):
                try:
                    pred_column_names = predicted_schemalinking[table_name]
                    if pred_column_names == None:
                        continue
                    else:
                        table_indices.append(table_idx)
                        match_table_names.append(table_name)
                        pred_column_names_dict[table_idx] = pred_column_names
                except:
                    continue

        elif dataset_type == "train":
            table_indices = ground_truth_table_indices


        if len(table_indices) == 0:
            table_indices = list(range(len(table_names)))

        for table_idx in table_indices:
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": column_names[table_idx],
                    "column_types": column_types[table_idx],
                    "column_comments": column_comments[table_idx],
                    "column_contents": column_contents[table_idx],
                    "pk_indicators": pk_indicators[table_idx]
                }
            )
            # extract matched contents of remained columns
            for column_name in column_names[table_idx]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        # replace the old matched contents with the filtered matched contents
        data["matched_contents"] = filtered_matched_contents
        print("\n")
    return dataset



def filter_schema_onlySLM_DTSSQL_Gold(dataset,  dataset_type):
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        for table_idx in table_indices:

            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": column_names[table_idx],
                    "column_types": column_types[table_idx],
                    "column_comments": column_comments[table_idx],
                    "column_contents": column_contents[table_idx],
                    "pk_indicators": pk_indicators[table_idx]
                }
            )
            # extract matched contents of remained columns
            for column_name in column_names[table_idx]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
    return dataset
def filter_schema_onlySLM_TASQL_Gold(dataset,  dataset_type):

    for index, data in enumerate(dataset):

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]


        table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]



        for table_idx in table_indices:
            column_indices = [column_idx for column_idx, column_label in
                              enumerate(data["column_labels"][table_idx])
                              if column_label == 1]


            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
    return dataset

def filter_schema_onlySLM_DTSSQL(dataset, schema_linking_file,  dataset_type):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        table_indices = []
        match_table_names = []
        pred_column_names_dict = dict()

        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]

        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]

        for table_idx, table_name in enumerate(table_names):
            try:
                pred_column_names = predicted_schemalinking[table_name]
                if pred_column_names == None:
                    continue
                else:
                    table_indices.append(table_idx)
                    match_table_names.append(table_name)
                    pred_column_names_dict[table_idx] = pred_column_names
            except:
                continue


        if dataset_type == "train":
            for g_t_idx in ground_truth_table_indices:
                if g_t_idx not in table_indices:
                    table_indices.append(g_t_idx)
            random.shuffle(table_indices)

        for table_idx in table_indices:
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": column_names[table_idx],
                    "column_types": column_types[table_idx],
                    "column_comments": column_comments[table_idx],
                    "column_contents": column_contents[table_idx],
                    "pk_indicators": pk_indicators[table_idx]
                }
            )
            # extract matched contents of remained columns
            for column_name in column_names[table_idx]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
    return dataset
def filter_schema_onlySLM_TASQL(dataset, schema_linking_file,  dataset_type):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        table_indices = []
        match_table_names = []
        pred_column_names_dict = dict()

        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]

        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]

        for table_idx, table_name in enumerate(table_names):
            try:
                pred_column_names = predicted_schemalinking[table_name]
                if pred_column_names == None:
                    continue
                else:
                    table_indices.append(table_idx)
                    match_table_names.append(table_name)
                    pred_column_names_dict[table_idx] = pred_column_names
            except:
                continue


        if dataset_type == "train":
            for g_t_idx in ground_truth_table_indices:
                if g_t_idx not in table_indices:
                    table_indices.append(g_t_idx)

            random.shuffle(table_indices)

        for table_idx in table_indices:
            if table_idx in pred_column_names_dict:
                pred_column_names = pred_column_names_dict[table_idx]
            else:
                pred_column_names = None
            column_indices = []
            if type(pred_column_names) == list:
                for column_idx, column_name in enumerate(column_names[table_idx]):
                    try:
                        if column_name in pred_column_names:
                            column_indices.append(column_idx)
                    except:
                        continue
                if dataset_type == "train":
                    ground_truth_column_indices = [column_idx for column_idx, column_label in
                                                   enumerate(data["column_labels"][table_idx])
                                                   if column_label == 1]
                    for g_c_idx in ground_truth_column_indices:
                        if g_c_idx not in column_indices:
                            column_indices.append(g_c_idx)
                    random.shuffle(column_indices)
            else:
                if dataset_type == "train":
                    column_indices = [column_idx for column_idx, column_label in
                                                   enumerate(data["column_labels"][table_idx])
                                                   if column_label == 1]
                else:
                    continue


            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
    return dataset

def filter_schema_onlySLM_rankGPT(dataset, schema_linking_file):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        table_indices = []
        match_table_names = []
        pred_column_names_dict = dict()

        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]

        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]
        for table_idx, table_name in enumerate(table_names):
            try:
                pred_column_names = predicted_schemalinking[table_name]
                if pred_column_names == None:
                    continue
                else:
                    table_indices.append(table_idx)
                    match_table_names.append(table_name)
                    pred_column_names_dict[table_idx] = pred_column_names
            except:
                continue
        for table_idx in table_indices:
            if table_idx in pred_column_names_dict:
                pred_column_names = pred_column_names_dict[table_idx]
            else:
                pred_column_names = column_names[table_idx]
            column_indices = []
            if type(pred_column_names) == list:
                for column_idx, column_name in enumerate(column_names[table_idx]):
                    try:
                        if column_name in pred_column_names:
                            column_indices.append(column_idx)
                    except:
                        continue
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        # replace the old matched contents with the filtered matched contents
        data["matched_contents"] = filtered_matched_contents





        rank_filtered_schema = dict()
        rank_filtered_schema["schema_items"] = []
        rank_filtered_schema["foreign_keys"] = []
        rank_table_indices = [rank_table_idx for rank_table_idx, rank_table_label in enumerate(data["table_labels"]) if
                         rank_table_label == 1]
        if len(rank_table_indices) < 6:
            unused_rank_table_indices = [rank_table_idx for rank_table_idx, rank_table_label in enumerate(data["table_labels"]) if
                                    rank_table_label == 0]
            rank_table_indices += random.sample(unused_rank_table_indices,
                                           min(len(unused_rank_table_indices), 6 - len(rank_table_indices)))
        random.shuffle(rank_table_indices)


        for rank_table_idx in rank_table_indices:
            rank_column_indices = [rank_column_idx for rank_column_idx, rank_column_label in enumerate(data["column_labels"][rank_table_idx])
                              if rank_column_label == 1]
            if len(rank_column_indices) < 10:
                unused_rank_column_indices = [rank_column_idx for rank_column_idx, rank_column_label in
                                         enumerate(data["column_labels"][rank_table_idx]) if rank_column_label == 0]
                rank_column_indices += random.sample(unused_rank_column_indices, min(len(unused_rank_column_indices),
                                                                           10 - len(rank_column_indices)))
            random.shuffle(rank_column_indices)

            rank_filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[rank_table_idx],
                    "table_comment": table_comments[rank_table_idx],
                    "column_names": [column_names[rank_table_idx][rank_column_idx] for rank_column_idx in rank_column_indices],
                    "column_types": [column_types[rank_table_idx][rank_column_idx] for rank_column_idx in rank_column_indices],
                    "column_comments": [column_comments[rank_table_idx][rank_column_idx] for rank_column_idx in rank_column_indices],
                    "column_contents": [column_contents[rank_table_idx][rank_column_idx] for rank_column_idx in rank_column_indices],
                    "pk_indicators": [pk_indicators[rank_table_idx][rank_column_idx] for rank_column_idx in rank_column_indices]
                }
            )
        rank_filtered_table_names = [table_names[rank_table_idx] for rank_table_idx in rank_table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in rank_filtered_table_names and target_table in rank_filtered_table_names:
                rank_filtered_schema["foreign_keys"].append(foreign_key)
        data["schema"] = rank_filtered_schema
    return dataset


def filter_schema_onlySLM(dataset, schema_linking_file):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        table_indices = []
        match_table_names = []
        pred_column_names_dict = dict()

        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]

        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]
        for table_idx, table_name in enumerate(table_names):
            try:
                pred_column_names = predicted_schemalinking[table_name]
                if pred_column_names == None:
                    continue
                else:
                    table_indices.append(table_idx)
                    match_table_names.append(table_name)
                    pred_column_names_dict[table_idx] = pred_column_names
            except:
                continue
        for table_idx in table_indices:
            if table_idx in pred_column_names_dict:
                pred_column_names = pred_column_names_dict[table_idx]
            else:
                pred_column_names = column_names[table_idx]
            column_indices = []
            if type(pred_column_names) == list:
                for column_idx, column_name in enumerate(column_names[table_idx]):
                    try:
                        if column_name in pred_column_names:
                            column_indices.append(column_idx)
                    except:
                        continue
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
        
    return dataset



def filter_schema_full(dataset):
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        table_indices = range(len(table_names))

        matched_entities = dict()

        for table_idx in table_indices:
            column_indices = range(len(column_names[table_idx]))

            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            matched_entities[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in column_indices]
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
        data["matched_entities"] = matched_entities
    return dataset

def filter_schema_ideal(dataset):
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                         table_label == 1]

        matched_entities = dict()

        for table_idx, table_name in enumerate(table_names):
            if table_idx not in table_indices:
                matched_entities[table_name] = None

        for table_idx in table_indices:
            column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                              if column_label == 1]
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            matched_entities[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in column_indices]
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents

        data["matched_entities"] = matched_entities
    return dataset


def filter_schema_codesStyle_ideal(dataset, dataset_type, sic, num_top_k_tables=6, num_top_k_columns=10):
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        sic_pred_results = sic.predict(data)
        if dataset_type == "eval":
            table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                             table_label == 1]
            if len(table_indices) < num_top_k_tables:
                sic_table_probs = [pred_result["table_prob"] for pred_result in sic_pred_results]
                sic_table_indices = np.argsort(-np.array(sic_table_probs),
                                               kind="stable").tolist()
                for i in range(len(sic_table_indices)):
                    this_sic_table_indice = sic_table_indices[i]
                    if this_sic_table_indice not in table_indices:
                        table_indices.append(this_sic_table_indice)
                    if len(table_indices)>=num_top_k_tables:
                        break

        for table_idx in table_indices:
            if dataset_type == "eval":
                column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                  if column_label == 1]
                if len(column_indices) < num_top_k_columns:
                    sic_column_probs = sic_pred_results[table_idx]["column_probs"]
                    sic_column_indices = np.argsort(-np.array(sic_column_probs),
                                                    kind="stable").tolist()  # [:num_top_k_columns]
                    for j in range(len(sic_column_indices)):
                        this_sic_column_indice = sic_column_indices[j]
                        if this_sic_column_indice not in column_indices:
                            column_indices.append(this_sic_column_indice)
                        if len(column_indices) >= num_top_k_columns:
                            break
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents

    return dataset



def filter_and_write_schema_ideal(dataset, target_dataset_path):
    print("filtering schema items for the dataset")
    target_dataset_path = target_dataset_path.replace(".json", "")
    schema_linking_results_path = target_dataset_path + "_results.json"
    new_schema_linking_results = []
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        this_schema_linking = dict()
        table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                         table_label == 1]

        matched_entities = dict()

        for table_idx, table_name in enumerate(table_names):
            if table_idx not in table_indices:
                this_schema_linking[table_name] =None

        for table_idx in table_indices:
            column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                              if column_label == 1]
            this_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in column_indices]
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            matched_entities[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in column_indices]
            # if len(matched_columnss) == 0:
            #     matched_entities[table_names[table_idx]] = "No need Column"
            # else:
            #     matched_entities[table_names[table_idx]] = matched_columnss
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents

        data["matched_entities"] = matched_entities
        new_schema_linking_results.append(this_schema_linking)
    json.dump(new_schema_linking_results, open(schema_linking_results_path, 'w'), indent=4)
    return dataset

def filter_and_write_schema_DTS(dataset, schema_linking_file, target_dataset_path):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")
    target_dataset_path = target_dataset_path.replace(".json", "")
    schema_linking_results_path = target_dataset_path + "_results.json"
    new_schema_linking_results = []
    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]
        print(index)
        print(ground_truth_schema_linking)
        print(predicted_schemalinking)

        table_indices = []
        this_schema_linking = dict()
        for table_idx, table_name in enumerate(table_names):
            try:
                pred_column_names = predicted_schemalinking[table_name]
                print(pred_column_names)
                if pred_column_names == None:
                    this_schema_linking[table_name] = None
                else:
                    table_indices.append(table_idx)
                    this_schema_linking[table_name] = column_names[table_idx]
            except:
                continue
        print(table_indices)
        # if len(table_indices) == 0:
        #     table_indices = list(range(len(table_names)))
        #     for kkk in table_indices:
        #         this_schema_linking[table_names[kkk]] = column_names[kkk]
        for table_idx in table_indices:
            print("Linking by filteredSL\n")
            print("{}: {}".format(table_names[table_idx], column_names[table_idx]))
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": column_names[table_idx],
                    "column_types": column_types[table_idx],
                    "column_comments": column_comments[table_idx],
                    "column_contents": column_contents[table_idx],
                    "pk_indicators": pk_indicators[table_idx]
                }
            )
            # extract matched contents of remained columns
            for column_name in column_names[table_idx]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        # replace the old matched contents with the filtered matched contents
        data["matched_contents"] = filtered_matched_contents
        new_schema_linking_results.append(this_schema_linking)
        print("\n")
    json.dump(new_schema_linking_results, open(schema_linking_results_path, 'w'), indent=4)
    return dataset


def filter_and_write_schema_ranking_cut(dataset, dataset_type, sic, num_top_k_tables=6, num_top_k_columns=10,
                                    target_dataset_path=None, cut=0.5):
    print("filtering schema items for the dataset")
    target_dataset_path = target_dataset_path.replace(".json", "")
    schema_linking_results_path = target_dataset_path + "_results.json"
    new_schema_linking_results = []

    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        this_schema_linking_results = dict()
        if dataset_type == "eval":
            # predict scores for each tables and columns
            pred_results = sic.predict(data)
            # remain top_k1 tables for each database and top_k2 columns for each remained table
            table_probs = [pred_result["table_prob"] for pred_result in pred_results]
            delete_table_index = set()
            for kkk in range(len(table_probs)):
                this_prob = table_probs[kkk]
                if this_prob < cut:
                    delete_table_index.add(kkk)
            if len(delete_table_index) == len(table_names):
                delete_table_index = set()
            base_table_index = np.argsort(-np.array(table_probs), kind="stable").tolist()
            table_indices = []
            for this_table_index in base_table_index:
                if this_table_index not in delete_table_index:
                    table_indices.append(this_table_index)
            # table_indices = table_indices[:num_top_k_tables]
            # table_indices = np.argsort(-np.array(table_probs), kind="stable")[:num_top_k_tables].tolist()


        for table_idx in range(len(table_names)):
            if table_idx in table_indices:
                if dataset_type == "eval":
                    column_probs = pred_results[table_idx]["column_probs"]
                    delete_column_index = set()
                    for kkk in range(len(column_probs)):
                        this_prob = column_probs[kkk]
                        if this_prob < cut:
                            delete_column_index.add(kkk)
                    if len(delete_column_index) == len(column_names):
                        delete_column_index = set()
                    base_column_index = np.argsort(-np.array(column_probs), kind="stable").tolist()
                    column_indices = []
                    for this_column_index in base_column_index:
                        if this_column_index not in delete_column_index:
                            column_indices.append(this_column_index)
                    # column_indices = column_indices[:num_top_k_columns]
                    # column_indices = np.argsort(-np.array(column_probs), kind="stable")[:num_top_k_columns].tolist()

                this_schema_linking_results[table_names[table_idx]] = [column_names[table_idx][column_idx] for
                                                                       column_idx in
                                                                       column_indices]
                filtered_schema["schema_items"].append(
                    {
                        "table_name": table_names[table_idx],
                        "table_comment": table_comments[table_idx],
                        "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                        "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                        "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                        "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                        "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                    }
                )
                for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                    tc_name = "{}.{}".format(table_names[table_idx], column_name)
                    if tc_name in data["matched_contents"]:
                        filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
            else:
                this_schema_linking_results[table_names[table_idx]] = None
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
        new_schema_linking_results.append(this_schema_linking_results)

    json.dump(new_schema_linking_results, open(schema_linking_results_path, 'w'), indent=4)
    return dataset
def filter_and_write_schema_ranking(dataset, dataset_type, sic, num_top_k_tables=6, num_top_k_columns=10,
                                    target_dataset_path=None):
    print("filtering schema items for the dataset")
    target_dataset_path = target_dataset_path.replace(".json", "")
    schema_linking_results_path = target_dataset_path + "_results.json"
    schema_linking_probs_path = target_dataset_path + "_probs.json"
    new_schema_linking_results = []
    new_schema_linking_probs = []

    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        this_schema_linking_results = dict()
        this_schema_linking_probs = dict()
        if dataset_type == "eval":
            # predict scores for each tables and columns
            pred_results = sic.predict(data)
            # remain top_k1 tables for each database and top_k2 columns for each remained table
            table_probs = [pred_result["table_prob"] for pred_result in pred_results]
            table_indices = np.argsort(-np.array(table_probs), kind="stable")[:num_top_k_tables].tolist()
        elif dataset_type == "train":
            # table_label应该是根据ground truth sql标注的
            table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                             table_label == 1]
            # print("table_indices")
            # print(table_indices)
            if len(table_indices) < num_top_k_tables:
                unused_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                                        table_label == 0]
                table_indices += random.sample(unused_table_indices,
                                               min(len(unused_table_indices), num_top_k_tables - len(table_indices)))
            # print("table_indices")
            # print(table_indices)
            random.shuffle(table_indices)

            pred_results = sic.predict(data)
            table_probs = [pred_result["table_prob"] for pred_result in pred_results]
            # remain top_k1 tables for each database and top_k2 columns for each remained table
        tableIndex_ranking_list = np.argsort(-np.array(table_probs), kind="stable").tolist()
        for table_idx in tableIndex_ranking_list:
            this_schema_linking_probs[table_names[table_idx]] = dict()
            this_schema_linking_probs[table_names[table_idx]]["table_prob"] = table_probs[table_idx]
            column_probs = pred_results[table_idx]["column_probs"]
            this_column_probs_list = []
            this_column_probs_dict = dict()
            columnIndex_ranking_list = np.argsort(-np.array(column_probs), kind="stable").tolist()
            for column_idx in columnIndex_ranking_list:
                this_column_probs_list.append((column_names[table_idx][column_idx], column_probs[column_idx]))
                this_column_probs_dict[column_names[table_idx][column_idx]] = column_probs[column_idx]
            this_schema_linking_probs[table_names[table_idx]]["column_probs_list"] = this_column_probs_list
            this_schema_linking_probs[table_names[table_idx]]["column_probs_dict"] = this_column_probs_dict


        for table_idx in range(len(table_names)):
            if table_idx in table_indices:
                if dataset_type == "eval":
                    column_probs = pred_results[table_idx]["column_probs"]
                    column_indices = np.argsort(-np.array(column_probs), kind="stable")[:num_top_k_columns].tolist()
                elif dataset_type == "train":
                    column_indices = [column_idx for column_idx, column_label in
                                      enumerate(data["column_labels"][table_idx])
                                      if column_label == 1]
                    if len(column_indices) < num_top_k_columns:
                        unused_column_indices = [column_idx for column_idx, column_label in
                                                 enumerate(data["column_labels"][table_idx]) if column_label == 0]
                        column_indices += random.sample(unused_column_indices, min(len(unused_column_indices),
                                                                                   num_top_k_columns - len(
                                                                                       column_indices)))
                    random.shuffle(column_indices)

                this_schema_linking_results[table_names[table_idx]] = [column_names[table_idx][column_idx] for
                                                                       column_idx in
                                                                       column_indices]
                filtered_schema["schema_items"].append(
                    {
                        "table_name": table_names[table_idx],
                        "table_comment": table_comments[table_idx],
                        "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                        "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                        "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                        "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                        "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                    }
                )
                for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                    tc_name = "{}.{}".format(table_names[table_idx], column_name)
                    if tc_name in data["matched_contents"]:
                        filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
            else:
                this_schema_linking_results[table_names[table_idx]] = None
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
        new_schema_linking_results.append(this_schema_linking_results)
        new_schema_linking_probs.append(this_schema_linking_probs)

    json.dump(new_schema_linking_results, open(schema_linking_results_path, 'w'), indent=4)
    json.dump(new_schema_linking_probs, open(schema_linking_probs_path, 'w'), indent=4)
    return dataset



def filter_and_write_schema_filtered_and_rankingSL(dataset, schema_linking_file, sic, num_top_k_tables, num_top_k_columns,
                                                   mode, t_padding, c_padding, target_dataset_path=None):
    if schema_linking_file is not None:
        schema_linking_results = json.load(open(schema_linking_file))
    else:
        schema_linking_results = None
    print(mode)
    print("t_padding {}".format(t_padding))
    print("c_padding {}".format(c_padding))
    print("filtering schema items for the dataset")

    target_dataset_path = target_dataset_path.replace(".json", "")
    schema_linking_results_path = target_dataset_path + "_results.json"
    new_schema_linking_results = []

    for index, data in enumerate(dataset):

        predicted_schemalinking = schema_linking_results[index]

        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        this_schema_linking_results = dict()

        pred_column_names_dict = dict()

        ground_truth_schema_linking = dict()
        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        for table_idx in ground_truth_table_indices:
            ground_truth_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                           if column_label == 1]
            ground_truth_schema_linking[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in ground_truth_column_indices]

        # print(ground_truth_schema_linking)


        SLM_match_table_indices = []
        table_indices = []
        for table_idx, table_name in enumerate(table_names):
            try:
                pred_column_names = predicted_schemalinking[table_name]
                if pred_column_names == None:
                    continue
                else:
                    table_indices.append(table_idx)
                    SLM_match_table_indices.append(table_idx)
                    pred_column_names_dict[table_idx] = pred_column_names
            except:
                continue


        sic_pred_results = sic.predict(data)
        sic_table_probs = [pred_result["table_prob"] for pred_result in sic_pred_results]
        sic_table_indices = np.argsort(-np.array(sic_table_probs), kind="stable").tolist()

        if num_top_k_tables is not None:
            # print("Table padding")
            if "add" in mode:
                if len(table_indices) + t_padding < num_top_k_tables:
                    this_num_top_k_tables = len(table_indices) + t_padding
                else:
                    this_num_top_k_tables = num_top_k_tables
            else:
                this_num_top_k_tables = num_top_k_tables
            if len(table_indices) < this_num_top_k_tables:
                for i in range(len(sic_table_indices)):
                    this_sic_table_indice = sic_table_indices[i]
                    if this_sic_table_indice not in table_indices:
                        table_indices.append(this_sic_table_indice)
                    if len(table_indices)>=this_num_top_k_tables:
                        break
        # else:
        #     print("No table padding")


        for table_idx in table_indices:
            if table_idx in SLM_match_table_indices:
                pred_column_names = pred_column_names_dict[table_idx]
                column_indices = []
                if type(pred_column_names) == list:
                    for column_idx, column_name in enumerate(column_names[table_idx]):
                        try:
                            if column_name in pred_column_names:
                                column_indices.append(column_idx)
                        except:
                            continue
                # print("Linking by filteredSL\n")
                # print("{}: {}".format(table_names[table_idx],
                #                       [column_names[table_idx][column_idx] for column_idx in column_indices]))
                if num_top_k_columns is not None:

                    if "add" in mode:
                        if len(column_indices) + c_padding < num_top_k_columns:
                            this_num_top_k_columns = len(column_indices) + c_padding
                        else:
                            this_num_top_k_columns = num_top_k_columns
                    else:
                        this_num_top_k_columns = num_top_k_columns
                    if len(column_indices) < this_num_top_k_columns:
                        sic_column_probs = sic_pred_results[table_idx]["column_probs"]
                        sic_column_indices = np.argsort(-np.array(sic_column_probs), kind="stable").tolist()
                        for j in range(len(sic_column_indices)):
                            this_sic_column_indice = sic_column_indices[j]
                            if this_sic_column_indice not in column_indices:
                                column_indices.append(this_sic_column_indice)
                            if len(column_indices) >= this_num_top_k_columns:
                                break
                #         print("Padding with rankingSL\n")
                #         print("{}: {}".format(table_names[table_idx],
                #                               [column_names[table_idx][column_idx] for column_idx in column_indices]))
                # else:
                #     print("No column padding")
            else:
                sic_column_probs = sic_pred_results[table_idx]["column_probs"]
                # 这种情况下 t4c4_p1c2实际上是第二行好，但45c6_p2c3第一行好，并是目前最高值
                # 所以我加入了and num_top_k_columns>4的判断条件
                if "add" in mode and num_top_k_columns>4:
                    column_indices = np.argsort(-np.array(sic_column_probs), kind="stable")[:c_padding].tolist()
                else:
                    column_indices = np.argsort(-np.array(sic_column_probs), kind="stable")[:num_top_k_columns].tolist()
                # print("Generated by rankingSL\n")
                # print("{}: {}".format(table_names[table_idx],
                #                       [column_names[table_idx][column_idx] for column_idx in column_indices]))

            this_schema_linking_results[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in
                                                           column_indices]
            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        # replace the old matched contents with the filtered matched contents
        data["matched_contents"] = filtered_matched_contents

        new_schema_linking_results.append(this_schema_linking_results)
        # print("\n")
    json.dump(new_schema_linking_results, open(schema_linking_results_path, 'w'), indent=4)
    return dataset



def filter_schema_codesStyle_ranking_cut_KSL(dataset, dataset_type, sic, cut=0.5):
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        pred_results = sic.predict(data)
        table_probs = [pred_result["table_prob"] for pred_result in pred_results]

        table_indices = []
        for t_index in range(len(table_probs)):
            this_prob = table_probs[t_index]
            if this_prob >= cut:
                table_indices.append(t_index)

        if dataset_type == "eval":
            if len(table_indices) == 0:
                table_indices = list(range(len(table_names)))
        if dataset_type == "train":
            table_indices_ground_truth = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                             table_label == 1]
            for t_index in table_indices_ground_truth:
                if t_index not in table_indices:
                    table_indices.append(t_index)
            random.shuffle(table_indices)

        matched_entities = dict()
        for table_idx in table_indices:
            column_probs = pred_results[table_idx]["column_probs"]
            column_indices = []
            for c_index in range(len(column_probs)):
                this_prob = column_probs[c_index]
                if this_prob >= cut:
                    column_indices.append(c_index)
            if dataset_type == "train":
                column_indices_ground_truth = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                  if column_label == 1]
                for c_index in column_indices_ground_truth:
                    if c_index not in column_indices:
                        column_indices.append(c_index)
                random.shuffle(column_indices)


                if table_idx in table_indices_ground_truth:
                    matched_entities[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in
                                                                column_indices_ground_truth]

            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )


            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents
        data["matched_entities"] = matched_entities

    return dataset




def filter_schema_codesStyle_ranking_cut(dataset, dataset_type, sic, num_top_k_tables=6, num_top_k_columns=10, cut=0.5):
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        if dataset_type == "eval":
            # predict scores for each tables and columns
            pred_results = sic.predict(data)
            # remain top_k1 tables for each database and top_k2 columns for each remained table
            table_probs = [pred_result["table_prob"] for pred_result in pred_results]
            delete_table_index = set()
            for kkk in range(len(table_probs)):
                this_prob = table_probs[kkk]
                if this_prob < cut:
                    delete_table_index.add(kkk)
            if len(delete_table_index) == len(table_names):
                delete_table_index = set()
            base_table_index = np.argsort(-np.array(table_probs), kind="stable").tolist()
            table_indices = []
            for this_table_index in base_table_index:
                if this_table_index not in delete_table_index:
                    table_indices.append(this_table_index)
            table_indices = table_indices[:num_top_k_tables]
            # table_indices = np.argsort(-np.array(table_probs), kind="stable")[:num_top_k_tables].tolist()

        elif dataset_type == "train":
            # table_label应该是根据ground truth sql标注的
            table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                             table_label == 1]
            # print("table_indices")
            # print(table_indices)
            if len(table_indices) < num_top_k_tables:
                unused_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                                        table_label == 0]
                table_indices += random.sample(unused_table_indices,
                                               min(len(unused_table_indices), num_top_k_tables - len(table_indices)))
            # print("table_indices")
            # print(table_indices)
            random.shuffle(table_indices)

        for table_idx in table_indices:
            if dataset_type == "eval":
                column_probs = pred_results[table_idx]["column_probs"]
                delete_column_index = set()
                for kkk in range(len(column_probs)):
                    this_prob = column_probs[kkk]
                    if this_prob < cut:
                        delete_column_index.add(kkk)
                if len(delete_column_index) == len(column_names):
                    delete_column_index = set()
                base_column_index = np.argsort(-np.array(column_probs), kind="stable").tolist()
                column_indices = []
                for this_column_index in base_column_index:
                    if this_column_index not in delete_column_index:
                        column_indices.append(this_column_index)
                column_indices = column_indices[:num_top_k_columns]
                # column_indices = np.argsort(-np.array(column_probs), kind="stable")[:num_top_k_columns].tolist()
            elif dataset_type == "train":
                column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                  if column_label == 1]
                # print("column_indices")
                # print(column_indices)
                if len(column_indices) < num_top_k_columns:
                    unused_column_indices = [column_idx for column_idx, column_label in
                                             enumerate(data["column_labels"][table_idx]) if column_label == 0]
                    column_indices += random.sample(unused_column_indices, min(len(unused_column_indices),
                                                                               num_top_k_columns - len(column_indices)))
                random.shuffle(column_indices)

            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # for column_idx in column_indices:
            #     print(column_names[table_idx][column_idx])
            # print()
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]

        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents

    return dataset



def filter_schema_codesStyle_KSL_filterByProb(dataset, dataset_type, recall_perT_file, num_top_k_tables=6, num_top_k_columns=10):
    print("filtering schema items for the dataset")

    recall_perT_file = json.load(open(recall_perT_file))
    for index in tqdm(range(len(dataset))):
        data = dataset[index]
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        instance_perT = recall_perT_file[index]
        table_recall_name = instance_perT["t_ranking_name_perT"]

        selected_table_names = table_recall_name[:num_top_k_tables]
        table_indices = []
        for t_index in range(len(table_names)):
            t_name = table_names[t_index]
            if t_name in selected_table_names:
                table_indices.append(t_index)
        matched_entities = dict()

        if dataset_type == "train":
            gold_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                             table_label == 1]
            for gold_table_idx in gold_table_indices:
                if gold_table_idx not in table_indices:
                    table_indices.append(gold_table_idx)
            random.shuffle(table_indices)
            for t_index in table_indices:
                matched_entities[table_names[t_index]] = None

        for table_idx in table_indices:
            t_name = table_names[table_idx]
            column_recall_name = instance_perT[t_name]["c_ranking_name_perT"]
            selected_column_names = column_recall_name[:num_top_k_columns]
            this_column_names = column_names[table_idx]
            column_indices = []
            for c_index in range(len(this_column_names)):
                c_name = this_column_names[c_index]
                if c_name in selected_column_names:
                    column_indices.append(c_index)

            if dataset_type == "train":
                gold_column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                  if column_label == 1]
                for gold_column_idx in gold_column_indices:
                    if gold_column_idx not in column_indices:
                        column_indices.append(gold_column_idx)
                random.shuffle(column_indices)

                if table_idx in gold_table_indices:
                    gold_column_names = []
                    for c_idx in column_indices:
                        if c_idx in gold_column_indices:
                            gold_column_names.append(column_names[table_idx][c_idx])
                    matched_entities[table_names[table_idx]] = gold_column_names



            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]

        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents

        data["matched_entities"] = matched_entities

    return dataset

def filter_schema_codesStyle(dataset, dataset_type, sic, num_top_k_tables=6, num_top_k_columns=10):
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        if dataset_type == "eval":
            # predict scores for each tables and columns
            pred_results = sic.predict(data)
            # remain top_k1 tables for each database and top_k2 columns for each remained table
            table_probs = [pred_result["table_prob"] for pred_result in pred_results]
            table_indices = np.argsort(-np.array(table_probs), kind="stable")[:num_top_k_tables].tolist()
        elif dataset_type == "train":
            # table_label应该是根据ground truth sql标注的
            table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                             table_label == 1]
            # print("table_indices")
            # print(table_indices)
            if len(table_indices) < num_top_k_tables:
                unused_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                                        table_label == 0]
                table_indices += random.sample(unused_table_indices,
                                               min(len(unused_table_indices), num_top_k_tables - len(table_indices)))
            # print("table_indices")
            # print(table_indices)
            random.shuffle(table_indices)

        for table_idx in table_indices:
            if dataset_type == "eval":
                column_probs = pred_results[table_idx]["column_probs"]
                column_indices = np.argsort(-np.array(column_probs), kind="stable")[:num_top_k_columns].tolist()
            elif dataset_type == "train":
                column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                                  if column_label == 1]
                # print("column_indices")
                # print(column_indices)
                if len(column_indices) < num_top_k_columns:
                    unused_column_indices = [column_idx for column_idx, column_label in
                                             enumerate(data["column_labels"][table_idx]) if column_label == 0]
                    column_indices += random.sample(unused_column_indices, min(len(unused_column_indices),
                                                                               num_top_k_columns - len(column_indices)))
                random.shuffle(column_indices)

            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            # for column_idx in column_indices:
            #     print(column_names[table_idx][column_idx])
            # print()
            # extract matched contents of remained columns
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]
                # 5.12仔细研究一下，这个matched_contents的逻辑，看起来是为了匹配question中的关键词，并不是单纯寻找matching column，
                # 回顾论文里的以及github里面的信息，看看我们自己

        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents

    return dataset


def filter_schema_rankingSL(dataset, sic, num_top_k_tables=6, num_top_k_columns=10):
    print("filtering schema items for the dataset")
    for index, data in enumerate(dataset):
        filtered_schema = dict()
        filtered_matched_contents = dict()
        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        pred_results = sic.predict(data)
        table_probs = [pred_result["table_prob"] for pred_result in pred_results]
        table_indices = np.argsort(-np.array(table_probs), kind="stable")[:num_top_k_tables].tolist()

        for table_idx in table_indices:
            column_probs = pred_results[table_idx]["column_probs"]
            column_indices = np.argsort(-np.array(column_probs), kind="stable")[:num_top_k_columns].tolist()

            filtered_schema["schema_items"].append(
                {
                    "table_name": table_names[table_idx],
                    "table_comment": table_comments[table_idx],
                    "column_names": [column_names[table_idx][column_idx] for column_idx in column_indices],
                    "column_types": [column_types[table_idx][column_idx] for column_idx in column_indices],
                    "column_comments": [column_comments[table_idx][column_idx] for column_idx in column_indices],
                    "column_contents": [column_contents[table_idx][column_idx] for column_idx in column_indices],
                    "pk_indicators": [pk_indicators[table_idx][column_idx] for column_idx in column_indices]
                }
            )
            for column_name in [column_names[table_idx][column_idx] for column_idx in column_indices]:
                tc_name = "{}.{}".format(table_names[table_idx], column_name)
                if tc_name in data["matched_contents"]:
                    filtered_matched_contents[tc_name] = data["matched_contents"][tc_name]

        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)

        data["filtered_schema"] = filtered_schema
        if index < 2:
            print(filtered_schema)
        data["matched_contents"] = filtered_matched_contents

    return dataset

import os
import torch
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import math
import numpy as np

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning schema item classifier.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev_filepath', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)
    opt = parser.parse_args()
    return opt

def clean_pred_columns(input_list):
    new_list = []
    for instance in input_list:
        instance = instance.replace("`", "")
        new_list.append(instance)
    return new_list


def calculate_precision_recall(ground_truth, predicted):

    ground_truth_set = set(ground_truth)
    predicted_set = set(predicted)

    intersection = ground_truth_set.intersection(predicted_set)
    precision = len(intersection) / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = len(intersection) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def calculate_precision_recall_debiased(ground_truth, predicted):

    ground_truth_set = set(ground_truth)
    predicted_set = set(predicted)

    intersection = ground_truth_set.intersection(predicted_set)

    if len(intersection) == len(ground_truth_set):
        recall = len(intersection) / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
        precision = len(intersection) / len(predicted_set) if len(predicted_set) > 0 else 0
    else:
        recall = 0
        precision = 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

def _eval(dev_file, data_file, dev_file_name):

    (table_labels_ground_truth, column_labels_ground_truth,
     total_recall_table, total_recall_column, total_precision_table,
     total_precision_column, total_f1_column, total_f1_table,
     total_auc_column, total_auc_table, total_Missing_table,
     total_Missing_column, total_Redundancy_table, total_Redundancy_column) =\
        ([], [], [], [], [], [], [], [], [], [], [], [], [], [])

    (total_debiased_recall_table, total_debiased_recall_column, total_debiased_precision_table,
    total_debiased_precision_column, total_debiased_f1_column, total_debiased_f1_table) = ([], [], [], [], [], [])

    table_num, column_num, table_num_ground_truth, column_num_ground_truth, eval_record = [], [], [], [], []

    total_debiased_f1_table_1 = []
    total_debiased_f1_table_2 = []
    total_debiased_f1_table_geq3 = []

    # for index in tqdm(range(500)):
    for index in tqdm(range(len(dev_file))):
        this_eval_record = dict()
        data = dev_file[index]
        predicted_schema_linking = data_file[index]

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"]) if
                         table_label == 1]
        matched_entities = dict()
        predicted_entities = dict()
        for table_idx, table_name in enumerate(table_names):
            predicted_entities[table_name] = None
            if table_idx not in table_indices:
                matched_entities[table_name] = None
        ground_truth_table_indices = []
        ground_truth_column_indices = []
        for table_idx in table_indices:
            column_indices = [column_idx for column_idx, column_label in enumerate(data["column_labels"][table_idx])
                              if column_label == 1]
            ground_truth_table_indices.append(str(table_idx))
            for column_idx in column_indices:
                ground_truth_column_indices.append(str(table_idx) + "." + str(column_idx))

            matched_entities[table_names[table_idx]] = [column_names[table_idx][column_idx] for column_idx in
                                                        column_indices]

        this_ground_truth_table = data["table_labels"]
        table_num_ground_truth.append(sum(this_ground_truth_table))
        table_labels_ground_truth.extend(this_ground_truth_table)
        this_ground_truth_column = []

        ground_truth_table_indices = []
        ground_truth_column_indices = []

        for table_idx in range(len(table_names)):
            if this_ground_truth_table[table_idx] == 1:
                column_linking_in_this_table = data["column_labels"][table_idx]
                this_ground_truth_column.extend(column_linking_in_this_table)

                ground_truth_table_indices.append(str(table_idx))
                for column_idx in range(len(column_linking_in_this_table)):
                    if column_linking_in_this_table[column_idx] == 1:
                        ground_truth_column_indices.append(str(table_idx) + "." + str(column_idx))

            else:
                this_ground_truth_column.extend([0] * len(data["column_labels"][table_idx]))
        column_num_ground_truth.append(sum(this_ground_truth_column))
        column_labels_ground_truth.extend(this_ground_truth_column)

        table_count = 0
        if type(predicted_schema_linking) != dict:
            this_predicted_tables = [0] * len(this_ground_truth_table)
            this_predicted_columns = [0] * len(this_ground_truth_column)
        else:
            this_predicted_tables = []
            this_predicted_columns = []
            predicted_table_indices = []
            predicted_column_indices = []
            for table_idx in range(len(table_names)):
                this_table_name = table_names[table_idx]
                this_column_names = data["schema"]["schema_items"][table_idx]["column_names"]
                try:
                    pred_columns = predicted_schema_linking[this_table_name]
                except:
                    pred_columns = None
                if pred_columns == None or isinstance(pred_columns, str):
                    this_predicted_tables.append(0)
                    this_predicted_columns.extend([0] * len(this_column_names))
                else:
                    if len(pred_columns) == 1 and pred_columns[0] == "*":
                        pred_columns = []
                    this_predicted_tables.append(1)
                    predicted_entities[this_table_name] = []
                    predicted_table_indices.append(str(table_idx))
                    column_num.append(len(pred_columns))
                    table_count += 1
                    if len(pred_columns) == 0:
                        this_predicted_columns.extend([0] * len(this_column_names))
                    else:
                        for column_idx in range(len(this_column_names)):
                            this_column_name = this_column_names[column_idx]
                            if this_column_name in set(pred_columns):
                                this_predicted_columns.append(1)
                                predicted_entities[this_table_name].append(this_column_name)
                                predicted_column_indices.append(str(table_idx) + "." + str(column_idx))
                            else:
                                this_predicted_columns.append(0)
        table_num.append(table_count)

        table_precision, table_recall, table_f1 = calculate_precision_recall(ground_truth_table_indices, predicted_table_indices)
        column_precision, column_recall, column_f1 = calculate_precision_recall(ground_truth_column_indices, predicted_column_indices)

        table_precision_debiased, table_recall_debiased, table_f1_debiased = calculate_precision_recall_debiased(ground_truth_table_indices, predicted_table_indices)
        column_precision_debiased, column_recall_debiased, column_f1_debiased = calculate_precision_recall_debiased(ground_truth_column_indices, predicted_column_indices)

        if len(np.unique(this_ground_truth_table)) != 1:
            table_auc = roc_auc_score(this_ground_truth_table, this_predicted_tables)
            total_auc_table.append(table_auc)

        if len(np.unique(this_ground_truth_column)) != 1:
            column_auc = roc_auc_score(this_ground_truth_column, this_predicted_columns)
            total_auc_column.append(column_auc)

        total_recall_table.append(table_recall)
        total_precision_table.append(table_precision)
        total_f1_table.append(table_f1)
        total_recall_column.append(column_recall)
        total_precision_column.append(column_precision)
        total_f1_column.append(column_f1)

        total_debiased_recall_table.append(table_recall_debiased)
        total_debiased_recall_column.append(column_recall_debiased)
        total_debiased_precision_table.append(table_precision_debiased)
        total_debiased_precision_column.append(column_precision_debiased)
        total_debiased_f1_table.append(table_f1_debiased)
        total_debiased_f1_column.append(column_f1_debiased)


        if len(ground_truth_table_indices) == 1:
            total_debiased_f1_table_1.append(table_f1_debiased)

        if len(ground_truth_table_indices) == 2:
            total_debiased_f1_table_2.append(table_f1_debiased)

        if len(ground_truth_table_indices) >= 3:
            total_debiased_f1_table_geq3.append(table_f1_debiased)


        t_adding_tap = 0
        t_missing_tap = 0
        for j in range(len(this_ground_truth_table)):
            if this_ground_truth_table[j] == 0 and this_predicted_tables[j] == 1:
                t_adding_tap += 1
        for j in range(len(this_ground_truth_table)):
            if this_ground_truth_table[j] == 1 and this_predicted_tables[j] == 0:
                t_missing_tap += 1
        c_adding_tap = 0
        c_missing_tap = 0
        for j in range(len(this_ground_truth_column)):
            if this_ground_truth_column[j] == 0 and this_predicted_columns[j] == 1:
                c_adding_tap += 1
        for j in range(len(this_ground_truth_column)):
            if this_ground_truth_column[j] == 1 and this_predicted_columns[j] == 0:
                c_missing_tap += 1
        this_eval_record["t_adding"] = t_adding_tap
        this_eval_record["t_missing"] = t_missing_tap
        this_eval_record["c_adding"] = c_adding_tap
        this_eval_record["c_missing"] = c_missing_tap
        this_eval_record["matched_entities"] = matched_entities
        this_eval_record["predicted_entities"] = predicted_entities
        eval_record.append(this_eval_record)
        # Missing
        if sum(this_ground_truth_table) != 0:
            if t_missing_tap > 0:
                entity_missing_table = 1
                entity_redundancy_table = 1
            else:
                entity_missing_table = 0
                entity_redundancy_table = t_adding_tap / sum(this_predicted_tables)

            total_Missing_table.append(entity_missing_table)
            total_Redundancy_table.append(entity_redundancy_table)

        if sum(this_ground_truth_column) != 0:
            if c_missing_tap > 0:
                entity_missing_column = 1
                entity_redundancy_column = 1

            else:
                entity_missing_column = 0
                entity_redundancy_column = c_adding_tap / sum(this_predicted_columns)

            total_Missing_column.append(entity_missing_column)
            total_Redundancy_column.append(entity_redundancy_column)


    total_recall_debiased_table_score = sum(total_debiased_recall_table) / len(total_debiased_recall_table)
    total_recall_debiased_column_score = sum(total_debiased_recall_column) / len(total_debiased_recall_column)
    total_precision_debiased_table_score = sum(total_debiased_precision_table) / len(total_debiased_precision_table)
    total_precision_debiased_column_score = sum(total_debiased_precision_column) / len(total_debiased_precision_column)
    total_f1_debiased_table_score = sum(total_debiased_f1_table) / len(total_debiased_f1_table)
    total_f1_debiased_column_score = sum(total_debiased_f1_column) / len(total_debiased_f1_column)

    print(dev_file_name)
    print("\nTable:\trecall-debiased: {:.2f}\tprecision-debiased: {:.2f}\tf1-debiased: {:.2f}".format(
        100 * total_recall_debiased_table_score, 100 * total_precision_debiased_table_score, 100 * total_f1_debiased_table_score))


    print("Column:\trecall-debiased: {:.2f}\tprecision-debiased: {:.2f}\tf1-debiased: {:.2f}".format(
        100 * total_recall_debiased_column_score,100 * total_precision_debiased_column_score,100 * total_f1_debiased_column_score))
    print("\n")



    return eval_record, (total_f1_debiased_table_score, total_f1_debiased_column_score)


def contains_all_keywords(directory_name, keywords):
    return all(keyword in directory_name for keyword in keywords)

def find_directories_with_keywords(base_directory, keywords):
    matching_directories = []
    for root, dirs, files in os.walk(base_directory):
        for directory in dirs:
            if contains_all_keywords(directory, keywords):
                matching_directories.append(os.path.join(root, directory))

    return matching_directories

if __name__ == "__main__":
    opt = parse_option()
    dev_file_name = opt.data
    if "bird" in opt.data:
         dev_file = json.load(open("./data/bird_dev_full.json", encoding='UTF-8'))
    elif "spider" in opt.data:
         dev_file = json.load(open("./data/spider_dev_full.json", encoding='UTF-8'))

    if opt.data.endswith('.json'):
        data_file = json.load(open(opt.data))
        if len(data_file) != len(dev_file):
            print(opt.data + "\t\t [Haven't completed]")
        else:
            _eval(dev_file, data_file, dev_file_name)
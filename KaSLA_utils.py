
import json
import numpy as np
import math



def estimation_relevance(recall_name, recall_prob, name_generative):
    _value_base = []
    for _index in range(len(recall_name)):
        recall_p = recall_prob[_index]
        if recall_name[_index] in name_generative:
            gen = 1
        else:
            gen = 0
        result = recall_p + gen
        if result > 1:
            result = 1
        _value_base.append(result)
    return _value_base



def estimation_weight(index, value_all):
    _p = value_all[index]
    return int(1/_p)


def file_name_DP(SL_file_path):
    def extract_substring(input_string):
        start_keyword = "icl-results/"
        end_keyword = "_question_to_columns.json"
        start_index = input_string.find(start_keyword)
        end_index = input_string.find(end_keyword, start_index)
        extracted = input_string[start_index + len(start_keyword):end_index]

        return extracted
    new_schema_ling_files = "linking-results/SL_KaSLA_{}.json".format(extract_substring(SL_file_path))
    return new_schema_ling_files



def KSL_perpare_inputs_wT(SL_file_path):
    if "bird" in SL_file_path:
        # load training set with ground truth linking result and predicted prob score
        train_recall_score_perT = json.load(open("data/SL_ranking_probs/SL_recall_bird_train_perT.json", encoding="utf-8"))
        train_ground_truth = json.load(open("data/SL_ranking_probs/SL_ideal_bird_train_results.json", encoding="utf-8"))

        dev_dataset = json.load(open("data/bird_dev_full.json", encoding="utf-8"))
        dev_recall_score_perT = json.load(open("data/SL_ranking_probs/SL_recall_bird_dev_perT.json", encoding="utf-8"))
        similarities_pool_question = np.load("data/SL_ranking_probs/bird_question_similarities_pool_dev.npy")

        dev_generative_SL_results = json.load(open(SL_file_path,
            encoding="utf-8"))


    elif "spider" in SL_file_path:
        train_recall_score_perT = json.load(open("data/SL_ranking_probs/SL_recall_spider_train_perT.json", encoding="utf-8"))
        train_ground_truth = json.load(open("data/SL_ranking_probs/SL_ideal_spider_train_results.json", encoding="utf-8"))

        dev_dataset = json.load(open("data/spider_dev_full.json", encoding="utf-8"))
        dev_recall_score_perT = json.load(open("data/SL_ranking_probs/SL_recall_spider_dev_perT.json", encoding="utf-8"))
        similarities_pool_question = np.load("data/SL_ranking_probs/spider_question_similarities_pool_dev.npy")

        dev_generative_SL_results = json.load(open(SL_file_path,
            encoding="utf-8"))

    return (dev_dataset, dev_generative_SL_results, dev_recall_score_perT,
            train_recall_score_perT, train_ground_truth, similarities_pool_question)




def knapsack_dp(wgt, val, cap):
    n = len(wgt)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]
    record = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for c in range(1, cap + 1):
            if wgt[i - 1] > c:
                dp[i][c] = dp[i - 1][c]
                record[i][c] = 0
            else:
                not_choose = dp[i - 1][c]
                choose_this = dp[i - 1][c - wgt[i - 1]] + val[i - 1]
                if choose_this > not_choose:
                    dp[i][c] = choose_this
                    record[i][c] = 1
                else:
                    dp[i][c] = not_choose
                    record[i][c] = 0
    i = n
    c = cap
    selected_entities = []
    while i > 0:
        if record[i][c] == 1:
            # print("choose {}".format(i))
            selected_entities.append(i-1)
            c = c - wgt[i-1]
            i = i - 1
        else:
            i = i - 1
    return selected_entities, dp[n][cap]

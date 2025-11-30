
import json
from tqdm import tqdm
import os
from KaSLA_utils import knapsack_dp, KSL_perpare_inputs_wT, \
    file_name_DP, estimation_weight, estimation_relevance

def new_schema_linking(dev_dataset, dev_generative_SL_results_dict, dev_probabilistic_score_perT,
                       train_probabilistic_score_perT, train_ground_truth, similarities_pool_question, new_schema_ling_files):

    new_schema_linking_results = []
    topK = 30
    dev_generative_SL_results = []
    if isinstance(dev_generative_SL_results_dict, dict):
        for k, v in dev_generative_SL_results_dict.items():
            dev_generative_SL_results.append(v["pred_linking"])
    else:
        dev_generative_SL_results = dev_generative_SL_results_dict
    print(len(dev_generative_SL_results_dict))
    print(len(dev_generative_SL_results))

    for index, data in tqdm(enumerate(dev_dataset)):
        this_similarities = similarities_pool_question[index]
        top_k_indices = sorted(range(len(this_similarities)), key=lambda x: this_similarities[x], reverse=True)[:topK]
        demonstration_table_sumWeight_groundT, demonstration_column_sumWeight_groundT= [], []

        for instructions_idx in top_k_indices:
            instance_train_perT = train_probabilistic_score_perT[instructions_idx]
            instance_train_ground_truth = train_ground_truth[instructions_idx]

            train_table_probabilistic_name = instance_train_perT["t_ranking_name_perT"]
            train_table_probabilistic_prob = instance_train_perT["t_ranking_prob_perT"]

            train_table_ground_truth_name = []
            for train_t_n, train_c_list in instance_train_ground_truth.items():
                if train_c_list == None:
                    continue
                train_table_ground_truth_name.append(train_t_n)

            train_table_relevance_all = estimation_relevance(
                train_table_probabilistic_name, train_table_probabilistic_prob, train_table_ground_truth_name)

            if len(train_table_ground_truth_name) > 0:

                train_table_weight_all, train_table_weight_groundtruth = [], []

                for train_t_index in range(len(train_table_probabilistic_name)):
                    train_t_w = estimation_weight(train_t_index, train_table_relevance_all)
                    train_table_weight_all.append(train_t_w)
                    train_t_name = train_table_probabilistic_name[train_t_index]
                    if train_t_name in train_table_ground_truth_name:
                        train_table_weight_groundtruth.append(train_t_w)
                demonstration_table_sumWeight_groundT.append(sum(train_table_weight_groundtruth))



            for train_t_n, train_column_ground_truth_name in instance_train_ground_truth.items():
                if train_column_ground_truth_name == None:
                    continue
                train_column_probabilistic_name = instance_train_perT[train_t_n]["c_ranking_name_perT"]
                train_column_probabilistic_prob = instance_train_perT[train_t_n]["c_ranking_prob_perT"]

                train_column_relevance_all = estimation_relevance(
                    train_column_probabilistic_name, train_column_probabilistic_prob, train_column_ground_truth_name)

                if len(train_column_ground_truth_name) > 0:
                    train_column_weight_all, train_column_weight_groundtruth = [], []
                    for train_c_index in range(len(train_column_probabilistic_name)):
                        train_c_w = estimation_weight(train_c_index, train_column_relevance_all)
                        train_column_weight_all.append(train_c_w)
                        train_c_name = train_column_probabilistic_name[train_c_index]
                        if train_c_name in train_column_ground_truth_name:
                            train_column_weight_groundtruth.append(train_c_w)
                    demonstration_column_sumWeight_groundT.append(sum(train_column_weight_groundtruth))


        new_instance = dict()
        instance_perT = dev_probabilistic_score_perT[index]
        instance_dev_generative = dev_generative_SL_results[index]


        if isinstance(instance_dev_generative, str):
            instance_dev_generative = dict()

        table_probabilistic_name = instance_perT["t_ranking_name_perT"]
        table_probabilistic_prob = instance_perT["t_ranking_prob_perT"]
        table_name_generative = []
        for t_n, column_list in instance_dev_generative.items():
            if column_list != None:
                table_name_generative.append(t_n)
        table_index_generative = []
        for t_index in range(len(table_probabilistic_name)):
            if table_probabilistic_name[t_index] in table_name_generative:
                table_index_generative.append(t_index)

        table_relevance_all = estimation_relevance(table_probabilistic_name, table_probabilistic_prob, table_name_generative)

        if len(demonstration_table_sumWeight_groundT) != 0:
            table_weight_all, table_relevance_ground_truth, table_weight_generative = [], [], []

            for t_index in range(len(table_relevance_all)):
                t_w = estimation_weight(t_index, table_relevance_all)
                table_weight_all.append(t_w)
                
            table_cap_setting = int(max(demonstration_table_sumWeight_groundT))
            table_value_all = table_relevance_all
            table_selected_index, max_value = knapsack_dp(table_weight_all, table_value_all, table_cap_setting)
        else:
            table_selected_index = None

        table_DP_name = []
        for t_index in table_selected_index:
            t_n = table_probabilistic_name[t_index]
            table_DP_name.append(t_n)

        table_selected_name = table_DP_name

        for t_n, train_c_list in instance_dev_generative.items():
            if train_c_list == None:
                continue
            if t_n not in table_selected_name:
                instance_dev_generative[t_n] = None
        for t_n in table_selected_name:
            if t_n not in instance_dev_generative:
                instance_dev_generative[t_n] = []
            else:
                if instance_dev_generative[t_n] == None:
                    instance_dev_generative[t_n] = []

        renew_table_name_generative = []
        for t_n, column_list in instance_dev_generative.items():
            if column_list != None:
                renew_table_name_generative.append(t_n)
        assert set(renew_table_name_generative)==set(table_selected_name)


        # # processing column
        for t_index in range(len(table_probabilistic_name)):
            t_name = table_probabilistic_name[t_index]
            column_probabilistic_name = instance_perT[t_name]["c_ranking_name_perT"]
            column_probabilistic_prob = instance_perT[t_name]["c_ranking_prob_perT"]

            if t_name in instance_dev_generative:
                column_name_generative = instance_dev_generative[t_name]
            else:
                new_instance[t_name] = None
                continue
            if column_name_generative == None:
                new_instance[t_name] = None
                continue
            if len(demonstration_column_sumWeight_groundT)!=0:
                column_cap_setting = int(max(demonstration_column_sumWeight_groundT))
                column_index_generative = []
                for c_index in range(len(column_probabilistic_name)):
                    if column_probabilistic_name[c_index] in column_name_generative:
                        column_index_generative.append(c_index)

                column_relevance_all = estimation_relevance(
                    column_probabilistic_name, column_probabilistic_prob, column_name_generative)
                column_weight_all = []
                for c_index in range(len(column_relevance_all)):
                    c_w = estimation_weight(c_index, column_relevance_all)
                    column_weight_all.append(c_w)
                column_value_all = column_relevance_all
                column_selected_index, max_value = knapsack_dp(column_weight_all, column_value_all, column_cap_setting)
            else:
                column_selected_index = []
            column_DP_name = []
            for c_index in column_selected_index:
                c_n = column_probabilistic_name[c_index]
                column_DP_name.append(c_n)
            column_selected_name = column_DP_name
            new_instance[t_name] = column_selected_name
        new_schema_linking_results.append(new_instance)

    json.dump(new_schema_linking_results, open(new_schema_ling_files, 'w'), indent=4)



def KSL_process(SL_file_path):
    (dev_dataset, dev_generative_SL_results_dict, dev_probabilistic_score_perT, train_probabilistic_score_perT, train_ground_truth,
     similarities_pool_question) = KSL_perpare_inputs_wT(SL_file_path)

    new_schema_ling_files = file_name_DP(SL_file_path)
    new_schema_linking(dev_dataset, dev_generative_SL_results_dict, dev_probabilistic_score_perT,
                       train_probabilistic_score_perT, train_ground_truth,  similarities_pool_question, new_schema_ling_files)
    sh_str = "python eval_schema-linking.py --data {}".format(new_schema_ling_files)
    os.system(sh_str)

SL_file_path = "icl-results/bird-dev_binary_function_deepseek-coder-1.3b_question_to_columns.json"


KSL_process(SL_file_path)

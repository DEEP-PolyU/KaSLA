import json


def prepare_fewshot_input_seq(tokenizer, max_tokens, mode, num_of_demonstrations, eval_data, demonstration_set, similarity):
    top_k_indices = sorted(range(len(similarity)), key = lambda x: similarity[x], reverse = True)[:num_of_demonstrations]
    # top_k_indices = list(reversed(top_k_indices))
    # top_k_indices = random.sample(range(len(similarity)), opt.num_of_demonstrations)
    print(top_k_indices)
    print(similarity[top_k_indices])

    few_shot_examples = ""
    example_num = 0


    if "codesStyle" in mode:
        if "str" in mode:
            target_input = (few_shot_examples
                     + "database schema:\n"
                     + eval_data["schema_sequence"]+ "\n" + eval_data["content_sequence"]+ "\nQuestion:\n" + eval_data["text"] + "\n")
            target_output = eval_data["sql"]
        elif "json" in mode:
            target_input = (few_shot_examples
                     + "[Text-to-SQL task]\ndatabase schema:\n"
                     + eval_data["schema_sequence"]+ "\n" + eval_data["content_sequence"]+ "\nQuestion:\n" + eval_data["text"]
                     + "\nGenerate SQL to solve the above question:\n")
            target_output = {
                "SQL": eval_data["sql"]
            }
            target_output = json.dumps(target_output)

    elif mode == "sg_t2s-sl_briefS_fewshot":
        target_input = ("Examples:\n" + few_shot_examples
                 + "\nFollow the examples, solve the following question in the same json template:\nSchema:\n"
                 + eval_data["schema_sequence"] + "\nQuestion:\n" + eval_data["text"]
                 + "\nGenerate SQL to solve the above question and list the relevant columns in each table:\n")
        target_output = {
            "SQL": eval_data["sql"],
            "Relevant columns": eval_data["matched_entities"]
        }
        target_output = json.dumps(target_output)
    elif mode == "t2s-brief-fewshot":
        target_input = ("Examples:\n" + few_shot_examples
                 + "\nFollow the examples, solve the following question in the same json template:\nSchema:\n"
                 + eval_data["schema_sequence"] + "\nQuestion:\n" + eval_data["text"]
                 + "\nGenerate SQL to solve the above question:\n")
        target_output = {
            "SQL": eval_data["sql"]
        }
        target_output = json.dumps(target_output)



    for idx in top_k_indices:
        demonstration_sql = demonstration_set[idx]["sql"]
        if demonstration_sql.endswith(";"):
            demonstration_sql = demonstration_sql[:-1].strip()
        one_shot_schema = demonstration_set[idx]["schema_sequence"]
        one_shot_question = demonstration_set[idx]["text"]

        if "codesStyle" in mode:
            one_shot_content_sequence = demonstration_set[idx]["content_sequence"]
            if "str" in mode:
                one_shot_output = demonstration_sql
                one_shot_example = ("database schema:\n" + one_shot_schema + "\n" + one_shot_content_sequence+ "\nQuestion:\n" + one_shot_question
                                  + "\n" + one_shot_output)
            elif "json" in mode:
                one_shot_output = {
                    "SQL": demonstration_sql
                }
                one_shot_example = ("[Text-to-SQL task]\ndatabase schema:\n" + one_shot_schema + "\n" + one_shot_content_sequence + "\nQuestion:\n" + one_shot_question
                         + "\nGenerate SQL to solve the above question:\n" + json.dumps(one_shot_output))
        elif mode == "sg_t2s-sl_briefS_fewshot":
            one_shot_output = {
                "SQL": demonstration_sql,
                "Relevant columns": demonstration_set[idx]["matched_entities"]
            }
            one_shot_example = ("Schema:\n" + one_shot_schema + "\nQuestion:\n" + one_shot_question
                              + "\nGenerate SQL to solve the above question and list the relevant columns in each table:\n"
                              + json.dumps(one_shot_output))
        elif mode == "t2s-brief-fewshot":
            one_shot_output = {
                "SQL": demonstration_sql
            }
            one_shot_example = ("Schema:\n" + one_shot_schema + "\nQuestion:\n" + one_shot_question
                              + "\nGenerate SQL to solve the above question:\n"
                              + json.dumps(one_shot_output))

        input_ids = [tokenizer.bos_token_id] + tokenizer(few_shot_examples + one_shot_example + "\n\n" + target_input + target_output,
                                                         truncation=False)["input_ids"]
        if len(input_ids) > max_tokens:
            break
        few_shot_examples += one_shot_example + "\n\n"
        example_num += 1
    print("{} shot examples".format(example_num))

    if "codesStyle" in mode:
        if "str" in mode:
            input = (few_shot_examples
                     + "database schema:\n"
                     + eval_data["schema_sequence"]+ "\n" + eval_data["content_sequence"]+ "\nQuestion:\n" + eval_data["text"] + "\n")
            output = eval_data["sql"]
        elif "json" in mode:
            input = (few_shot_examples
                     + "[Text-to-SQL task]\ndatabase schema:\n"
                     + eval_data["schema_sequence"]+ "\n" + eval_data["content_sequence"]+ "\nQuestion:\n" + eval_data["text"]
                     + "\nGenerate SQL to solve the above question:\n")
            output = {
                "SQL": eval_data["sql"]
            }
            output = json.dumps(output)

    elif mode == "sg_t2s-sl_briefS_fewshot":
        input = ("Examples:\n" + few_shot_examples
                 + "\nFollow the examples, solve the following question in the same json template:\nSchema:\n"
                 + eval_data["schema_sequence"] + "\nQuestion:\n" + eval_data["text"]
                 + "\nGenerate SQL to solve the above question and list the relevant columns in each table:\n")
        output = {
            "SQL": eval_data["sql"],
            "Relevant columns": eval_data["matched_entities"]
        }
        output = json.dumps(output)
    elif mode == "t2s-brief-fewshot":
        input = ("Examples:\n" + few_shot_examples
                 + "\nFollow the examples, solve the following question in the same json template:\nSchema:\n"
                 + eval_data["schema_sequence"] + "\nQuestion:\n" + eval_data["text"]
                 + "\nGenerate SQL to solve the above question:\n")
        output = {
            "SQL": eval_data["sql"]
        }
        output = json.dumps(output)

    return [input, output]

def prepare_sequence_t2s_dataset(data):
    if "content_sequence" in data:
        input_full_schema_sequence =  data["full_schema_sequence"]
        input_linked_schema_sequence = data["schema_sequence"]+ "\n" + data["content_sequence"]
        input_question = data["text"]
    else:
        input_full_schema_sequence =  data["full_schema_sequence"]
        input_linked_schema_sequence = data["schema_sequence"]
        input_question = data["text"]

    return [input_full_schema_sequence, input_linked_schema_sequence, input_question]

def prepare_sequence_t2s_codesStyle_json_GPT(data):
    if "content_sequence" in data:
        input = ("[Text-to-SQL task]\nFull database schema:\n" + data["full_schema_sequence"]
                 + "\nLinked database schema:\n" + data["schema_sequence"]
                 + "\n" + data["content_sequence"] + "\nQuestion:\n" + data["text"]
                 + "\nGenerate SQL to solve the above question:\n")
    else:
        input = ("[Text-to-SQL task]\nFull database schema:\n" + data["full_schema_sequence"]
                 + "\nLinked database schema:\n" + data["schema_sequence"]
                 + "\nQuestion:\n" + data["text"]
                 + "\nGenerate SQL to solve the above question:\n")
    output = {
        "SQL": data["sql"]
    }
    return [input, json.dumps(output)]

def prepare_sequence_t2s_codesStyle_json(data):
    if "content_sequence" in data:
        input = ("[Text-to-SQL task]\ndatabase schema:\n" + data["schema_sequence"] + "\n" + data["content_sequence"] + "\nQuestion:\n" + data["text"]
                 + "\nGenerate SQL to solve the above question:\n")
    else:
        input = ("[Text-to-SQL task]\ndatabase schema:\n" + data["schema_sequence"] + "\nQuestion:\n" + data["text"]
                 + "\nGenerate SQL to solve the above question:\n")
    output = {
        "SQL": data["sql"]
    }
    return [input, json.dumps(output)]
# def prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike(data, mode):
#     if "content_sequence" in data:
#         input = ("[Schema-Linking task]\ndatabase schema:\n" + data["schema_sequence"] + "\n" + data["content_sequence"] + "\nQuestion:\n" + data["text"])
#     else:
#         input = ("[Schema-Linking task]\ndatabase schema:\n" + data["schema_sequence"] + "\nQuestion:\n" + data["text"])
#     if "table_linking" in mode:
#         input = input + "\nGenerate SQL to solve the above question and list the relevant tables:\n"
#     elif "column_linking" in mode:
#
#     else:
#         input = input + "\nGenerate SQL to solve the above question and list the relevant columns in each table:\n"
#
#     output = {
#         "Reasoning For Table Linking": "To solve the above question, the following tables are necessary.\n",
#         "Relevant Tables": data["matched_entities"],
#         "Reasoning For SQL Generation": "With the linked tables, the following SQL can solve the given problem.\n",
#         "SQL": data["sql"],
#         "Relevant columns": data["matched_entities"]
#     }
#     return [input, json.dumps(output)]

def prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema_codesStyleLike(data):
    if "content_sequence" in data:
        input = ("[Schema-Linking task]\ndatabase schema:\n" + data["schema_sequence"] + "\n" + data["content_sequence"] + "\nQuestion:\n" + data["text"]
                 + "\nGenerate SQL to solve the above question and list the relevant columns in each table:\n")
    else:
        input = ("[Schema-Linking task]\ndatabase schema:\n" + data["schema_sequence"] + "\nQuestion:\n" + data["text"]
                 + "\nGenerate SQL to solve the above question and list the relevant columns in each table:\n")
    output = {
        "SQL": data["sql"],
        "Relevant columns": data["matched_entities"]
    }
    return [input, json.dumps(output)]


def prepare_sequence_sg_t2sSg_i_1schema_o_1_schema_codesStyleLike(data):
    input = ("[Schema-Linking task]\ndatabase schema:\n" + data["schema_sequence"] + "\nQuestion:\n" + data["text"]
             + "\nList the relevant columns in each table:\n")
    output = {
        "Relevant columns": data["matched_entities"]
    }
    return [input, json.dumps(output)]








def prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema(data):
    # input = ("[Schema-Linking task]\nSchema:\n" + data["schema_sequence"] + "\nQuestion:\n" + data["text"]
    #          + "\nGenerate SQL to solve the above question and list the relevant columns in each table:\n")
    input = ("Schema:\n" + data["schema_sequence"] + "\nQuestion:\n" + data["text"]
             + "\nGenerate SQL to solve the above question and list the relevant columns in each table:\n")
    output = {
        "SQL": data["sql"],
        "Relevant columns": data["matched_entities"]
    }
    return [input, json.dumps(output)]


def prepare_sequence_t2s_i_1schema_o_1_sql(data):
    if "content_sequence" in data:
        input = ("Schema:\n" + data["schema_sequence"]+ "\n" + data["content_sequence"] + "\nQuestion:\n" + data["text"]
                 + "\nGenerate SQL to solve the above question:\n")
    else:
        input = ("Schema:\n" + data["schema_sequence"] + "\nQuestion:\n" + data["text"]
                      + "\nGenerate SQL to solve the above question:\n")
    output = {
        "SQL": data["sql"]
    }
    return [input, json.dumps(output)]


def prepare_sequence_t2s_i_1schema_o_1_sql_codes(data):
    input = ("Schema:\n" + data["schema_sequence"] + "\n" + data["content_sequence"] + "\nQuestion:\n" + data["text"] + "\n")
    output = data["sql"]
    return [input, output]



def prepare_sequence_t2s_codesStyle_json_fewshot(data):
    if "content_sequence" in data:
        input = {
            "database schema": data["schema_sequence"] + "\n" + data["content_sequence"],
            "question": data["text"]
        }
    else:
        input = {
            "database schema": data["schema_sequence"],
            "question": data["text"]
        }
    output = {
        "SQL": data["sql"]
    }
    return [input, json.dumps(output)]


def prepare_sequence_t2s_codesStyle_str(data):
    input = ("database schema:\n" + data["schema_sequence"] + "\n" + data["content_sequence"] + "\nQuestion:\n" + data["text"]
             + "\n")
    output = data["sql"]
    return [input, output]
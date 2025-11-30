import json
import torch

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

random.seed(42)
def add_quotation_mark(s):
    return "`" + s + "`"
def detect_special_char(name):
    for special_char in ['(', '-', ')', ' ', '/']:
        if special_char in name:
            return True
    return False

def filter_schema_with_table_linking_file(dataset, table_linking_file):
    if table_linking_file is not None:
        schema_linking_results = json.load(open(table_linking_file))
    else:
        schema_linking_results = None
    print("filtering schema items for the dataset")
    for index, data in tqdm(enumerate(dataset)):
        predicted_schemalinking = schema_linking_results[str(index)]
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

        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        ground_truth_table_names = []
        for table_idx in ground_truth_table_indices:
            ground_truth_table_names.append(table_names[table_idx])
        print(index)
        print("ground_truth_table_names: {}".format(ground_truth_table_names))
        print("predicted_schemalinking: {}".format(predicted_schemalinking))
        table_indices = []
        match_table_names = []

        for table_idx, table_name in enumerate(table_names):
            if table_name in predicted_schemalinking:
                table_indices.append(table_idx)
                match_table_names.append(table_name)

        if len(table_indices) == 0:
            table_indices = list(range(len(table_names)))
        print("table_indices: {}".format(table_indices))
        print("\n")

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
        # replace the old matched contents with the filtered matched contents
        data["matched_contents"] = filtered_matched_contents
    return dataset


def prepare_each_input(prefix_seq, tokenizer, max_prefix_length):
    input_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)["input_ids"]
    if len(input_ids) > max_prefix_length:
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length - 1):]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64)
    }, len(input_ids)

def generation_func(model, inputs, tokenizer, max_new_tokens, eos_token_id, num_beams):
    generated_answers = []
    for this_input in inputs:
        input_length = this_input["input_ids"].shape[1]
        with torch.no_grad():
            generate_ids = model.generate(
                **this_input,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,  # 4 留着效果非常差，可能对没训练过的llms不能加这个参数，目前还没有搞明白
                use_cache=True,
                # eos_token_id=eos_token_id,
                # tempareture=1,
            ).detach().cpu()
            generated_answer = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)
            generated_answers.extend(generated_answer)
    return generated_answers



def filter_schema_with_column_linking_file(dataset, column_linking_file):
    if column_linking_file is not None:
        column_linking_results = json.load(open(column_linking_file))
    else:
        column_linking_results = None

    for index, data in tqdm(enumerate(dataset)):
        predicted_column_linking = column_linking_results[str(index)]
        filtered_schema = dict()

        filtered_schema["schema_items"] = []
        filtered_schema["foreign_keys"] = []

        table_names = [table["table_name"] for table in data["schema"]["schema_items"]]
        table_comments = [table["table_comment"] for table in data["schema"]["schema_items"]]
        column_names = [table["column_names"] for table in data["schema"]["schema_items"]]
        column_dict = {}
        for table_index in range(len(table_names)):
            column_dict[table_names[table_index]] = column_names[table_index]
        column_types = [table["column_types"] for table in data["schema"]["schema_items"]]
        column_comments = [table["column_comments"] for table in data["schema"]["schema_items"]]
        column_contents = [table["column_contents"] for table in data["schema"]["schema_items"]]
        pk_indicators = [table["pk_indicators"] for table in data["schema"]["schema_items"]]

        ground_truth_table_indices = [table_idx for table_idx, table_label in enumerate(data["table_labels"])
                                      if table_label == 1]
        ground_truth_table_names = []
        for table_idx in ground_truth_table_indices:
            ground_truth_table_names.append(table_names[table_idx])
        print(index)
        print("ground_truth_table_names: {}".format(ground_truth_table_names))
        table_indices = []
        match_table_names = []

        schema_linking_dict = {}
        for predicted_str in predicted_column_linking:
            try:
                this_table, this_column = predicted_str.split(".")
            except:
                continue
            if this_table in table_names:
                if this_table not in schema_linking_dict:
                    schema_linking_dict[this_table] = []
                if this_column in column_dict[this_table] and this_column not in schema_linking_dict[this_table]:
                    schema_linking_dict[this_table].append(this_column)
        print("predicted_schema_linking: {}".format(schema_linking_dict))
        print("\n")

        for table_idx, table_name in enumerate(table_names):
            if table_name in schema_linking_dict:
                table_indices.append(table_idx)
                match_table_names.append(table_name)

        if len(schema_linking_dict) == 0:
            table_indices = list(range(len(table_names)))

        for table_idx in table_indices:
            if len(schema_linking_dict) == 0:
                column_indices = list(range(len(column_names[table_idx])))
            else:
                column_indices = []
                for column_idx, column_name in enumerate(column_names[table_idx]):
                    if column_name in schema_linking_dict[table_names[table_idx]]:
                        column_indices.append(column_idx)
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
        # extract foreign keys among remianed tables
        filtered_table_names = [table_names[table_idx] for table_idx in table_indices]
        for foreign_key in data["schema"]["foreign_keys"]:
            source_table, source_column, target_table, target_column = foreign_key
            if source_table in filtered_table_names and target_table in filtered_table_names:
                filtered_schema["foreign_keys"].append(foreign_key)
        data["filtered_schema"] = filtered_schema
    return dataset


import sqlite3


def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [max(len(str(value[i])) for value in values + [column_names]) for i in range(len(column_names))]

    # Print the column names
    header = ''.join(f'{column.rjust(width)} ' for column, width in zip(column_names, widths))
    for value in values:
        row = ''.join(f'{str(v).rjust(width)} ' for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + '\n' + rows
    return final_output


def get_db_schema_sequence_all_mcs_sql(schema, db_path):
    schema_sequence = "\n### SQLite SQL tables, with their properties:\n"

    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        if detect_special_char(table_name):
            table_name = add_quotation_mark(table_name)
        column_info_list = []
        for column_name in table["column_names"]:
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)
            column_info_list.append(column_name)
        schema_sequence += "# " + table_name + " ( " + " , ".join(column_info_list) + " )\n"

    if len(schema["foreign_keys"]) != 0:
        schema_sequence += "\n"
        for foreign_key in schema["foreign_keys"]:
            for i in range(len(foreign_key)):
                if detect_special_char(foreign_key[i]):
                    foreign_key[i] = add_quotation_mark(foreign_key[i])
            schema_sequence += "{}.{} = {}.{}\n".format(foreign_key[0], foreign_key[1], foreign_key[2], foreign_key[3])

    schema_sequence += "\n\n### The type and description of each column:\n"

    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        if detect_special_char(table_name):
            table_name = add_quotation_mark(table_name)
        column_info_list = []
        for column_name, column_type, column_comment in \
                zip(table["column_names"], table["column_types"], table["column_comments"]):
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)
            additional_column_info = []
            # column type
            additional_column_info.append(column_type)
            # column comment
            if column_comment != "":
                additional_column_info.append("comment : " + column_comment)

            if len(additional_column_info) !=0:
                column_info_list.append(column_name + " ( " + " | ".join(additional_column_info) + " )")
            else:
                column_info_list.append(column_name)

        schema_sequence += "# [ " + table_name + " ]"
        schema_sequence += "\n- ".join(column_info_list) + "\n"

    schema_sequence += "\n\n### Sample rows of each table in csv format:\n"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        if detect_special_char(table_name):
            table_name = add_quotation_mark(table_name)
        this_column_names = []
        for this_column_name in table["column_names"]:
            if detect_special_char(this_column_name):
                this_column_names.append(add_quotation_mark(this_column_name))
        if len(this_column_names)!=0:
            this_column_names_prompt = ", ".join(this_column_names)
        else:
            this_column_names_prompt = "*"
        try:
            execute_query = "SELECT {} FROM {} LIMIT {}".format(this_column_names_prompt, table_name, 3)
            print(execute_query)
            cursor.execute(execute_query)
            values = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]
            rows_prompt = nice_look_table(column_names=column_names, values=values)
        except:
            rows_prompt = this_column_names_prompt
        schema_sequence += "# [ " + table_name + " ]\n" + rows_prompt + "\n"
    schema_sequence += "\n"
    return schema_sequence

def get_db_schema_sequence_onlyType(schema):
    schema_sequence_list = []
    for table in schema["schema_items"]:
        table_name, table_comment = table["table_name"], table["table_comment"]
        if detect_special_char(table_name):
            table_name = add_quotation_mark(table_name)
        column_info_list = []
        for column_name, column_type in zip(table["column_names"], table["column_types"]):
            if detect_special_char(column_name):
                column_name = add_quotation_mark(column_name)
            additional_column_info = []
            # column type
            additional_column_info.append(column_type)
            if len(additional_column_info) !=0:
                column_info_list.append(column_name + ": " + " | ".join(additional_column_info))
            else:
                column_info_list.append(column_name)
        random.shuffle(column_info_list)
        schema_sequence_list.append("# " + table_name + " , ( " + " , ".join(column_info_list) + " )")
    random.shuffle(schema_sequence_list)
    schema_sequence = "\n".join(schema_sequence_list)
    return schema_sequence.strip()




def prepare_fewshot_input_seq_msc_sql(tokenizer, max_tokens, mode, num_of_demonstrations, eval_data, demonstration_set, similarity):
    top_k_indices = sorted(range(len(similarity)), key = lambda x: similarity[x], reverse = True)[:num_of_demonstrations]

    few_shot_examples = "\n<examples>\n"
    example_num = 0

    instruction_1 = "### Given a database schema and question, generate the correct sqlite SQL query for the question.\n"
    few_shot_examples_end = "\n</examples>\n"

    template_examples = """\n
    <example1>
    ### SQLite SQL tables, with their properties:
    # transactions_1k ( TransactionID: integer, Date: date, Time: text, CustomerID: integer, CardID: integer, GasStationID: integer, ProductID: integer, Amount: integer, Price: real )
    # yearmonth ( CustomerID: integer, Date: text, Consumption: real )
    ### Question:
    For all the people who paid more than 29.00 per unit of product id No .5. Give their consumption status in the August of 2012.
    ### Your Answer:
    {
        "reasoning": "August of 2012 means Date contains '201208' in the yearmonth.date of the database. Price per unit of product = Price / Amount in table transactions_1k",
        "sql": "SELECT yearmonth.consumption FROM transactions_1k INNER JOIN yearmonth ON transactions_1k.customerid = yearmonth.customerid WHERE transactions_1k.price / transactions_1k.amount > 29.00 AND transactions_1k.productid = 5 AND yearmonth.date = '201208'"
    }
    ### Answer end
    </example1>

    <example2> ### SQLite SQL tables, with their properties:
    # yearmonth ( CustomerID: integer, Date: text, Consumption: real )
    ### Question:
    How much did customer 6 consume in total between August and November 2013?
    ### Your Answer:
    {
        "reasoning": "Between August And November 2013 refers to Between 201308 And 201311; First 4 strings of Date represents the year.",
        "sql": "SELECT sum(consumption) FROM yearmonth WHERE customerid = 6 AND Date BETWEEN '201308' AND '201311'"
    }
    ### Answer end
    </example2>

    <example3> ### SQLite SQL tables, with their properties:
    # drivers ( driverId: integer, driverRef: text, number: integer, code: text, forename: text, surname: text, dob: date, nationality: text, url: text )
    ### Question:
    How many Australian drivers who were born in 1980?
    ### Your Answer:
    {
        "reasoning": "born in 1980 refers to year(dob) = 1980.",
        "sql": "SELECT count(driverid) FROM drivers WHERE nationality = 'Australian' AND strftime('%Y', dob) = '1980'"
    }
    ### Answer end
    </example3>\n
    """

    instruction_2 = """
Your answer should strictly follow the json format.
### Your Answer:
"""
    target_input = (instruction_1 + few_shot_examples + few_shot_examples_end
            + "\n" +template_examples + "\n"
                    + eval_data["schema_sequence"] + "\n### Question:\n" + eval_data["text"]
                    + "\n"+ instruction_2)
    target_output = {
        "reasoning": eval_data["text"],
        "sql": eval_data["sql"]
    }
    target_output = json.dumps(target_output)

    all_fewshot_examples = []
    for idx in top_k_indices:
        demonstration_sql = demonstration_set[idx]["sql"]
        if demonstration_sql.endswith(";"):
            demonstration_sql = demonstration_sql[:-1].strip()
        # one_shot_schema = demonstration_set[idx]["schema_sequence"]
        one_shot_question = demonstration_set[idx]["text"]

        one_shot_output = demonstration_sql
        one_shot_example = ("\n# Question: " + one_shot_question + "\n# Gold SQL: " + one_shot_output)

        input_ids = [tokenizer.bos_token_id] + tokenizer(few_shot_examples + one_shot_example + "\n"
                                                         + target_input + target_output,
                                                         truncation=False)["input_ids"]
        if len(input_ids) > max_tokens:
            break
        all_fewshot_examples.append(one_shot_example)
        example_num += 1
    random.shuffle(all_fewshot_examples)
    few_shot_examples += "\n\n".join(all_fewshot_examples)
    print("{} shot examples".format(example_num))

    input = (instruction_1 + few_shot_examples + few_shot_examples_end
             + "\n" + template_examples + "\n"
             + eval_data["schema_sequence"] + "\n### Question:\n" + eval_data["text"]
             + "\n" + instruction_2)

    output = {
        "reasoning": eval_data["text"],
        "sql": eval_data["sql"]
    }
    output = json.dumps(output)
    return [input, output]



def prepare_sequence_table_linking(data):
    instruction_1 = "### Given a database schema, question, and knowledge evidence, extract a list of tables that should be referenced to convert the question into SQL.\n"
    examples = """\n
<example1>
### SQLite SQL tables, with their properties:
# customers ( CustomerID: integer, Segment: text, Currency: text )
# gasstations ( GasStationID: integer, ChainID: integer, Country: text, Segment: text )
# products ( ProductID: integer, Description: text )
# transactions_1k ( TransactionID: integer, Date: date, Time: text, CustomerID: integer, CardID: integer, GasStationID: integer, ProductID: integer, Amount: integer, Price: real )
# yearmonth ( CustomerID: integer, Date: text, Consumption: real )
### Question:
For all the people who paid more than 29.00 per unit of product id No .5. Give their consumption status in the August of 2012.
### Your Answer:
{
    "reasoning": "August of 2012 means Date contains '201208' in the yearmonth.date of the database. Price per unit of product = Price / Amount in table transactions_1k",
    "tables": ["transactions_1k", "yearmonth"]
}
### Answer end
</example1>

<example2> ### SQLite SQL tables, with their properties:
# customers ( CustomerID: integer, Segment: text, Currency: text )
# gasstations ( GasStationID: integer, ChainID: integer, Country: text, Segment: text )
# products ( ProductID: integer, Description: text )
# transactions_1k ( TransactionID: integer, Date: date, Time: text, CustomerID: integer, CardID: integer, GasStationID: integer, ProductID: integer, Amount: integer, Price: real )
# yearmonth ( CustomerID: integer, Date: text, Consumption: real )
### Question:
How much did customer 6 consume in total between August and November 2013?
### Your Answer:
{
    "reasoning": "Between August And November 2013 refers to Between 201308 And 201311; First 4 strings of Date represents the year.",
    "tables": ["yearmonth"]
}
### Answer end
</example2>

<example3> ### SQLite SQL tables, with their properties:
# drivers ( driverId: integer, driverRef: text, number: integer, code: text, forename: text, surname: text, dob: date, nationality: text, url: text )
### Question:
How many Australian drivers who were born in 1980?
### Your Answer:
{
    "reasoning": "born in 1980 refers to year(dob) = 1980.",
    "tables": ["drivers"]
}
### Answer end
</example3>
"""

    instruction_2 = """
Your answer should strictly follow the json format.
### Your Answer:
"""
    input = (instruction_1 + examples + "\n### SQLite SQL tables, with their properties:\n" + data[
        "schema_sequence"] + "\n### Question:\n" + data["text"]
             + "\n" + instruction_2)
    return input


def prepare_sequence_column_linking(data):
    instruction_1 = "### Given a database schema, question, and knowledge evidence, extract a list of columns that should be referenced to convert the question into SQL.\n"
    examples = """\n
<example1>
### SQLite SQL tables, with their properties:
# transactions_1k ( TransactionID: integer, Date: date, Time: text, CustomerID: integer, CardID: integer, GasStationID: integer, ProductID: integer, Amount: integer, Price: real )
# yearmonth ( CustomerID: integer, Date: text, Consumption: real )
### Question:
For all the people who paid more than 29.00 per unit of product id No .5. Give their consumption status in the August of 2012.
### Your Answer:
{
    "reasoning": "August of 2012 means Date contains '201208' in the yearmonth.date of the database. Price per unit of product = Price / Amount in table transactions_1k",
    "columns": [ "transactions_1k.CustomerID" , "transactions_1k.ProductID" , "transactions_1k.Amount" , "transactions_1k.Price" , "yearmonth.CustomerID" , "yearmonth.Date" , "yearmonth.Consumption" ]
}
### Answer end
</example1>

<example2> ### SQLite SQL tables, with their properties:
# yearmonth ( CustomerID: integer, Date: text, Consumption: real )
### Question:
How much did customer 6 consume in total between August and November 2013?
### Your Answer:
{
    "reasoning": "Between August And November 2013 refers to Between 201308 And 201311; First 4 strings of Date represents the year.",
    "columns": [ "yearmonth.CustomerID" , "yearmonth.Date" , "yearmonth.Consumption" ]
}
### Answer end
</example2>

<example3> ### SQLite SQL tables, with their properties:
# drivers ( driverId: integer, driverRef: text, number: integer, code: text, forename: text, surname: text, dob: date, nationality: text, url: text )
### Question:
How many Australian drivers who were born in 1980?
### Your Answer:
{
    "reasoning": "born in 1980 refers to year(dob) = 1980.",
    "columns": [ "drivers.driverid" , "drivers.dob" , "drivers.nationality" ]
}
### Answer end
</example3>
"""

    instruction_2 = """
Your answer should strictly follow the json format.
### Your Answer:
"""
    input = (instruction_1 + examples + "\n### SQLite SQL tables, with their properties:\n" + data[
        "schema_sequence"] + "\n### Question:\n" + data["text"]
             + "\n" + instruction_2)
    return input

def prepare_sequence_sql_mask(data):
    instruction_1 = "### Given a DB schema, a question and a SQL, mask the table name, column name, and values in the SQL.\n"
    examples = """\n
<example1>
### SQLite SQL tables, with their properties:
# customers ( CustomerID: integer, Segment: text, Currency: text )
# gasstations ( GasStationID: integer, ChainID: integer, Country: text, Segment: text )
# products ( ProductID: integer, Description: text )
# transactions_1k ( TransactionID: integer, Date: date, Time: text, CustomerID: integer, CardID: integer, GasStationID: integer, ProductID: integer, Amount: integer, Price: real )
# yearmonth ( CustomerID: integer, Date: text, Consumption: real )
### Question:
For all the people who paid more than 29.00 per unit of product id No .5. Give their consumption status in the August of 2012.
### SQL:
SELECT yearmonth.consumption FROM transactions_1k INNER JOIN yearmonth ON transactions_1k.customerid = yearmonth.customerid WHERE transactions_1k.price / transactions_1k.amount > 29.00 AND transactions_1k.productid = 5 AND yearmonth.date = '201208'
### Your Answer:
{
    "Original SQL": "SELECT yearmonth.consumption FROM transactions_1k INNER JOIN yearmonth ON transactions_1k.customerid = yearmonth.customerid WHERE transactions_1k.price / transactions_1k.amount > 29.00 AND transactions_1k.productid = 5 AND yearmonth.date = '201208'",
    "Masked SQL": "SELECT [TABLE].[COLUMN] FROM [TABLE] INNER JOIN [TABLE] ON [TABLE].[COLUMN] = [TABLE].[COLUMN] = WHERE [TABLE].[COLUMN] = / [TABLE].[COLUMN] = > [VALUE] AND [TABLE].[COLUMN] = [VALUE] AND [TABLE].[COLUMN] = [VALUE]"
}
### Answer end
</example1>

<example2> ### SQLite SQL tables, with their properties:
# customers ( CustomerID: integer, Segment: text, Currency: text )
# gasstations ( GasStationID: integer, ChainID: integer, Country: text, Segment: text )
# products ( ProductID: integer, Description: text )
# transactions_1k ( TransactionID: integer, Date: date, Time: text, CustomerID: integer, CardID: integer, GasStationID: integer, ProductID: integer, Amount: integer, Price: real )
# yearmonth ( CustomerID: integer, Date: text, Consumption: real )
### Question:
How much did customer 6 consume in total between August and November 2013?
### SQL:
SELECT sum(consumption) FROM yearmonth WHERE customerid = 6 AND Date BETWEEN '201308' AND '201311'
### Your Answer:
{
    "Original SQL": "SELECT sum(consumption) FROM yearmonth WHERE customerid = 6 AND Date BETWEEN '201308' AND '201311'",
    "Masked SQL": "SELECT sum([COLUMN]) FROM [TABLE] WHERE [COLUMN] = [VALUE] AND [COLUMN] BETWEEN [VALUE] AND [VALUE]"
}
### Answer end
</example2>

<example3> ### SQLite SQL tables, with their properties:
# drivers ( driverId: integer, driverRef: text, number: integer, code: text, forename: text, surname: text, dob: date, nationality: text, url: text )
### Question:
How many Australian drivers who were born in 1980?
### SQL:
SELECT count(driverid) FROM drivers WHERE nationality = 'Australian' AND strftime('%Y', dob) = '1980'
### Your Answer:
{
    "Original SQL": "SELECT count(driverid) FROM drivers WHERE nationality = 'Australian' AND strftime('%Y', dob) = '1980'",
    "Masked SQL": "SELECT count([COLUMN]) FROM [TABLE] WHERE [COLUMN] = 'Australian' AND strftime('%Y', [COLUMN]) = [VALUE]"
}
### Answer end
</example3>
"""
    instruction_2 = """
Your answer should strictly follow the following json format.
### Your Answer:
"""
    input = (instruction_1 + examples + "\n### SQLite SQL tables, with their properties:\n" + data[
        "schema_sequence"] + "\n### Question:\n" + data["text"]+ "\n### SQL:\n" + data["sql"]
             + "\n" + instruction_2)
    return input

def prepare_sequence_question_mask(data):
    instruction_1 = "### Given a DB schema and a question, mask the table name, column name, and values in the question.\n"
    examples = """\n
<example1>
### SQLite SQL tables, with their properties:
# customers ( CustomerID: integer, Segment: text, Currency: text )
# gasstations ( GasStationID: integer, ChainID: integer, Country: text, Segment: text )
# products ( ProductID: integer, Description: text )
# transactions_1k ( TransactionID: integer, Date: date, Time: text, CustomerID: integer, CardID: integer, GasStationID: integer, ProductID: integer, Amount: integer, Price: real )
# yearmonth ( CustomerID: integer, Date: text, Consumption: real )
### Question:
For all the people who paid more than 29.00 per unit of product id No .5. Give their consumption status in the August of 2012.
### Your Answer:
{
    "Original Question": "For all the people who paid more than 29.00 per unit of product id No .5. Give their consumption status in the August of 2012.",
    "Masked Question": "For all the [TABLE] who paid more than [VALUE] per unit of [ COLUMN] [VALUE]. Give their consumption status in the [VALUE]."
}
### Answer end
</example1>

<example2> ### SQLite SQL tables, with their properties:
# customers ( CustomerID: integer, Segment: text, Currency: text )
# gasstations ( GasStationID: integer, ChainID: integer, Country: text, Segment: text )
# products ( ProductID: integer, Description: text )
# transactions_1k ( TransactionID: integer, Date: date, Time: text, CustomerID: integer, CardID: integer, GasStationID: integer, ProductID: integer, Amount: integer, Price: real )
# yearmonth ( CustomerID: integer, Date: text, Consumption: real )
### Question:
How much did customer 6 consume in total between August and November 2013?
### Your Answer:
{
    "Original Question": "How much did customer 6 consume in total between August and November 2013?",
    "Masked Question": "How much did [TABLE] [VALUE] consume in total between [VALUE] and [VALUE]?"
}
### Answer end
</example2>

<example3> ### SQLite SQL tables, with their properties:
# drivers ( driverId: integer, driverRef: text, number: integer, code: text, forename: text, surname: text, dob: date, nationality: text, url: text )
### Question:
How many Australian drivers who were born in 1980?
### Your Answer:
{
    "Original Question": "How many Australian drivers who were born in 1980?",
    "Masked Question": "How many [VALUE] [TABLE] who were born in [VALUE]?"
}
### Answer end
</example3>
"""
    instruction_2 = """
Your answer should strictly follow the following json format.
### Your Answer:
"""
    input = (instruction_1 + examples + "\n### SQLite SQL tables, with their properties:\n" + data[
        "schema_sequence"] + "\n### Question:\n" + data["text"]
             + "\n" + instruction_2)
    return input


# def prepare_sequence_few_shot_SQL_generation(data):
#     instruction = """
#     ### Given a database schema, question, and knowledge evidence, generate the correct sqlite SQL query for the question.
#     """
#     examples = """
#     <examples>
#     # Question: Among all the customers, what is the percentage of the customer’s nation being Germany?
#     # Knowledge Evidence: DIVIDE(COUNT(c_custkey when n_name = ’GERMANY’), COUNT( c_custkey)) as percentage;
#     # Gold SQL: SELECT CAST(SUM(IIF(T2.n_name = ’GERMANY’, 1, 0)) AS REAL) * 100 / COUNT (T1.c_custkey) FROM customer AS T1 INNER JOIN nation AS T2 ON T1.c_nationkey = T2.n_nationkey
#     """




def prepare_sequence_sg_t2sSg_i_1schema_o_1_sql_1_schema(data):
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
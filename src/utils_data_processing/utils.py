import torch


import nltk
def extract_skeleton(text):
    tokens_and_tags = nltk.pos_tag(nltk.word_tokenize(text))

    output_tokens = []
    for token, tag in tokens_and_tags:
        if tag in ['NN', 'NNP', 'NNS', 'NNPS', 'CD', 'SYM', 'FW', 'IN']:
            output_tokens.append("_")
        elif token in ['$', "''", '(', ')', ',', '--', '.', ':']:
            pass
        else:
            output_tokens.append(token)

    text_skeleton = " ".join(output_tokens)
    text_skeleton = text_skeleton.replace("_ 's", "_")
    text_skeleton = text_skeleton.replace(" 's", "'s")

    while ("_ _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ _", "_")
    while ("_ , _" in text_skeleton):
        text_skeleton = text_skeleton.replace("_ , _", "_")

    if text_skeleton.startswith("_ "):
        text_skeleton = text_skeleton[2:]

    return text_skeleton


def prepare_inputs(prefix_seq, tokenizer, max_prefix_length):
    input_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)["input_ids"]

    if len(input_ids) > max_prefix_length:
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length - 1):]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64)
    }, len(input_ids)


def text2sql_func(model, inputs, tokenizer, max_new_tokens, num_beams, eos_token_id):
    input_length = inputs["input_ids"].shape[1]
    if eos_token_id:
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,  # 4
                use_cache = True,
                eos_token_id = eos_token_id
            ).detach().cpu()
    else:
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_beams,
            ).detach().cpu()


    # print(tokenizer.decode(generate_ids[0]))
    generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    # print(generated_sqls)

    return generated_sqls


# def text2sql_func(model, inputs, tokenizer, max_new_tokens, num_beams, eos_token_id):
#     input_length = inputs["input_ids"].shape[1]
#     with torch.no_grad():
#         generate_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             num_beams=num_beams,
#             num_return_sequences=num_beams,  # 4
#             use_cache = True,
#             eos_token_id = eos_token_id
#         ).detach().cpu()
#
#     # print(tokenizer.decode(generate_ids[0]))
#     generated_sqls = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens=True,
#                                             clean_up_tokenization_spaces=False)
#     # print(generated_sqls)
#
#     return generated_sqls
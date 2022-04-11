from statistics import mean

import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer

model = GPT2Model.from_pretrained("gpt2")


def get_attention(tokens):
    """Given a string, return the attention matrix"""
    output = model(tokens, output_attentions=True)
    return output["attentions"]


def process_attentions(attentions):
    # Concatenate the attentions for each layer
    concatenated = torch.cat(attentions, dim=0)
    # Sum along layers
    mean_attentions = torch.mean(concatenated, dim=(0, 1, 2))
    return mean_attentions


def sort_attentions(attentions, tokens, entities, tokenizer):
    a = attentions.tolist()
    t = tokens.tolist()
    scores = []
    for e in entities:
        score, entity_index = calculate_attention_scores(t, e, tokenizer, a)
        scores.append((e, score, entity_index))
    sorted_scores = sorted(scores, key=lambda x: x[1])
    sorted_scores.reverse()
    return sorted_scores


def calculate_attention_scores(tokens, entity, tokenizer, attentions):
    # Check if at the beginning of sentence
    entity_tokens = tokenizer(entity)["input_ids"]
    entity_indices = subfinder(tokens, entity_tokens)
    if len(entity_indices) > 0:
        try:
            score = mean(attentions[entity_indices[0] : entity_indices[-1] + 1])
        except Exception:
            score = 0
        return score, entity_indices[-1]
    entity_tokens = tokenizer(" " + entity)["input_ids"]
    entity_indices = subfinder(tokens, entity_tokens)
    try:
        score = mean(attentions[entity_indices[0] : entity_indices[-1] + 1])
    except Exception:
        score = 0
    return score, entity_indices[-1]


def subfinder(word_list, pattern):
    word_indices = []
    window_length = len(pattern)
    for i, word1 in enumerate(word_list):
        sublist = word_list[i : i + window_length]
        if sublist == pattern:
            return list(range(i, i + window_length))
    return word_indices


def process_row(row, tokenizer):
    text = row["text"]
    tokens = tokenizer(text)["input_ids"][:model.config.n_positions]
    tokens = torch.LongTensor(tokens)
    attentions = get_attention(tokens)
    attentions = process_attentions(attentions)
    sorted_attentions = sort_attentions(attentions, tokens, row["entities"], tokenizer)

    return sorted_attentions


def process_data(df, save_file, tokenizer, debug=True):
    # Load data
    if debug:
        df = df.iloc[:10]

    with tqdm(total=df.shape[0]) as pbar:
        for index, row in df.iterrows():
            row["entities"] = process_row(row, tokenizer)
            pbar.update(1)
    if not debug:
        df.to_pickle(save_file)
    print("Finished")
    return df


if __name__ == "__main__":
    data = "data/augmented_datasets/pickle/augmented_valid.pkl"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    process_data(data, tokenizer, debug=False)

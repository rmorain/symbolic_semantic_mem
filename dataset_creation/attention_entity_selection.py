import torch
from transformers import GPT2Model

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
        scores.append(calculate_attention_scores(t, e, tokenizer, a))
    sorted_scores = sorted(scores, key=lambda x: x[1])
    sorted_scores.reverse()
    return sorted_scores


def calculate_attention_scores(tokens, entity, tokenizer, attentions):
    t = [tokenizer.decode(x).lower().strip(" ") for x in tokens]
    sub = subfinder(t, entity.lower().split(" "))
    entity_scores = (entity, sum([attentions[i] for i in sub]))
    return entity_scores


def subfinder(word_list, pattern):
    word_indices = []
    window_length = len(pattern)
    for i, word1 in enumerate(word_list):
        sublist = word_list[i : i + window_length]
        if sublist == pattern:
            return list(range(i, i + window_length))
    return word_indices

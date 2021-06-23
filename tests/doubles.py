from kirby.run_params import RunParams
from transformers import GPT2Tokenizer


def KnowledgeInputDouble(text, knowledge):
    run_params = RunParams()
    tokenizer = GPT2Tokenizer.from_pretrained(run_params.model)
    tokenizer.pad_token = tokenizer.eos_token
    input_tokens = tokenize(text, run_params.seq_length, tokenizer)
    knowledge_tokens = tokenize(knowledge, run_params.knowledge_buffer, tokenizer)
    knowledge_input = {
        "input_ids": input_tokens["input_ids"],
        "attention_mask": input_tokens["attention_mask"],
        "knowledge": knowledge_tokens["input_ids"],
        "knowledge_mask": knowledge_tokens["attention_mask"],
    }
    return knowledge_input


# Tokenize a sequence
def tokenize(text, max_length, tokenizer):
    tokens = tokenizer(
        text, return_tensors="pt", padding="max_length", max_length=max_length
    )
    return tokens

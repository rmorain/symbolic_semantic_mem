from torch import LongTensor
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

config = GPT2Config()
model = GPT2Model(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def get_attention(text):
    """Given a string, return the attention matrix"""
    tokens = tokenizer(text)
    tokens = LongTensor(tokens["input_ids"])
    output = model(tokens, output_attentions=True)
    return output["attentions"]

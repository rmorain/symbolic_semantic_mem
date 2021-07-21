import pandas as pd
from transformers import GPT2Tokenizer

df = pd.read_pickle("data/augmented_datasets/pickle/wikiknowledge.pkl")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

for index, row in df.iterrows():
    tokens = tokenizer(row["text"])["input_ids"]
    if len(tokens) > 1024:
        __import__("pudb").set_trace()
        row["text"] = tokenizer.decode(tokens[:1024])
# df.to_pickle("data/augmented_datasets/pickle/wikiknowledge.pkl")

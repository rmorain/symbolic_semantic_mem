import pandas as pd
from datasets import load_dataset
from transformers import GPT2Tokenizer

ds = load_dataset("text", data_files=["data/wiki.valid.raw"])
ds = ds["train"]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
df = ds.to_pandas()

for index, row in df.iterrows():
    tokens = tokenizer(row["text"])["input_ids"]
    if len(tokens) > 1024:
        row["text"] = tokenizer.decode(tokens[:1024])
print("Finished")
df.to_pickle("data/wikitext_valid.pkl")

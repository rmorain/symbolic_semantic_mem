import pandas as pd
from datasets import Dataset
from kirby.run_params import RunParams
from tqdm import tqdm
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
run_params = RunParams()

count = 0


def check(x):
    tokens = tokenizer(x)
    if len(tokens["input_ids"]) > run_params.seq_length:
        return x
    else:
        return None


tqdm.pandas()

df = pd.read_pickle("data/text_knowledge_baseline_valid.pkl")

df.text = df.text.progress_apply(check)
df = df.dropna()
df = df.reset_index(drop=True)
ds = Dataset.from_pandas(df)
df.to_pickle("data/text_knowledge_baseline_fair_valid.pkl")

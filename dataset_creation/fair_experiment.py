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

df = pd.read_pickle("data/text_knowledge_baseline.pkl")

df.text = df.text.progress_apply(check)
df2 = df.dropna()
df3 = pd.read_pickle("data/augmented_datasets/pickle/min_med_max.pkl")
df4 = df2.join(df3, rsuffix="_other")
df4 = df4.drop("text_other", axis=1)
df4 = df4.reset_index(drop=True)
ds = Dataset.from_pandas(df4)
df.to_pickle("data/augmented_datasets/pickle/min_med_max_fair.pkl")

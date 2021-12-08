import pandas as pd
from tqdm import tqdm
from transformers import GPT2Tokenizer

PATH = "data/"
FILE = "wikitext_valid.pkl"
DEST = "text_knowledge_baseline_valid.pkl"
SIZE = 697303
DEBUG = True

df = pd.read_pickle(PATH + FILE)

# Remove blanks
length_filter = df.text.str.len() == 1
length_filter = ~length_filter
df = df[length_filter]

# Remove headings
headings_filter = df.text.str.startswith(" =")
headings_filter = ~headings_filter
df = df[headings_filter]

# ASCII only
ascii_only = df.text.str.contains("[ -~]*")
df = df[ascii_only]

# Reset index
df = df.reset_index(drop=True)

# Reduce size
df = df.iloc[:SIZE]

# Create two columns
tqdm.pandas()
df.progress_apply(tokenize)


if not DEBUG:
    df.to_pickle(PATH + DEST)
else:
    __import__("pudb").set_trace()

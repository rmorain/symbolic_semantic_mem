import pandas as pd

PATH = "data/augmented_datasets/pickle/"
ds = "max_attention_valid"

df = pd.read_pickle(PATH + ds + ".pkl")

# Remove disambiguation pages
text_filter = df.knowledge.str.contains("Wikimedia disambiguation page")
text_filter = ~text_filter
df = df[text_filter]

# Remove numbers
text_filter = df.knowledge.str.contains("playing card")
text_filter = ~text_filter
df = df[text_filter]

# Reset index
df = df.reset_index(drop=True)

df.to_pickle(PATH + ds + "_clean.pkl")

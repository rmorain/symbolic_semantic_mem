import pandas as pd
from tqdm import tqdm

f = "data/augmented_datasets/pickle/wikidata_with_max_attention_entity_selection_valid.pkl"
df = pd.read_pickle(f)
debug = True
with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        if not isinstance(row["knowledge"], str):
            row["knowledge"] = "None"

df.to_pickle(f)
print("Finished")

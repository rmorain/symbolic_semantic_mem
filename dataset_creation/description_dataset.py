import json

import pandas as pd
from tqdm import tqdm


def get_description(row):
    """Return the description of the first entity that has one"""
    ks = row["knowledge"]
    for k in ks:
        try:
            keys = ["label", "description"]
            # Grab attributes we care about
            k_dict = {key: k[key] for key in keys}
            # Save as JSON string
            row["knowledge"] = json.dumps(k_dict)
            return row
        except Exception:
            pass


df = pd.read_pickle("data/augmented_datasets/pickle/combined_complete.pkl")

with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        row = get_description(row)
        pbar.update(1)

df.to_pickle("data/augmented_datasets/pickle/description.pkl")

import json

import pandas as pd
from datasets import Dataset
from tqdm import tqdm


def get_knowledge(entity_string, knowledge_list):
    for k in knowledge_list:
        if k and k["label"] == entity_string:
            return k["description"]
    return None


df = pd.read_pickle("data/augmented_datasets/pickle/sorted_attentions.pkl")
debug = True
count = 0
with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        description = None
        while row["entities"]:
            # Get median entity
            entity = row["entities"].pop(len(row["entities"]) // 2)[0]
            description = get_knowledge(entity, row["knowledge"])
            if description:
                k_dict = {"label": entity, "description": description}
                # Save as JSON string
                row["knowledge"] = json.dumps(k_dict)
                break
        if not isinstance(row["knowledge"], str):
            count += 1
            row["knowledge"] = "No info"
        pbar.update(1)
df = df.drop("entities", axis=1)
if not debug:
    df.to_pickle("data/augmented_datasets/pickle/median_attention.pkl")
try:
    ds = Dataset.from_pandas(df)
except Exception as e:
    raise e
print(count / df.shape[0] * 100)
print("Finished")

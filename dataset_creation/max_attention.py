import json

import pandas as pd
from tqdm import tqdm
from datasets import Dataset


def get_knowledge(entity_string, knowledge_list):
    for k in knowledge_list:
        if k and k["label"] == entity_string:
            return k["description"]
    return None


df = pd.read_pickle("data/augmented_datasets/pickle/sorted_attentions_valid.pkl")
debug = False
count = 0
with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        description = None
        for entity in row["entities"]:
            max_entity = entity[0]
            description = get_knowledge(max_entity, row["knowledge"])
            if description:
                k_dict = {"label": max_entity, "description": description}
                # Save as JSON string
                row["knowledge"] = json.dumps(k_dict)
                break
        if not isinstance(row['knowledge'], str):
            count += 1
            row["knowledge"] = "No info"
        pbar.update(1)
df = df.drop("entities", axis=1)
if not debug:
    df.to_pickle("data/augmented_datasets/pickle/max_attention.pkl")

try:
    ds = Dataset.from_pandas(df)
except Exception as e:
    for index, row  in df.iterrows():
        if not isinstance(row['knowledge'], str):
            __import__('pudb').set_trace()
    raise e
print("Finished")
print(count)
print(df.shape[0])

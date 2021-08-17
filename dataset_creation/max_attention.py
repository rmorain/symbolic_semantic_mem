import json

import pandas as pd
from tqdm import tqdm


def get_knowledge(entity_string, knowledge_list):
    for k in knowledge_list:
        if k and k["label"] == entity_string:
            return k["description"]
    return None


df = pd.read_pickle("data/augmented_datasets/pickle/sorted_attention.pkl")
debug = False
with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        for entity in row["entities"]:
            max_entity = entity[0]
            description = get_knowledge(max_entity, row["knowledge"])
            if description:
                k_dict = {"label": max_entity, "description": description}
                # Save as JSON string
                row["knowledge"] = json.dumps(k_dict)
                break
        if description is None:
            row["knowledge"] = ""
        pbar.update(1)
df = df.drop("entities", axis=1)
if not debug:
    df.to_pickle("data/augmented_datasets/pickle/max_attention.pkl")
print("Finished")

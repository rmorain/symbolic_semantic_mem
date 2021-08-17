import json

import pandas as pd
from tqdm import tqdm


def get_knowledge(entity_string, knowledge_list):
    for k in knowledge_list:
        if k and k["label"] == entity_string:
            return k["description"]
    return None


df = pd.read_pickle("data/augmented_datasets/pickle/sorted_attentions_valid.pkl")
debug = False
with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        description = None
        # Reverse entities to get min
        row["entities"].sort(reverse=False, key=lambda x: x[1])
        for entity in row["entities"]:
            entity = entity[0]
            description = get_knowledge(entity, row["knowledge"])
            if description:
                k_dict = {"label": entity, "description": description}
                # Save as JSON string
                row["knowledge"] = json.dumps(k_dict)
                break
        if description is None:
            row["knowledge"] = "{No info}"
        pbar.update(1)
df = df.drop("entities", axis=1)
if not debug:
    df.to_pickle("data/augmented_datasets/pickle/min_attention_valid.pkl")
print("Finished")

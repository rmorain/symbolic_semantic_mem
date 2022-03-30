import json

from datasets import Dataset
from tqdm import tqdm


def get_knowledge(entity_string, knowledge_list):
    for k in knowledge_list:
        if k and k["label"] == entity_string:
            return k["description"]
    return None


def process_max_attention(df, save_file, debug=True):
    count = 0
    with tqdm(total=df.shape[0]) as pbar:
        entity_index = []
        for index, row in df.iterrows():
            description = None
            for entity in row["entities"]:
                max_entity = entity[0]
                description = get_knowledge(max_entity, row["knowledge"])
                if description:
                    k_dict = {"label": max_entity, "description": description}
                    # Save as JSON string
                    row["knowledge"] = json.dumps(k_dict)
                    entity_index.append(entity[-1])
                    break
            if not isinstance(row["knowledge"], str):
                count += 1
                row["knowledge"] = "No info"
                entity_index.append(-1)
            pbar.update(1)
    df = df.drop("entities", axis=1)
    df["entity_index"] = entity_index
    if not debug:
        df.to_pickle(save_file)
    try:
        Dataset.from_pandas(df)
    except Exception as e:
        for index, row in df.iterrows():
            if not isinstance(row["knowledge"], str):
                __import__("pudb").set_trace()
        raise e
    print("Finished")
    print(count)
    print(df.shape[0])
    return df

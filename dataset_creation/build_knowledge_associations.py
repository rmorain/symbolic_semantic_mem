import pandas as pd
from kirby.entities_client import EntitiesClient
from tqdm import tqdm


def filter_knowledge(k):
    if not k:  # K is none
        return k
    stopwords = [
        "disambiguation page",
        "playing card",
    ]
    for sw in stopwords:
        if sw in k["description"]:
            return None
    return k


def add_knowledge(example, db=None):
    """
    Adds a `knowledge` column to the data point
    Knowledge is a list of dictionaries.

    Example:
        [
            {
                'description':'A very good description.',
                'association': 'More associations',
                ...
            }
        ]
    """
    knowledge_list = []
    for i, entity in enumerate(example["entities"]):
        k = db.find_by_label(entity[0])
        knowledge_list.append(k)
    example["entities"] = [
        example["entities"][i]
        for i in range(len(example["entities"]))
        if knowledge_list[i]
    ]
    example["knowledge"] = [
        knowledge_list[i] for i in range(len(knowledge_list)) if knowledge_list[i]
    ]
    return example


# Load dataset
def process_knowledge(data_path, save_path, debug=True):
    df = pd.read_pickle(data_path)
    if debug:
        df = df[:10]

    knowledge_list = [None for i in range(df.shape[0])]
    df["knowledge"] = knowledge_list
    tqdm.pandas(desc="Just getting knowledge.")

    entity_client = EntitiesClient()

    df.progress_apply(add_knowledge, db=entity_client, axis=1)

    if not debug:
        df.to_pickle(save_path)
    return df

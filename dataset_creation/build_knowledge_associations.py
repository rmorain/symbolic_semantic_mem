import pandas as pd
from datasets import Dataset
from kirby.database_proxy import WikiDatabase
from kirby.run_params import RunParams
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
    for entity in example["entities"]:
        k = db.get_knowledge(entity)
        k = filter_knowledge(k)
        knowledge_list.append(k)
    example["knowledge"] = knowledge_list
    return example


# Load dataset
def process_knowledge(data_path, save_path, debug=True):
    ds = pd.DataFrame.from_dict(Dataset.load_from_disk(data_path))
    if debug:
        ds = ds[:10]

    knowledge_list = [None for i in range(ds.shape[0])]
    ds["knowledge"] = knowledge_list

    rp = RunParams()
    db = WikiDatabase(rp)
    with tqdm(total=ds.shape[0]) as pbar:
        for index, row in ds.iterrows():
            row = add_knowledge(row, db)
            pbar.update(1)

    if not debug:
        ds.to_pickle("data/augmented_datasets/pickle/augmented_valid.pkl")
    return ds

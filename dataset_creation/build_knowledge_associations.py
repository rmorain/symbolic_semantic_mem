import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from kirby.database_proxy import WikiDatabase
from kirby.run_params import RunParams


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
        knowledge_list.append(k)
    example["knowledge"] = knowledge_list
    return example


# Load dataset
split = "train"
save_location = "data/augmented_datasets/entities/" + split + "/"
ds = pd.DataFrame.from_dict(Dataset.load_from_disk(save_location))

# Split in half
ds = ds.iloc[: ds.shape[0] // 3]
# Augment data
# Add `knowledge` column
knowledge_list = [None for i in range(ds.shape[0])]
ds["knowledge"] = knowledge_list

rp = RunParams()
db = WikiDatabase(rp)
with tqdm(total=ds.shape[0]) as pbar:
    for index, row in ds.iterrows():
        row = add_knowledge(row, db)
        pbar.update(1)
        if index % 100 == 0:
            ds.to_pickle("data/augmented_datasets/pickle/augmented_train1.pkl")

# Save
ds.to_pickle("data/augmented_datasets/pickle/augmented_train1.pkl")

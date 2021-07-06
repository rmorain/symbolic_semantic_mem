# Build description dataset
import pandas as pd
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


# Pickle files
f1 = "data/augmented_datasets/pickle/combined1.pkl"

# Load pickle files
df = pd.read_pickle(f1)

rp = RunParams()
db = WikiDatabase(rp)
# Add knowledge in neccessary
with tqdm(total=df.shape[0]) as pbar:
    for index, row in df.iterrows():
        if index >= 635201 and index < 709978:
            row = add_knowledge(row, db)
        pbar.update(1)
df.to_pickle(f1)
print("Finished")

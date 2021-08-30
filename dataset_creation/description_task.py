import pandas as pd
from tqdm import tqdm

data_list = []


def get_descriptions(row):
    global data_list
    for i, label in enumerate(row["entities"]):
        description = row["knowledge"][i]
        if description:
            data_list.append(
                {"label": label, "description": description["description"]}
            )


debug = False

tqdm.pandas(desc="Progress")
df = pd.read_pickle("data/augmented_datasets/pickle/wikiknowledge.pkl")
if debug:
    df = df.iloc[:10]

df = df.progress_apply(get_descriptions, axis=1)

new_df = pd.DataFrame(data_list).drop_duplicates()
if not debug:
    new_df.to_pickle("data/augmented_datasets/pickle/label_description.pkl")
else:
    __import__("pudb").set_trace()

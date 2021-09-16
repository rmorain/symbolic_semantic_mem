import pandas as pd
from tqdm import tqdm

data_list = []


def get_questions(row):
    global data_list
    random_samples = df.sample(n=num_choices - 1)
    distractors = random_samples["description"].tolist()
    data = {
        "question": "What is " + row["label"] + "?",
        "correct": row["description"],
        "distractors": distractors,
        "knowledge": "{" + row["label"] + " : " + row["description"] + "}",
    }
    data_list.append(data)


debug = False
num_choices = 4

tqdm.pandas(desc="Progress")
df = pd.read_pickle("data/augmented_datasets/pickle/label_description.pkl")
if debug:
    df = df.iloc[:10]

df = df.progress_apply(get_questions, axis=1)

new_df = pd.DataFrame(data_list)
if not debug:
    new_df.to_pickle("data/augmented_datasets/pickle/description_qa_knowledge.pkl")
else:
    __import__("pudb").set_trace()

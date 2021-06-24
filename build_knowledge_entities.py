import spacy

from kirby.data_manager import DataManager
from kirby.run_params import RunParams


def convert_to_strings(entities):
    """
    Given a tuple of entities, return a list of strings
    """
    return [e.text for e in entities]


def get_entities(example):
    """
    Augment the data by adding an `entities` column to the dataset
    """
    entities = nlp(example["text"]).ents
    example["entities"] = convert_to_strings(entities)
    return example


# Load dataset
run_params = RunParams(debug=False)
split = "valid"
dm = DataManager(run_params)
ds = dm.load(split)
save_location = "data/augmented_datasets/entities/" + split + "/"
nlp = spacy.load(
    "en_core_web_sm",
    disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
)

# Augment data
ds = ds.map(get_entities, batched=False, num_proc=4)

# Save and load
ds.save_to_disk(save_location)

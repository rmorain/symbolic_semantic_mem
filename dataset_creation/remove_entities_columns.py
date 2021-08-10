import pandas as pd

f = "data/augmented_datasets/pickle/wikidata_with_max_attention_entity_selection_valid.pkl"
df = pd.read_pickle(f)

df = df.drop("entities", axis=1)
df.to_pickle(f)

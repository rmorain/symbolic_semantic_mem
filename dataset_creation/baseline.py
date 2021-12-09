import pandas as pd

PATH = "data/augmented_datasets/pickle/"
FILE = "min_attention.pkl"
DEST = "baseline.pkl"

df = pd.read_pickle(PATH + FILE)
df = df.drop("knowledge", axis=1)

df.to_pickle(PATH + DEST)

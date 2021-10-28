import pandas as pd
import tqdm

min_df = pd.read_pickle("data/augmented_datasets/pickle/min_attention.pkl")
med_df = pd.read_pickle("data/augmented_datasets/pickle/median_attention.pkl")

min_df["knowledge2"] = med_df.knowledge

pd.to_pickle(min_df, "data/augmented_datasets/pickle/min_med.pkl")

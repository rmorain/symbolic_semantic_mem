import pandas as pd

PATH = "data/augmented_datasets/pickle/"

min_df = pd.read_pickle(PATH + "min_attention.pkl")
med_df = pd.read_pickle(PATH + "median_attention.pkl")
max_df = pd.read_pickle(PATH + "max_attention.pkl")

debug = False

# min_med
min_med = pd.DataFrame()
min_med["text"] = min_df.text
min_med["min"] = min_df.knowledge
min_med["median"] = med_df.knowledge
if not debug:
    pd.to_pickle(min_med, PATH + "min_med.pkl")
else:
    __import__("pudb").set_trace()

# min_max
min_max = pd.DataFrame()
min_max["text"] = min_df.text
min_max["min"] = min_df.knowledge
min_max["max"] = max_df.knowledge
if not debug:
    pd.to_pickle(min_max, PATH + "min_max.pkl")

# med_max
med_max = pd.DataFrame()
med_max["text"] = med_df.text
med_max["median"] = med_df.knowledge
med_max["max"] = max_df.knowledge
if not debug:
    pd.to_pickle(med_max, PATH + "med_max.pkl")

# min_med_max
min_med_max = pd.DataFrame()
min_med_max["text"] = med_df.text
min_med_max["min"] = min_df.knowledge
min_med_max["median"] = med_df.knowledge
min_med_max["max"] = max_df.knowledge
if not debug:
    pd.to_pickle(min_med_max, PATH + "min_med_max.pkl")

import pandas as pd

data_path = "data/augmented_datasets/pickle/"
dfs = [
    data_path + "label_description.pkl",
]
__import__("pudb").set_trace()
for df_name in dfs:
    df = pd.read_pickle(df_name)
    print("old shape: ", df.shape)
    for col in df.columns:
        ascii_only = df[col].str.contains("[ -~]*")
        df = df[ascii_only]
        print("new shape: ", df.shape)

    # df.to_pickle(df_name)

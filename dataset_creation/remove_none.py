import pandas as pd
from kirby.data_manager import DataManager
from kirby.run_params import RunParams
from tqdm import tqdm
from transformers import GPT2Tokenizer

data_files = {
    "train": ["data/augmented_datasets/pickle/min_attention.pkl"],
    "valid": ["data/augmented_datasets/pickle/min_attention_valid.pkl"],
}
run_params = RunParams(
    run_name="min_attention",
    debug=False,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
)
dm = DataManager(run_params)
# tds, vds = dm.prepare_data()
debug = True
df = pd.read_pickle(data_files["train"][0])
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
with tqdm(total=df.shape[0]) as pbar:
    for i, row in df.iterrows():
        if row["text"] is None:
            __import__("pudb").set_trace()
        if row["knowledge"] is None:
            __import__("pudb").set_trace()

        pbar.update(1)

# df.to_pickle(f)
print("Finished")

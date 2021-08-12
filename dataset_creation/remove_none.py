import pandas as pd
from kirby.data_manager import DataManager
from kirby.run_params import RunParams
from tqdm import tqdm
from transformers import GPT2Tokenizer

data_files = {
    "train": [
        "data/augmented_datasets/pickle/wikidata_with_max_attention_entity_selection.pkl"
    ],
    "valid": [
        "data/augmented_datasets/pickle/wikidata_with_max_attention_entity_selection_valid.pkl"
    ],
}
run_params = RunParams(
    run_name="max_attention",
    debug=False,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
)
dm = DataManager(run_params)
tds, vds = dm.prepare_data()
debug = True
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
with tqdm(total=vds.shape[0]) as pbar:
    for i, row in enumerate(vds):
        if row["input_ids"][0].shape[0] != 192:
            __import__("pudb").set_trace()
        pbar.update(1)

# df.to_pickle(f)
print("Finished")

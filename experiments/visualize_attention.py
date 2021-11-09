import numpy as np
import pandas as pd
import torch
from kirby.data_manager import DataManager
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams
from tqdm import tqdm

data_files = {
    "train": ["data/augmented_datasets/pickle/max_attention.pkl"],
    "valid": ["data/augmented_datasets/pickle/max_attention_valid.pkl"],
}
run_params = RunParams(
    run_name="max_attention_visualize",
    debug=False,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
    max_epochs=1,
    output_attentions=True,
    num_workers=1,
)
model = KnowledgeModel(run_params)
run_label = "epoch=23-val_loss=0.92-max_attention"
PATH = f"checkpoints/{run_label}.ckpt"

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["state_dict"])
# Freeze weights
for param in model.parameters():
    param.requires_grad = False

dm = DataManager(run_params)

ds = dm.prepare_ds("valid")

running_total = torch.zeros(run_params.seq_length + run_params.knowledge_buffer)
for i in tqdm(range(len(ds))):
    x = ds[i]
    attentions = model(x)
    attentions = torch.cat(attentions)
    attentions = torch.mean(attentions, dim=(0, 1, 2))
    running_total += attentions

avg_attentions = running_total / len(ds)
if not run_params.debug:
    torch.save(avg_attentions, "data/attention_values/" + run_label + ".pt")
    avg_attentions_np = avg_attentions.numpy()
    avg_attentions_df = pd.DataFrame(avg_attentions_np)
    avg_attentions_df.to_csv("data/attention_values/" + run_label + ".csv")
else:
    __import__("pudb").set_trace()

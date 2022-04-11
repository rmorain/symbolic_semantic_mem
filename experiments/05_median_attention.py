from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

data_files = {
    "train": ["volume/data/augmented_datasets/pickle/median_attention_clean.pkl"],
    "valid": ["volume/data/augmented_datasets/pickle/median_attention_valid_clean.pkl"],
}
run_params = RunParams(
    run_name="median_attention_medium",
    debug=False,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
    batch_size=32,
    num_workers=12,
    model="gpt2-medium",
)
model = KnowledgeModel(run_params)

experiment = Experiment(run_params, model)
experiment.run()

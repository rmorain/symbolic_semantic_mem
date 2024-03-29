from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

data_files = {
    "train": ["data/augmented_datasets/pickle/min_attention_clean.pkl"],
    "valid": ["data/augmented_datasets/pickle/min_attention_valid_clean.pkl"],
}
run_params = RunParams(
    run_name="min_attention",
    debug=True,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
    num_workers=1,
)
model = KnowledgeModel(run_params)

experiment = Experiment(run_params, model)
experiment.run()

from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

data_files = {
    "train": ["data/augmented_datasets/pickle/min_med.pkl"],
    "valid": ["data/augmented_datasets/pickle/min_med_valid.pkl"],
}
run_params = RunParams(
    run_name="min_med",
    debug=False,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
    random_seed=3,
    patience=3,
    num_workers=8,
    batch_size=32,
)
model = KnowledgeModel(run_params)

experiment = Experiment(run_params, model)
experiment.run()

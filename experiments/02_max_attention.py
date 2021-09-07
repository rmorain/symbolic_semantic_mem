from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

data_files = {
    "train": [
        "data/augmented_datasets/pickle/max_attention.pkl"
    ],
    "valid": [
        "data/augmented_datasets/pickle/max_attention_valid.pkl"
    ],
}
run_params = RunParams(
    run_name="max_attention",
    debug=False,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
    num_gpus=7,
)
model = KnowledgeModel(run_params)

experiment = Experiment(run_params, model)
experiment.run()

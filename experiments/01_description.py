from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

data_files = {
    "train": ["data/augmented_datasets/pickle/description.pkl"],
    "valid": ["data/augmented_datasets/pickle/description_valid.pkl"],
}
run_params = RunParams(
    run_name="description",
    debug=True,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
    knowledge_front=True,
)
model = KnowledgeModel(run_params)
experiment = Experiment(run_params, model)
experiment.run()

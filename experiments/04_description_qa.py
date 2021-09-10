from kirby.experiment import Experiment
from kirby.qa_model import QAModel
from kirby.run_params import RunParams

data_files = {
    "train": "data/augmented_datasets/pickle/description_qa_train.pkl",
    "valid": "data/augmented_datasets/pickle/description_qa_val.pkl",
}
run_params = RunParams(
    run_name="description_qa_test",
    debug=True,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=False,
    batch_size=8,
)
model = QAModel(run_params)

experiment = Experiment(run_params, model)
experiment.run()

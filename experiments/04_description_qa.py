from kirby.experiment import Experiment
from kirby.qa_model import QAModel
from kirby.run_params import RunParams

data_file = "data/augmented_datasets/pickle/description_qa_knowledge.pkl"
run_params = RunParams(
    run_name="description_qa_knowledge_20",
    debug=False,
    pretrained=True,
    data_files=data_file,
    data_file_type="pandas",
    knowledge_tokenize=True,
    mask_percent=0.2,
    batch_size=8,
)
model = QAModel(run_params)

experiment = Experiment(run_params, model)
experiment.run()

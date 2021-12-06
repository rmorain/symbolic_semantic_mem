from kirby.experiment import Experiment
from kirby.knowledge_bert import KnowledgeBert
from kirby.run_params import RunParams

data_files = {
    "train": ["data/augmented_datasets/pickle/min_attention_clean.pkl"],
    "valid": ["data/augmented_datasets/pickle/min_attention_valid_clean.pkl"],
}
run_params = RunParams(
    run_name="min_attention_bert",
    debug=False,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
    model="bert-base-uncased",
)
model = KnowledgeBert(run_params)

experiment = Experiment(run_params, model)
experiment.run()

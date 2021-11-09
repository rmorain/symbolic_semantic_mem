from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

num_runs = 10
data_files = {
    "train": ["data/augmented_datasets/pickle/max_attention.pkl"],
    "valid": ["data/augmented_datasets/pickle/max_attention_valid.pkl"],
}

for i in range(num_runs):
    run_params = RunParams(
        run_name="max_attention_" + str(i),
        debug=False,
        pretrained=True,
        data_files=data_files,
        data_file_type="pandas",
        knowledge_tokenize=True,
    )
    model = KnowledgeModel(run_params)

    experiment = Experiment(run_params, model)
    experiment.run()

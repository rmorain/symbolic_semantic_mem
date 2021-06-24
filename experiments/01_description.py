from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

data_files = {"train": ["data/augmented_datasets/wikitext_with_knowledge.csv"]}
run_params = RunParams(
    run_name="description",
    debug=True,
    pretrained=True,
    data_files=data_files,
    data_file_type="csv",
)
model = KnowledgeModel()
experiment = Experiment(run_params)
experiment.run()

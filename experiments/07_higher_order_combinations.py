import sys

from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

# Take in file from command line
PATH = "volume/data/augmented_datasets/pickle/"
FILE = sys.argv[1]


data_files = {
    "train": [PATH + FILE + ".pkl"],
    "valid": [PATH + FILE + "_valid.pkl"],
}
run_params = RunParams(
    run_name=FILE,
    debug=True,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
    knowledge_tokenize=True,
    num_workers=1
)
model = KnowledgeModel(run_params)

experiment = Experiment(run_params, model)
experiment.run()

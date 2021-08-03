from kirby.basic_model import BasicModel
from kirby.experiment import Experiment
from kirby.run_params import RunParams

data_files = {
    "train": ["data/wikitext_train.pkl"],
    "valid": ["data/wikitext_valid.pkl"],
}
run_params = RunParams(
    run_name="baseline",
    debug=False,
    pretrained=True,
    data_files=data_files,
    data_file_type="pandas",
)
model = BasicModel(run_params)
experiment = Experiment(run_params, model)
experiment.run()

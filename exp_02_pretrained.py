from kirby.experiment import Experiment
from kirby.run_params import RunParams
import wandb

wandb.login()

run_params = RunParams(run_name='no_augmentation_pretrained', debug=False, pretrained=True)
experiment = Experiment(run_params)
experiment.run()

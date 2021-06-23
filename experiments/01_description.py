from kirby.experiment import Experiment
from kirby.run_params import RunParams

wandb.login()

run_params = RunParams(run_name='description', debug=True)
experiment = Experiment(run_params)
experiment.run()

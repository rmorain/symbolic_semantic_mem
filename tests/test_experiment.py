import unittest

from kirby.experiment import Experiment
from kirby.run_params import RunParams
from kirby.basic_model import BasicModel


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.run_params = RunParams()
        self.model = BasicModel(self.run_params)
        self.experiment = Experiment(self.run_params, self.model)

    def tearDown(self):
        pass

    def test_with_debug_true(self):
        self.experiment.run()

    def test_with_debug_false(self):
        self.run_params.debug = False
        self.run_params.max_epochs = 1
        self.run_params.data_set_percentage = 1
        self.experiment.run()


if __name__ == "__main__":
    unittest.main()

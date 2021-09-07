import pickle
import unittest

from kirby.experiment import Experiment
from kirby.qa_model import QAModel
from kirby.run_params import RunParams


class TestQAModel(unittest.TestCase):
    def setUp(self):
        data_path = "data/augmented_datasets/pickle/"
        self.run_params = RunParams(
            data_files={"train": data_path + "description_qa.pkl"},
            debug=True,
        )
        self.model = QAModel(self.run_params)
        self.x = "tests/pickle/description_qa_x.pkl"

    def tearDown(self):
        pass

    def test_qa(self):
        pass

    def test_setup(self):
        self.model.setup(stage=None)
        self.assertIsNotNone(self.model.train_ds)
        with open(self.x, "wb") as x:
            pickle.dump(self.model.train_ds[3], x)

    def test_forward(self):
        with open(self.x, "rb") as x:
            x = pickle.load(x)
        outputs = self.model.forward(x)
        self.assertIsNotNone(outputs)

    def test_debug_experiment(self):
        data_files = {
            "train": "data/augmented_datasets/pickle/description_qa_train.pkl",
            "valid": "data/augmented_datasets/pickle/description_qa_val.pkl",
        }
        run_params = RunParams(
            run_name="description_qa",
            debug=True,
            pretrained=True,
            data_files=data_files,
            data_file_type="pandas",
            knowledge_tokenize=False,
            batch_size=1,
        )
        model = QAModel(run_params)

        experiment = Experiment(run_params, model)
        experiment.run()


if __name__ == "__main__":
    unittest.main()

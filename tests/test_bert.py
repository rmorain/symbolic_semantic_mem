import unittest

from kirby.experiment import Experiment
from kirby.knowledge_bert import KnowledgeBert
from kirby.run_params import RunParams


class TestExperiment(unittest.TestCase):
    def setUp(self):
        data_files = {
            "train": ["data/augmented_datasets/pickle/min_attention_clean.pkl"],
            "valid": ["data/augmented_datasets/pickle/min_attention_valid_clean.pkl"],
        }
        self.run_params = RunParams(
            run_name="test",
            debug=True,
            pretrained=True,
            data_files=data_files,
            data_file_type="pandas",
            model="bert-base-uncased",
            knowledge_tokenize=True,
        )
        self.model = KnowledgeBert(self.run_params)
        self.experiment = Experiment(self.run_params, self.model)

    def tearDown(self):
        pass

    def test_with_debug_true(self):
        self.experiment.run()


if __name__ == "__main__":
    unittest.main()

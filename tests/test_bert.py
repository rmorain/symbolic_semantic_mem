import unittest

from kirby.experiment import Experiment
from kirby.knowledge_bert import KnowledgeBert
from kirby.run_params import RunParams


class TestBERTExperiment(unittest.TestCase):
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
            bert=True,
            knowledge_tokenize=True,
            num_workers=1,
        )
        self.model = KnowledgeBert(self.run_params)
        self.experiment = Experiment(self.run_params, self.model)

    def tearDown(self):
        pass

    def test_with_debug_true(self):
        self.experiment.run()

    def test_baseline(self):
        data_files = {
            "train": ["data/wikitext_train.pkl"],
            "valid": ["data/wikitext_valid.pkl"],
        }
        run_params = RunParams(
            run_name="test",
            debug=True,
            pretrained=True,
            data_files=data_files,
            data_file_type="pandas",
            model="bert-base-uncased",
            bert=True,
            knowledge_tokenize=False,
            num_workers=1,
        )
        self.model = KnowledgeBert(run_params)
        experiment = Experiment(self.run_params, self.model)

        experiment.run()


if __name__ == "__main__":
    unittest.main()

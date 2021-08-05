import unittest

from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

from tests.doubles import KnowledgeInputDouble


class TestKnowledgeModel(unittest.TestCase):
    def setUp(self):
        self.run_params = RunParams()
        self.model = KnowledgeModel(self.run_params)
        self.test_input = KnowledgeInputDouble(
            "Stephen Curry is my favorite basketball player",
            "{Stephen Curry: {description: American basketball player}}",
        )

    def tearDown(self):
        pass

    # def test_knowledge_model_forward_with_knowledge_input(self):
    # output = self.model(self.test_input)
    # self.assertIsNotNone(output)

    def test_knowledge_model_experiment(self):
        data_files = {
            "train": ["data/augmented_datasets/pickle/description.pkl"],
            "valid": ["data/augmented_datasets/pickle/description_valid.pkl"],
        }
        run_params = RunParams(
            run_name="description",
            debug=True,
            pretrained=True,
            knowledge_tokenize=True,
            data_files=data_files,
            data_file_type="pandas",
        )
        model = KnowledgeModel(run_params)
        experiment = Experiment(run_params, model)
        experiment.run()


if __name__ == "__main__":
    unittest.main()

import unittest

import pytorch_lightning as pl
from kirby.experiment import Experiment
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

from tests.doubles import KnowledgeInputDouble


class TestKnowledgeModel(unittest.TestCase):
    def setUp(self):
        self.run_params = RunParams(
            run_name="description",
            debug=True,
            pretrained=True,
            knowledge_tokenize=True,
        )
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

    def test_generate(self):
        prompt = "Harry Potter raised his wand and said "
        max_new_tokens = 30
        model = KnowledgeModel(self.run_params)
        generated_text = model.generate(prompt, max_new_tokens)

    def test_knowledge_model(self):
        data_files = {
            "train": ["data/augmented_datasets/pickle/max_attention.pkl"],
            "valid": ["data/augmented_datasets/pickle/max_attention_valid.pkl"],
        }
        run_params = RunParams(
            run_name="description",
            debug=True,
            pretrained=True,
            knowledge_tokenize=True,
            data_files=data_files,
            data_file_type="pandas",
            num_gpus=1,
        )
        model = KnowledgeModel(run_params)
        trainer = pl.Trainer(
            fast_dev_run=self.run_params.debug,
        )
        trainer.fit(model)


if __name__ == "__main__":
    unittest.main()

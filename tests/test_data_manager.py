import unittest

from datasets import load_dataset
from kirby.data_manager import DataManager
from kirby.run_params import RunParams
from transformers import GPT2Tokenizer


class TestDataManager(unittest.TestCase):
    def setUp(self):
        self.run_params = RunParams()
        self.dm = DataManager(self.run_params)
        self.train_ds, self.valid_ds = self.dm.prepare_data()

    def tearDown(self):
        pass

    def test_train_ds_input_ids_length(self):
        """Each sequence in the dataset should have the same length"""
        for x in self.train_ds:
            self.assertEqual(x["input_ids"][0].shape[-1], self.run_params.seq_length)

    def test_valid_ds_input_ids_length(self):
        """Each sequence in the dataset should have the same length"""
        for x in self.valid_ds:
            self.assertEqual(x["input_ids"][0].shape[-1], self.run_params.seq_length)

    # def test_split_debug(self):
    # self.assertEqual(self.train_ds.shape[0], 484)

    # def test_split_percentage(self):
    # dm = DataManager(RunParams(debug=False, data_set_percentage=1))
    # train_ds, valid_ds = dm.prepare_data()
    # self.assertEqual(train_ds.shape[0], 8531)
    # self.assertEqual(valid_ds.shape[0], 20)

    def test_knowledge_loading(self):
        data_file = "data/augmented_datasets/pickle/description.pkl"
        ds = load_dataset("pandas", data_files=data_file)
        self.assertIsNotNone(ds)

    def test_tokenize(self):
        # Test with text only
        x = {"text": "Stephen Curry is my favorite basketball player"}
        tokenizer = GPT2Tokenizer.from_pretrained(self.run_params.model)
        tokenizer.pad_token = tokenizer.eos_token
        tokens = self.dm.tokenize(x, tokenizer=tokenizer)
        self.assertIsNotNone(tokens)

        # Test with text and knowledge
        x[
            "knowledge"
        ] = "{label: Stephen Curry, description: American basketball player}"
        tokens = self.dm.tokenize(x, tokenizer=tokenizer)
        self.assertIsNotNone(tokens)

    def test_knowledge_tokenizing(self):
        data_files = {
            "train": ["data/augmented_datasets/pickle/description.pkl"],
            "valid": ["data/augmented_datasets/pickle/description_valid.pkl"],
        }
        dm = DataManager(RunParams(data_files=data_files, data_file_type="pandas"))
        train_ds, valid_ds = dm.prepare_data()
        self.assertIsNotNone(train_ds)
        self.assertIsNotNone(valid_ds)

    def test_knowledge_tokenizer_higher_order(self):
        data_files = {
            "train": ["data/augmented_datasets/pickle/min_med.pkl"],
            "valid": ["data/augmented_datasets/pickle/min_med.pkl"],
        }
        dm = DataManager(
            RunParams(
                data_files=data_files, data_file_type="pandas", knowledge_tokenize=True
            )
        )
        train_ds, _ = dm.prepare_data()
        for x in train_ds:
            total_length = self.run_params.seq_length + self.run_params.knowledge_buffer
            self.assertEqual(x["input_ids"][0].shape[-1], total_length)
            self.assertEqual(x["attention_mask"][0].shape[-1], total_length)
            self.assertEqual(x["labels"][0].shape[-1], total_length)


if __name__ == "__main__":
    unittest.main()

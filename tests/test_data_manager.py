import unittest

from kirby.data_manager import DataManager
from kirby.run_params import RunParams


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
            self.assertEqual(x["input_ids"].shape[-1], self.run_params.seq_length)

    def test_valid_ds_input_ids_length(self):
        """Each sequence in the dataset should have the same length"""
        for x in self.valid_ds:
            self.assertEqual(x["input_ids"].shape[-1], self.run_params.seq_length)

    def test_split_debug(self):
        self.assertEqual(self.train_ds.shape[0], 484)

    def test_split_percentage(self):
        dm = DataManager(RunParams(debug=False, data_set_percentage=1))
        train_ds, valid_ds = dm.prepare_data()

        self.assertEqual(train_ds.shape[0], 8820)
        self.assertEqual(valid_ds.shape[0], 13)

    def test_knowledge_tokenizing(self):
        __import__("pudb").set_trace()
        data_file = "data/augmented_datasets/pickle/description.pkl"
        run_params = RunParams(data_file_type="pandas", data_files=data_file)
        dm = DataManager(run_params)
        train_ds, valid_ds = dm.prepare_knowledge_ds()


if __name__ == "__main__":
    unittest.main()

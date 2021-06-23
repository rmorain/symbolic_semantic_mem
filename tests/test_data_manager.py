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
        """Each sequence in the dataset should have the same length
        """
        for x in self.train_ds:
            self.assertEqual(x["input_ids"].shape[-1], self.run_params.seq_length)

    def test_valid_ds_input_ids_length(self):
        """Each sequence in the dataset should have the same length
        """
        for x in self.valid_ds:
            self.assertEqual(x["input_ids"].shape[-1], self.run_params.seq_length)


if __name__ == "__main__":
    unittest.main()

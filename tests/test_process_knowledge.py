import unittest

from dataset_creation.build_knowledge_associations import process_knowledge


class TestEntitiesClient(unittest.TestCase):
    def setUp(self):
        self.data_path = "data/augmented_datasets/pickle/sorted_attentions_sealed.pkl"
        self.save_path = None
        self.debug = True
        pass

    def tearDown(self):
        pass

    def test_process_knowledge(self):
        df = process_knowledge(self.data_path, self.save_path, self.debug)
        self.assertIsNotNone(df)


if __name__ == "__main__":
    unittest.main()

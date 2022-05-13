import unittest

import pandas as pd
from kirby.data_manager import tokenize_knowledge


class TestKnowledgeTokenize(unittest.TestCase):
    def setUp(self):
        data_files = {
            "train": ["data/augmented_datasets/pickle/max_attention_sealed.pkl"],
            "valid": ["data/augmented_datasets/pickle/max_attention_valid_sealed.pkl"],
        }
        df = pd.read_pickle(data_files["valid"])
        self.x = df.iloc[0]

    def tearDown(self):
        pass

    def test_knowledge_tokenize(self):
        result = tokenize_knowledge(self.x)
        self.assertIsNotNone(result)
        pass


if __name__ == "__main__":
    unittest.main()

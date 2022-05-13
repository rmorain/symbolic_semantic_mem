import unittest

import pandas as pd
import torch
from dataset_creation.attention_entity_selection import (get_attention,
                                                         process_attentions,
                                                         process_data,
                                                         sort_attentions)
from transformers import GPT2Tokenizer


class TestAttentionEntitySelection(unittest.TestCase):
    def setUp(self):
        # self.text = "The animal didn't cross the road because it was too tired"
        self.text = "Apple is looking at buying U.K. startup for $1 billion."
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokens = self.tokenizer(self.text)
        self.tokens = torch.LongTensor(self.tokens["input_ids"])
        self.entities = ["Apple", "U.K.", "$1 billion"]
        # self.entities = ["animal", "cross the road"]

    def test_model_returns_attention(self):
        attention = get_attention(self.tokens)
        self.assertIsNotNone(attention)

    def test_process_attentions(self):
        attentions = get_attention(self.tokens)
        attentions = process_attentions(attentions)
        expected_shape = torch.Size([12])
        self.assertEqual(attentions.shape, expected_shape)

    def test_sort_attentions(self):
        attentions = get_attention(self.tokens)
        attentions = process_attentions(attentions)
        sorted_attentions = sort_attentions(
            attentions, self.tokens, self.entities, self.tokenizer
        )
        self.assertIsNotNone(sorted_attentions)

    def test_process_data(self):
        data = "data/augmented_datasets/pickle/augmented_train_sealed.pkl"
        df = pd.read_pickle(data)
        processed_data = process_data(df, None, self.tokenizer, debug=True)
        self.assertIsNotNone(processed_data)


if __name__ == "__main__":
    unittest.main()

import unittest

import torch
from kirby.knowledge_gpt2_model import KnowledgeAttention
from transformers import GPT2Config


class TestKnowledgeGPT2Model(unittest.TestCase):
    def setUp(self):
        config = GPT2Config()
        self.knowledge_attention = KnowledgeAttention(config)

    def tearDown(self):
        pass

    def test_knowledge_attention(self):
        query = torch.rand(8, 12, 192, 64)
        key = torch.rand(8, 12, 192, 64)
        value = torch.rand(8, 12, 192, 64)
        leti = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        result = self.knowledge_attention._attn(query, key, value, leti=leti)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()

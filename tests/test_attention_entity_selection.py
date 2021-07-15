import unittest

from dataset_creation.attention_entity_selection import get_attention


class TestAttentionEntitySelection(unittest.TestCase):
    def setUp(self):
        self.text = "The animal didn't cross the road because it was too tired"

    def test_model_returns_attention(self):
        attention = get_attention(self.text)
        self.assertIsNotNone(attention)

    def test_filter_by_entitites(self):
        pass


if __name__ == "__main__":
    unittest.main()

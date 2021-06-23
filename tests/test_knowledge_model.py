import unittest

from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams

from doubles import KnowledgeInputDouble


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

    def test_knowledge_model_forward_with_knowledge_input(self):
        output = self.model(self.test_input)
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()

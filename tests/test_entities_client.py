import unittest

from kirby.entities_client import EntitiesClient


class TestEntitiesClient(unittest.TestCase):
    def setUp(self):
        self.entities_client = EntitiesClient()

    def tearDown(self):
        pass

    def test_find_by_label(self):
        label = "Bill Gates"
        description = "American business magnate and philanthropist"
        knowledge = self.entities_client.find_by_label(label)
        self.assertEqual(description, knowledge["description"])

    def test_filter_criteria(self):
        entity = {"description": "Wikimedia disambiguation page"}
        result = self.entities_client.filter_criteria(entity)
        self.assertFalse(result)
        entity = {"description": "playing card"}
        result = self.entities_client.filter_criteria(entity)
        self.assertFalse(result)

    def test_not_found(self):
        label = "Robert Morain"
        result = self.entities_client.find_by_label(label)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

import time
import unittest

from kirby.database_proxy import WikiDatabase
from kirby.run_params import RunParams


class TestWikiDatabase(unittest.TestCase):
    def setUp(self):
        self.run_params = RunParams()
        self.db = WikiDatabase(self.run_params)
        self.test_entity = [
            "Q5284",
            "Bill Gates",
            "American business magnate and philanthropist",
        ]
        self.test_associations = [
            ["P25", "Q454928"],
            ["P19", "Q5083"],
            ["P26", "Q463877"],
            ["P106", "Q131524"],
            ["P27", "Q30"],
            ["P108", "Q655286"],
            ["P69", "Q7879362"],
            ["P910", "Q7112419"],
            ["P31", "Q5"],
            ["P551", "Q1506847"],
            ["P166", "Q12201445"],
            ["P1343", "Q17311605"],
            ["P735", "Q12344159"],
            ["P39", "Q484876"],
            ["P734", "Q16870134"],
            ["P40", "Q23011254"],
            ["P172", "Q7435494"],
            ["P463", "Q463303"],
            ["P3438", "Q575937"],
            ["P1412", "Q1860"],
            ["P21", "Q6581097"],
            ["P1830", "Q2283"],
            ["P1441", "Q12125162"],
            ["P6886", "Q1860"],
            ["P641", "Q188966"],
            ["P552", "Q789447"],
            ["P22", "Q684014"],
            ["P8017", "L19333-F1"],
            ["P6553", "L485"],
            ["P3373", "Q92466067"],
            ["P1344", "Q16972891"],
        ]
        self.test_formatted_associations = {
            "id": "Q5284",
            "label": "Bill Gates",
            "description": "American business magnate and philanthropist",
            "mother": "Mary Maxwell Gates",
            "place of birth": "Seattle",
            "spouse": "Melinda Gates",
            "occupation": "entrepreneur",
            "country of citizenship": "United States of America",
            "employer": "Bill & Melinda Gates Foundation",
            "educated at": "Lakeside School",
            "topic's main category": "Category:Bill Gates",
            "instance of": "human",
            "residence": "Medina",
            "award received": "Knight Commander of the Order of the British Empire",
            "described by source": "Lentapedia (full versions)",
            "given name": "William",
            "position held": "chief executive officer",
            "family name": "Gates",
            "child": "Jennifer Katherine Gates",
            "ethnic group": "Scotch-Irish Americans",
            "member of": "American Academy of Arts and Sciences",
            "vehicle normally used": "Porsche 959",
            "languages spoken, written or signed": "English",
            "sex or gender": "male",
            "owner of": "Microsoft",
            "present in work": "iSteve",
            "writing language": "English",
            "sport": "contract bridge",
            "handedness": "left-handedness",
            "father": "William H. Gates Jr.",
            "sibling": "Kristianne Gates",
            "participant in": "commencement at the Harvard University",
        }

    def tearDown(self):
        pass

    def test_get_entity_by_label(self):
        # Successful
        entity_string = "Bill Gates"
        entity = self.db.get_entity_by_label(entity_string)
        self.assertEqual(entity, self.test_entity)

        # More cases
        entity_string = "Valkyria Chronicles III"
        entity = self.db.get_entity_by_label(entity_string)
        self.assertIsNotNone(entity)

        # More cases
        entity_string = "Valkyria Chronicles"
        entity = self.db.get_entity_by_label(entity_string)
        self.assertIsNotNone(entity)

        # Exception
        entity_string = None
        entity = self.db.get_entity_by_label(entity_string)
        self.assertIsNone(entity)

    def test_get_entity_associations(self):
        # Successful
        entity_id = "Q5284"

        associations = self.db.get_entity_associations(entity_id)
        self.assertEqual(associations, self.test_associations)

        # Exception
        entity_id = None
        associations = self.db.get_entity_associations(entity_id)
        self.assertIsNone(associations)

    def test_no_none_associations(self):
        entity_id = "Q5287"
        associations = self.db.get_entity_associations(entity_id)
        for a in associations:
            self.assertIsNotNone(a)

    def test_no_none_related_entity_labels(self):
        entity_label = "Japanese"
        k = self.db.get_knowledge(entity_label)
        self.assertNotIn(None, k.values())

    def test_format_associations(self):
        # Successful
        formatted_associations = self.db.format_associations(
            self.test_entity, self.test_associations
        )
        self.assertEqual(formatted_associations, self.test_formatted_associations)

        # None values
        formatted_associations = self.db.format_associations(None, None)
        self.assertIsNone(formatted_associations)

    def test_none_associations_format_associations(self):
        formatted_associations = self.db.format_associations(self.test_entity, None)
        self.assertIsNone(formatted_associations)

    def test_get_knowledge(self):
        # Successful
        k = self.db.get_knowledge(self.test_entity[1])
        self.assertEqual(k, self.test_formatted_associations)

        # More cases
        entity_label = "Japanese"
        k = self.db.get_knowledge(entity_label)
        self.assertIsNotNone(k)

        entity_label = "Unrecorded Chronicles"
        k = self.db.get_knowledge(entity_label)
        self.assertIsNone(k)

        # Entity is None
        k = self.db.get_knowledge(None)
        self.assertIsNone(k)

    def test_get_knowledge_none_then_not_none(self):
        entity_label = "Unrecorded Chronicles"
        k = self.db.get_knowledge(entity_label)
        self.assertIsNone(k)

        entity_label = "Japanese"
        k = self.db.get_knowledge(entity_label)
        self.assertIsNotNone(k)

    def test_redis(self):
        self.assertIsNotNone(self.db.redis_connection)
        entity_label = "Japanese"
        self.db.redis_connection.delete(entity_label)
        start = time.time()
        self.db.get_knowledge(entity_label)
        end = time.time()
        first_time = end - start
        start = time.time()
        self.db.get_knowledge(entity_label)
        end = time.time()
        second_time = end - start
        print(f"Redis is {first_time / second_time:.2f}x faster")
        self.assertTrue(second_time < first_time)


if __name__ == "__main__":
    unittest.main()

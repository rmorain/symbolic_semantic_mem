import unittest
from run_params import RunParams
from database_proxy import WikiDatabase

class TestWikiDatabase(unittest.TestCase):
    
    def setUp(self):
        self.run_params = RunParams(db='../db/wikidata.db')
        self.db = WikiDatabase(self.run_params) 

    def tearDown(self):
        pass

    def test_get_knowledge(self):
        pass

    def test_get_entity_by_label(self):
        # Successful
        entity_string = "Bill Gates"
        test_entity = ['Q5284', 'Bill Gates', 
                'American business magnate and philanthropist']
        entity = self.db.get_entity_by_label(entity_string)
        self.assertEqual(entity, test_entity) 

        # Exception
        entity_string = "Robert Morain"
        entity = self.db.get_entity_by_label(entity_string)
        self.assertIsNone(entity)

if __name__ == '__main__':
    unittest.main()

from pymongo import MongoClient


class EntitiesClient:
    def __init__(
        self, host="localhost", port=27017, db="wikidata", collection="entities"
    ):
        self.client = MongoClient(host, port)
        self.db = self.client[db]
        self.collection = self.db[collection]

    @staticmethod
    def filter_criteria(entity):
        stopwords = [
            "Wikimedia disambiguation page",
            "playing card",
        ]
        for w in stopwords:
            if entity["description"] == w:
                return False
        return True

    def find_by_label(self, entity_label):
        entities = self.collection.find({"label": entity_label})
        entities = [e for e in entities]
        # Check if found
        entities.sort(key=lambda x: int(x["entity_id"][1:]))  # Sort by entity id
        # Remove disambiguation page
        entities = list(
            filter(
                self.filter_criteria,
                entities,
            )
        )
        # Return first in list
        if len(entities) == 0:
            return None
        return entities[0]

import pprint

from pymongo import MongoClient

MONGO_URL = "mongodb://localhost:27017"
client = MongoClient(MONGO_URL)
db = client.admin
server_status_result = db.command("serverStatus")
db = client.wikidata
collection = db["knowledge"]

pipeline = [
    {
        "$match": {
            "instance of": "human",
            "occupation": "artist",
        }
    },
    {
        "$group": {
            "_id": "$place of birth",
        }
    },
    {"$limit": 50},
]

result = list(collection.aggregate(pipeline))
pprint.pprint(result)

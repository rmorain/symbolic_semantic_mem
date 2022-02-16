import time

from kirby.database_proxy import WikiDatabase
from kirby.run_params import RunParams
from pymongo import MongoClient

MONGO_URL = "mongodb://localhost:27017"
client = MongoClient(MONGO_URL)
db = client.admin
server_status_result = db.command("serverStatus")

# Create SQL connection
run_params = RunParams()
wiki_database = WikiDatabase(run_params)


# Create db and collection
db = client.wikidata
collection = db["knowledge"]

# SQL query
start = time.time()
knowledge = wiki_database.get_knowledge("Indira Gandhi")
end = time.time()
sql_time = end - start
print(end - start)
print(knowledge)

# MONGO query
start = time.time()
knowledge = collection.find_one({"label": "Indira Gandhi"})
end = time.time()
mongo_time = end - start
print(end - start)
print(knowledge)

print(f"Mongo is {(sql_time / mongo_time) * 100:.2f}% faster than SQLite3")

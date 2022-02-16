from pprint import pprint

import pandas as pd
from kirby.database_proxy import WikiDatabase
from kirby.run_params import RunParams
from pymongo import MongoClient
from tqdm import tqdm

# Create mongo connection
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

# Iterate over db
total = pd.read_sql_query("SELECT COUNT(1) FROM Entities", wiki_database.conn)
total = total["COUNT(1)"][0]
pbar = tqdm(total=total)
for chunk in pd.read_sql_query(
    "SELECT * FROM Entities;", wiki_database.conn, chunksize=1
):
    sql = f"""
SELECT relations.label, Entities.label as related_entity_label
FROM Entities
JOIN (
SELECT p.label, pr.related_entity_id
FROM Properties_relations as pr
LEFT JOIN Entities as e
ON pr.entity_id = e.entity_id
LEFT JOIN Properties as p
ON pr.property_id = p.property_id
WHERE pr.entity_id = "{chunk.entity_id[0]}"
)
AS relations
ON relations.related_entity_id = Entities.entity_id
"""
    data = pd.read_sql_query(sql, wiki_database.conn)
    data_dict = {
        "_id": chunk.entity_id[0],
        "label": chunk.label[0],
        "description": chunk.description[0],
    }
    # Add associations
    for row in data.itertuples(index=False):
        data_dict[row[0]] = row[1]

    # Write to mongo
    collection.insert_one(data_dict)
    pbar.update(1)
pbar.close()

import pandas as pd
from kirby.database_proxy import WikiDatabase
from kirby.run_params import RunParams
from tqdm.auto import tqdm

from db_creation.properties import properties


def insert(x, db):
    sql = f"INSERT INTO Properties (property_id, label) VALUES ({x[0]},{x[1]});"
    pd.read_sql_query(sql, db.conn)


tqdm.pandas()

properties = properties()
run_params = RunParams()
db = WikiDatabase(run_params)

# Convert properties into pandas df
data_items = properties.items()
data_list = list(data_items)
df = pd.DataFrame(data_list)
# df.progress_apply(lambda x: insert(x, db), axis=1)
df.to_sql("Properties", db.conn, if_exists="replace", index=False)

import time

from pyspark.sql import SparkSession

spark = SparkSession.builder.config(
    "spark.jars.packages", "org.xerial:sqlite-jdbc:3.34.0"
).getOrCreate()

db_df = (
    spark.read.format("jdbc")
    .options(
        driver="org.sqlite.JDBC",
        dbtable="entities",
        url="jdbc:sqlite:db/wikidata.db",
    )
    .load()
)

csv_df = spark.read.option("header", True).csv("db/Entities.csv")

start = time.time()
gates = db_df.filter(db_df.label == "Bill Gates")
gates.show()
end = time.time()
print(f"DB time: {end - start}")

start = time.time()
gates = csv_df.filter(csv_df.label == "Bill Gates")
gates.show()
end = time.time()
print(f"CSV time: {end - start}")

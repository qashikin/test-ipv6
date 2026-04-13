import sys
import time
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

print("=== STARTING SELF-CONTAINED SPARK UI TEST ===")

# ============================================
# Generate datasets entirely in memory
# ============================================
print("=== GENERATING DATASETS ===")

df1 = spark.range(0, 1000000).toDF("id")
df1 = (
    df1.withColumn("category", (F.col("id") % 100).cast("string"))
    .withColumn("value", F.rand() * 1000)
    .withColumn("city", F.concat(F.lit("City_"), (F.col("id") % 50).cast("string")))
    .withColumn("timestamp", F.current_timestamp())
)
df1.cache()
print(f"Dataset 1 count: {df1.count()}")

df2 = spark.range(0, 500000).toDF("id")
df2 = (
    df2.withColumn("department", F.concat(F.lit("Dept_"), (F.col("id") % 20).cast("string")))
    .withColumn("salary", F.rand() * 100000)
)
df2.cache()
print(f"Dataset 2 count: {df2.count()}")

# ============================================
# Iterative transformations
# ============================================
for i in range(5):
    print(f"\n=== ITERATION {i+1} ===")

    # Aggregation
    print("Running aggregation...")
    agg_df = df1.groupBy("category", "city").agg(
        F.count("*").alias("count"),
        F.sum("value").alias("total_value"),
        F.avg("value").alias("avg_value"),
        F.max("value").alias("max_value"),
        F.min("value").alias("min_value"),
    )
    print(f"Aggregation result count: {agg_df.count()}")

    # Join (shuffle)
    print("Running join...")
    joined_df = df1.join(df2, "id", "inner")
    print(f"Join result count: {joined_df.count()}")

    # Sort (shuffle)
    print("Running sort...")
    df1.orderBy(F.desc("value")).take(100)

    # Window function
    print("Running window function...")
    window_spec = Window.partitionBy("category").orderBy(F.desc("value"))
    windowed_df = df1.withColumn("rank", F.row_number().over(window_spec))
    windowed_df.filter(F.col("rank") <= 10).count()

    # Distinct (shuffle)
    print("Running distinct...")
    distinct_count = df1.select("category", "city").distinct().count()
    print(f"Distinct combinations: {distinct_count}")

    print("Sleeping 30 seconds...")
    time.sleep(30)

# ============================================
# Final heavy operations
# ============================================
print("\n=== FINAL HEAVY OPERATIONS ===")

# Cross join on small sample
print("Running cross join sample...")
small_df1 = df1.limit(1000)
small_df2 = df2.limit(1000)
cross_df = small_df1.crossJoin(small_df2)
print(f"Cross join count: {cross_df.count()}")

# Pivot
print("Running pivot...")
pivot_df = df1.groupBy("city").pivot("category").agg(F.sum("value"))
print(f"Pivot columns: {len(pivot_df.columns)}")

# Union and dedup
print("Running union and dedup...")
union_df = df1.union(df1).union(df1)
dedup_df = union_df.dropDuplicates()
print(f"Dedup count: {dedup_df.count()}")

# Repartition (full shuffle)
print("Running repartition...")
repartitioned = df1.repartition(200, "category")
print(f"Repartitioned count: {repartitioned.count()}")

# Coalesce
print("Running coalesce...")
coalesced = df1.coalesce(10)
print(f"Coalesced partition count: {coalesced.rdd.getNumPartitions()}")

# ============================================
# Keep job alive for Spark UI viewing
# ============================================
print("\n=== KEEPING JOB ALIVE FOR SPARK UI ===")
for i in range(6):
    print(f"Waiting... {i+1}/6 (30 sec intervals)")
    df1.sample(0.01).count()
    time.sleep(30)

print("\n=== JOB COMPLETE ===")
job.commit()

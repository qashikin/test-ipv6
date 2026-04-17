from pyspark.context import SparkContext
from awsglue.context import GlueContext

sc = SparkContext()
glueContext = GlueContext(sc)

# Read from a non-existent table - will fail
glueContext.create_dynamic_frame.from_catalog(
    database="nonexistent_database",
    table_name="nonexistent_table"
)

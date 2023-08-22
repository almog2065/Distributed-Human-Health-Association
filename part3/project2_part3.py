from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
import os
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

SCHEMA = StructType([StructField("Arrival_Time", LongType(), True),
                     StructField("Creation_Time", LongType(), True),
                     StructField("Device", StringType(), True),
                     StructField("Index", LongType(), True),
                     StructField("Model", StringType(), True),
                     StructField("User", StringType(), True),
                     StructField("gt", StringType(), True),
                     StructField("x", DoubleType(), True),
                     StructField("y", DoubleType(), True),
                     StructField("z", DoubleType(), True)])

spark = SparkSession.builder.appName('demo_app') \
	.config("spark.kryoserializer.buffer.max", "512m") \
	.config("spark.driver.memory", "40g") \
	.getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
	"--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'

topic = "static"
static_df = spark.read \
	.format("kafka") \
	.option("kafka.bootstrap.servers", kafka_server) \
	.option("subscribe", topic) \
	.option("startingOffsets", "earliest") \
	.option("failOnDataLoss", False) \
	.option("maxOffsetsPerTrigger", 432) \
	.load() \
	.select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

topic = "activities"

ord_udf = f.udf(lambda x: ord(x) - ord("a"), LongType())
model_list = ["nexus4"]
device_list = ["nexus4_1", "nexus4_2"]
relevant_columns = ["Arrival_Time", "Creation_Time", "nexus4_1", "nexus4_2", "Index", "nexus4", "User",
                    "x", "y", "z"]
vecAssembler = VectorAssembler(inputCols=relevant_columns, outputCol="features")
gt_list = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'null']
gt_udf = f.udf(lambda x: gt_list.index(x), IntegerType())

all_data = static_df

all_data = all_data.withColumn("User", ord_udf(f.col("User")))
for model in model_list:
	all_data = all_data.withColumn(model, (all_data["Model"] == model).cast("integer"))
for device in device_list:
	all_data = all_data.withColumn(device, (all_data["Device"] == device).cast("integer"))
all_data = all_data.withColumn("gt", gt_udf(f.col("gt")))
all_data = vecAssembler.transform(all_data)

ml_model = RandomForestClassifier(numTrees=5, maxDepth=20, labelCol="gt", seed=42)
ml_model = ml_model.fit(all_data)

streaming = spark.readStream \
	.format("kafka") \
	.option("kafka.bootstrap.servers", kafka_server) \
	.option("subscribe", topic) \
	.option("startingOffsets", "earliest") \
	.option("failOnDataLoss", False) \
	.option("maxOffsetsPerTrigger", 200000) \
	.load() \
	.select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")).select("value.*")

acc_list = []


def predict_data(new_data, id):
	global acc_list

	new_count = new_data.count()

	# Ending the code if less than 200,000 records were received.
	if new_count < 200000:
		print("Total accuracy is {}%".format(round(100 * sum(acc_list) / len(acc_list), 2)))
		return

	print("Streaming {} new records".format(new_count))

	new_data = new_data.withColumn("User", ord_udf(f.col("User")))
	for model in model_list:
		new_data = new_data.withColumn(model, (new_data["Model"] == model).cast("integer"))
	for device in device_list:
		new_data = new_data.withColumn(device, (new_data["Device"] == device).cast("integer"))
	new_data = new_data.withColumn("gt", gt_udf(f.col("gt")))
	new_data = vecAssembler.transform(new_data)

	pred = ml_model.transform(new_data)
	pred = pred.select("gt", "prediction")
	accuracy = pred.withColumn("correct", (pred["prediction"] == pred["gt"]).cast("integer"))
	score = accuracy.agg(f.mean("correct")).toDF(*['avg']).collect()[0].avg
	acc_list.append(score)
	if score is not None:
		print('Accuracy is {}% for batch size of 200,000'.format(round(100 * score, 2)))
	else:
		print(None)


streaming \
	.writeStream.foreachBatch(predict_data) \
	.start() \
	.awaitTermination()

print("The End")

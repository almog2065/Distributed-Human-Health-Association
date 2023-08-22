import findspark
findspark.init()
from pyspark.sql import functions as f, SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import *


def get_train_test():
	spark = SparkSession.builder.config("spark.driver.memory", "3g").appName('proj_b').getOrCreate()
	sc = spark.sparkContext
	df_ml = spark.read.json(r"data.json")
	return df_ml.randomSplit([0.7, 0.3], 24)


def prep_data(data):
	ord_udf = f.udf(lambda x: ord(x) - ord("a"), LongType())
	data = data.withColumn("User", ord_udf(f.col("User")))
	model_list = data.select("Model").distinct().collect()
	for row in model_list:
		data = data.withColumn(row.Model, (data["Model"] == row.Model).cast("integer"))
	device_list = data.select("Device").distinct().collect()
	for row in device_list:
		data = data.withColumn(row.Device, (data["Device"] == row.Device).cast("integer"))
	relevant_columns = list(set(data.columns) - {'Model', 'Device', 'gt'})
	vecAssembler = VectorAssembler(inputCols=relevant_columns, outputCol="features")
	data = vecAssembler.transform(data)
	gt_list = ['bike', 'sit', 'stand', 'walk', 'stairsup', 'stairsdown', 'null']
	gt_udf = f.udf(lambda x: gt_list.index(x), IntegerType())
	return data.withColumn("gt", gt_udf(f.col("gt")))


def check_accuracy(train, test, numtrees, maxdepth):
	rf = RandomForestClassifier(numTrees=numtrees, maxDepth=maxdepth, labelCol="gt", seed=42, maxBins=300)
	model = rf.fit(train)
	pred = model.transform(test)
	pred = pred.select("gt", "prediction")
	accuracy = pred.withColumn("correct", (pred["prediction"] == pred["gt"]).cast("integer"))
	score = accuracy.agg(f.mean("correct")).toDF(*['avg']).collect()[0].avg
	print(f"{numtrees} trees with depth of {maxdepth} : {score}")


def experiment(train, test):
	depths = [15, 20, 25, 30]
	nums = [5, 10, 15, 20]
	for num in nums:
		for depth in depths:
			check_accuracy(train, test, num, depth)


def main():
	train, test = get_train_test()
	train = prep_data(train)
	test = prep_data(test)
	# experiment(train, test)
	check_accuracy(train, test, numtrees=5, maxdepth=20)


if __name__ == "__main__":
	main()

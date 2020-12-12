# Databricks notebook source
import pyspark.sql.functions as f
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import Bucketizer
from pyspark.sql.types import StructField, StructType, StringType, LongType,TimestampType,FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer



# COMMAND ----------

heartSchema = StructType( \
  [StructField('id', LongType(), True), \
   StructField('age', LongType(), True), \
   StructField('sex', StringType(), True), \
   StructField('chol', LongType(), True), \
   StructField('pred', StringType(), True), \
])

# COMMAND ----------

trainingPath = '/FileStore/tables/heartTraining.csv'
testingPath = '/FileStore/tables/heartTesting.csv'
training = spark.read.format('csv').schema(heartSchema).option("header", True).load(trainingPath)
testing = spark.read.format('csv').schema(heartSchema).option("header", True).load(testingPath)
training = training.na.drop()
testing.take(10)

# COMMAND ----------

# define needed functions for the pipeline and create the pipeline
# 1) Bucket ages
splits = [0,40,50,60,70,float("inf")] #up to 120 years old to include any possible outliers
bucketizer = Bucketizer(splits=splits, inputCol="age", outputCol="age_buckets")
# data = bucketizer.transform(training)
# data.take(10)

# 2) convert sex,label andpred to numeric
indexer_sex = StringIndexer(inputCol="sex", outputCol="sexIndex")
indexer_label = StringIndexer(inputCol="pred", outputCol="label")
# trans = indexer_sex.fit(training).transform(training)
# trans.take(10)

# 3) Assemble wanted features for the model
assembler = VectorAssembler(inputCols=['age_buckets','sexIndex','chol'],outputCol="features")

# 4) define the logistic regression: (What Parameters should I add to this model????)
lr = LogisticRegression(maxIter=10, regParam=0.01)

#  5) Make the pipeline
myStages = [bucketizer,indexer_sex,indexer_label,assembler, lr]
pipe = Pipeline(stages=myStages)

# COMMAND ----------

# Why don't we need to drop unused columns???

# COMMAND ----------

pipeModel = pipe.fit(training)

# COMMAND ----------

predictions = pipeModel.transform(testing)
predMod = predictions.select('id','features','probability','prediction')
predMod.show()

# COMMAND ----------



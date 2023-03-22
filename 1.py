import pyspark
import pandas
from pyspark.ml.feature import Tokenizer
#import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
from pyspark.sql.types import BinaryType
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sql("set spark.sql.files.ignoreCorruptFiles=true")
df = spark.read.format("binaryFile")\
               .option("pathGlobFilter", "*.jpg")\
               .option("recursiveFileLookup", "true")\
               .load("/user/marc/256_sampledata")
      
      
df2 = spark.read.option("mode","FAILFAST")\
               .option("delimiter","\t")\
               .csv("/user/marc/zoo.csv")
    
    
df2.limit(10).toPandas()
df2.show(10,truncate=False)


sentence_data_frame = spark.createDataFrame([
    (0, "Hi I think pyspark is cool ","happy"),
    (1, "All I want is a pyspark cluster","indifferent"),
    (2, "I finally understand how ML works","fulfilled"),
    (3, "Yet another sentence about pyspark and ML","indifferent"),
    (4, "Why didn’t I know about mllib before","sad"),
    (5, "Yes, I can","happy")
], ["id", "sentence", "sentiment"])



tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
tokenized = tokenizer.transform(sentence_data_frame)



tokenized.show(10,truncate=False)



from pyspark.ml.feature import StopWordsRemover

remover = StopWordsRemover(inputCol="words", outputCol="meaningful_words")
meaningful_data_frame = remover.transform(tokenized)
# I use the show function here for educational purposes only; with a large 
# dataset, you should avoid it.
meaningful_data_frame.select("words","meaningful_words").show(5,truncate=False)
 


 
from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol="sentiment", outputCol="categoryIndex")
indexed = indexer.fit(meaningful_data_frame).transform(meaningful_data_frame)
indexed.show(5)


from pyspark.sql.functions import col, regexp_extract

def extract_label(path_col): 
    return regexp_extract(path_col,"256_sampledata/([^/]+)",1)
  

  
df_result = spark.read.format("binaryFile")\
               .option("pathGlobFilter", "*.jpg")\
               .option("recursiveFileLookup", "true")\
               .load("/user/marc/256_sampledata")

images_with_label = df_result.select( 
    col("path"),
    extract_label(col("path")).alias("label"),
    col("content"))


df2 = spark.read.option("mode","FAILFAST")\
               .option("delimiter","\t")\
                .option("header", "true")\
               .csv("/user/marc/zoo.csv")
  
df2.limit(10).toPandas()

print(df2.columns)

inputCols = [
 'fins',
 'domestic'
]

from pyspark.ml.feature import VectorAssembler

#https://towardsdatascience.com/feature-encoding-made-simple-with-spark-2-3-0-part-2-5bfc869a809a


dfwine = spark.read.option("mode","FAILFAST")\
               .option("delimiter","\t")\
                .option("header", "true")\
               .csv("/user/marc/wineQualityReds.csv")
      
dfwinerenamed = dfwine.withColumnRenamed("_c0", "ID")\
.withColumnRenamed("fixed.acidity", "fixed_acidity")\
.withColumnRenamed("volatile.acidity", "volatile_acidity")\
.withColumnRenamed("citric.acid", "citric_acid")\
.withColumnRenamed("residual.sugar", "residual_sugar")\
.withColumnRenamed("free.sulfur.dioxide", "free_sulfur_dioxide")\
.withColumnRenamed("total.sulfur.dioxide", "total_sulfur_dioxide")

  
dfwinerenamed.limit(10).toPandas()

print(dfwinerenamed.columns)

model_name = "spark-model"
features = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data_df = spark.createDataFrame([\
            (5.1, 3.5, 1.4, 0.2, "setosa"),\
            (7.0, 3.2, 4.7, 1.4, "versicolor")\
           ],\
        features
    )
print("Data:")
data_df.show()

#https://github.com/amesar/mlflow-tools/blob/master/tests/spark/test_sparkml_udf_workaround.py

feature_cols = data_df.columns[:-1]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

vector_df = assembler.transform(data_df)

from pyspark.ml.stat import Summarizer
summarizer = Summarizer.metrics("mean","sum","variance","std")

statistics_df = vector_df.select(summarizer.summary(vector_df.features))
# statistics_df will plot all the metrics
statistics_df.show(truncate=False)

# compute statistics for single metric (here, std) without the rest
vector_df.select(Summarizer.std(vector_df.features)).show(truncate=False)



 












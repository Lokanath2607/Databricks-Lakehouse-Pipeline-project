🚀 End-to-End Databricks Lakehouse & ML Pipeline
An enterprise-grade Big Data ETL and Machine Learning pipeline built on Databricks utilizing the Medallion Architecture (Bronze, Silver, Gold), PySpark, Delta Lake, and MLflow.

🏗️ Architecture Blueprint
Plaintext
[Raw Data Ingestion] 
       │
       ▼
 🥉 **Bronze Layer** (Raw Data Ingestion & Delta Lake Managed Tables)
       │
       ▼
 🥈 **Silver Layer** (Data Cleaning, Transformation & Null Filtering)
       │
       ▼
 🥇 **Gold Layer** (Business Aggregations & Analytical KPIs)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
 📊 **Databricks Dashboard**     🤖 **MLOps via MLflow**
    (Visual Delay Analytics)     (Logistic Regression Model & Tracking)
🛠️ Tech Stack & Core Tools
Platform: Databricks Lakehouse Platform

Engine & Languages: Apache Spark, PySpark, Spark SQL

Storage & Data Format: Delta Lake (Managed Metastore Tables)

MLOps & Experiment Tracking: Scikit-Learn, MLflow

Data Visualization: Databricks SQL & Lakehouse Dashboards

📊 Pipeline Implementation Stages
1. Bronze Layer (Raw Ingestion)
Reads raw flight operational data into a PySpark DataFrame and persists it as a managed raw Delta Lake table.

Python
file_path = "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv"

raw_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(file_path)

# Save as a managed Delta table
raw_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("bronze_flights")

    
2. Silver Layer (Data Cleaning & Transformation)
Cleans raw records by removing missing values, filtering invalid delay entries, and storing the clean dataset in the Silver Delta table.

Python
bronze_df = spark.read.table("bronze_flights")

# Clean data (drop nulls and filter valid records)
silver_df = bronze_df.dropna().filter(bronze_df["delay"] > -100)

# Save as a managed silver table
silver_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("silver_flights")


    
3. Gold Layer (Business Aggregations & KPIs)
Calculates core operational performance metrics (average departure delay per origin airport) and saves it into an analytical Gold Delta table.

Python
from pyspark.sql.functions import col, avg, desc

silver_df = spark.read.table("silver_flights")

# Calculate average delay by origin airport
gold_df = silver_df.groupBy("origin") \
    .agg(avg("delay").alias("avg_delay")) \
    .orderBy(desc("avg_delay"))

# Save as a managed gold table
gold_df.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("gold_flights_summary")

    
4. MLOps & Experiment Tracking (MLflow)
Trains a binary classification model (Scikit-Learn Logistic Regression) to predict flight delay risks and logs hyperparameters, metrics, and model artifacts directly into MLflow.

Python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert Silver Spark DataFrame to Pandas sample
pdf = silver_df.select("distance", "delay") \
    .withColumn("is_delayed", (col("delay") > 15).cast("int")) \
    .limit(50000).toPandas()

X = pdf[["distance"]]
y = pdf["is_delayed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log parameters, metrics, and model artifact with MLflow
with mlflow.start_run(run_name="flight_delay_prediction_model"):
    model = LogisticRegression(C=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    mlflow.log_param("regularization_C", 1.0)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "flight_delay_model")

    
📈 Insights & Dashboard Delivery
Managed multi-layer Data Lakehouse pipeline using Delta Lake tables.

Spark SQL queries used to aggregate delay metrics across origin airports.

Built and published an interactive Databricks Lakehouse Dashboard highlighting top delay origins.

Full end-to-end experiment tracking and model registration using MLflow.

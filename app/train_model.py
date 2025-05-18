#!/usr/bin/env python
# coding: utf-8

# **Urdu News Classification Pipeline**
# This script implements a complete machine learning pipeline for classifying Urdu news articles. The main steps include:
# 
# *   Loading and Preprocessing: Read the cleaned Urdu news data with title, content, and label columns, then combine title and content.
# *   Train-Test Split: Splitting data into train, validation, and test data.
# *   TF-IDF Feature Extraction: Vectorize the combined text using TF-IDF with up to 5000 features and (1,2)-gram ranges.
# *   Model Training: Train a multiclass Logistic Regression classifier (max_iter=3000, class_weight='balanced', solver='saga') on the TF-IDF features.
# *   Evaluation: Compute accuracy, F1-score, confusion matrix, and classification report on a held-out test set.
# *   Sentiment Analysis: Use the pretrained Hugging Face model mahwizzzz/UrduClassification to assign a sentiment label to each article.
# *   Experiment Tracking (MLflow): Log parameters, metrics, and the trained model using MLflow for reproducibility.
# *   Saving Predictions: Store the predictions and metadata (title, content, predicted label, predicted sentiment) in a PostgreSQL table called predictions.

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# For train-test split and model training/evaluation
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
import os

# Transformers for pretrained Urdu sentiment analysis
from transformers import pipeline
import torch

# For saving to PostgreSQL
import psycopg2

# Stanza for Named Entity Recognition (NER) and preprocessing
import stanza

# Initialize Stanza pipeline for Urdu
nlp = stanza.Pipeline(lang='ur', processors='tokenize,mwt,pos,lemma')

# ### Load the cleaned Urdu news dataset
db_params = {
    "dbname": os.environ.get("POSTGRES_DB", "news_db"),
    "user": os.environ.get("POSTGRES_USER", "affan"),
    "password": os.environ.get("POSTGRES_PASSWORD", "pass123"),
    "host": os.environ.get("POSTGRES_HOST", "postgres"),
    "port": os.environ.get("POSTGRES_PORT", 5432),
}
conn = psycopg2.connect(**db_params)
data = pd.read_sql("SELECT title, content, label FROM cleaned_articles", conn)
conn.close()

if data.empty:
    print("[WARNING] No data found in cleaned_articles table.")
    exit(0)

print(data.head())

# # 2. Train-Test Split
title = data['title']
content = data['content']
labels = data['label']

# First split: Train vs Temp
title_train, title_temp, content_train, content_temp, y_train, y_temp = train_test_split(
    title, content, labels, test_size=0.2, random_state=42
)

# Second split: Validation + Test
title_val, title_test, content_val, content_test, y_val, y_test = train_test_split(
    title_temp, content_temp, y_temp, test_size=0.5, random_state=42
)

# Combine title and content for TF-IDF training
X_train_text = (title_train + " " + content_train).values
X_test_text = (title_test + " " + content_test).values

print("Training:", title_train.shape, content_train.shape, y_train.shape)
print("Validation:", title_val.shape, content_val.shape, y_val.shape)
print("Test:", title_test.shape, content_test.shape, y_test.shape)

# # 3. TF-IDF Vectorization
# Fill missing values before combining
title_train = title_train.fillna('')
content_train = content_train.fillna('')
title_val = title_val.fillna('')
content_val = content_val.fillna('')
title_test = title_test.fillna('')
content_test = content_test.fillna('')

# Combine title and content for all sets using your variable names
X_train_text = (title_train + " " + content_train).values
X_val_text = (title_val + " " + content_val).values
X_test_text = (title_test + " " + content_test).values

# TF-IDF Vectorization 
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")

# # 4. Model Training
# Initialize the Logistic Regression classifier
clf = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    solver='saga',
    multi_class='multinomial',
    random_state=42
)

# Train on training set
clf.fit(X_train_tfidf, y_train)
print("Training completed.")

# Evaluate on validation set
y_val_pred = clf.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")

# # 5. Model Evaluation
y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (Weighted): {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# # 6. Sentiment Analysis (Hugging Face)
# We apply a pretrained Hugging Face model mahwizzzz/UrduClassification to predict the sentiment of each news article.
sentiment_pipeline = pipeline("sentiment-analysis", model="mahwizzzz/UrduClassification", tokenizer="mahwizzzz/UrduClassification")
data['text'] = data['title'].fillna('') + " " + data['content'].fillna('')
data['predicted_sentiment'] = data['text'].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])

# Show value counts of sentiment labels
print(data['predicted_sentiment'].value_counts())

# Save sentiment stats
import json
sentiment_dist = data['predicted_sentiment'].value_counts().to_dict()
with open("sentiment_distribution.json", "w", encoding="utf-8") as f:
    json.dump(sentiment_dist, f, ensure_ascii=False)

# # 7. Experiment Tracking with MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

# Inspect existing experiments
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    print(f"Name: {exp.name}, ID: {exp.experiment_id}, Artifact Location: {exp.artifact_location}")

# Log to MLflow
mlflow.set_experiment("Urdu_News_Classification")

with mlflow.start_run(run_name="LogReg_TFIDF_Multiclass"):
    mlflow.log_param("solver", "saga")
    mlflow.log_param("max_iter", 3000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("ngram_range", (1, 2))
    mlflow.log_param("max_features", 5000)

    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("weighted_f1", f1)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("sentiment_distribution.json")

    mlflow.sklearn.log_model(clf, "model")
    print("MLflow tracking complete.")

# # 8. Final Predictions and Saving to PostgreSQL
data['combined_text'] = data['title'] + " " + data['content']
data_tfidf = vectorizer.transform(data['combined_text'])
data['predicted_label'] = clf.predict(data_tfidf)

# Prepare final DataFrame
results_df = data[['title', 'content', 'predicted_label', 'predicted_sentiment']].copy()

# PostgreSQL connection
db_params = {
    "dbname": os.environ.get("POSTGRES_DB", "news_db"),
    "user": os.environ.get("POSTGRES_USER", "affan"),
    "password": os.environ.get("POSTGRES_PASSWORD", "pass123"),
    "host": os.environ.get("POSTGRES_HOST", "postgres"),
    "port": os.environ.get("POSTGRES_PORT", 5432)
}

# Connect and insert predictions into PostgreSQL
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()

create_table_query = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    predicted_label TEXT,
    predicted_sentiment TEXT
);
"""
cursor.execute(create_table_query)
conn.commit()

insert_query = """
INSERT INTO predictions (title, content, predicted_label, predicted_sentiment)
VALUES (%s, %s, %s, %s);
"""
for _, row in results_df.iterrows():
    cursor.execute(insert_query, (
        row['title'],
        row['content'],
        row['predicted_label'],
        row['predicted_sentiment']
    ))
conn.commit()

cursor.close()
conn.close()
print("Predictions saved to PostgreSQL.")


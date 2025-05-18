import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import mlflow
import mlflow.sklearn
from transformers import pipeline
import torch
import stanza
import psycopg2

# Load Stanza
nlp = stanza.Pipeline(lang='ur', processors='tokenize,mwt,pos,lemma')

# PostgreSQL connection parameters via Railway env vars
db_params = {
    "dbname": os.environ.get("POSTGRES_DB", "news_db"),
    "user": os.environ.get("POSTGRES_USER", "affan"),
    "password": os.environ.get("POSTGRES_PASSWORD", "pass123"),
    "host": os.environ.get("POSTGRES_HOST", "localhost"),
    "port": os.environ.get("POSTGRES_PORT", 5432),
}

conn = psycopg2.connect(**db_params)
data = pd.read_sql("SELECT title, content, label FROM cleaned_articles", conn)
conn.close()

if data.empty:
    print("[WARNING] No data found in cleaned_articles table.")
    exit(0)

# Split data
title = data['title'].fillna('')
content = data['content'].fillna('')
labels = data['label']

title_train, title_temp, content_train, content_temp, y_train, y_temp = train_test_split(
    title, content, labels, test_size=0.2, random_state=42
)
title_val, title_test, content_val, content_test, y_val, y_test = train_test_split(
    title_temp, content_temp, y_temp, test_size=0.5, random_state=42
)

X_train_text = (title_train + " " + content_train).values
X_val_text = (title_val + " " + content_val).values
X_test_text = (title_test + " " + content_test).values

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# Train model
clf = LogisticRegression(
    max_iter=3000,
    class_weight='balanced',
    solver='saga',
    multi_class='multinomial',
    random_state=42
)
clf.fit(X_train_tfidf, y_train)

y_val_pred = clf.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

y_pred = clf.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Sentiment using Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis", model="mahwizzzz/UrduClassification", tokenizer="mahwizzzz/UrduClassification")
data['text'] = data['title'].fillna('') + " " + data['content'].fillna('')
data['predicted_sentiment'] = data['text'].apply(lambda x: sentiment_pipeline(x[:512])[0]['label'])

# Save sentiment stats
sentiment_dist = data['predicted_sentiment'].value_counts().to_dict()
with open("sentiment_distribution.json", "w", encoding="utf-8") as f:
    json.dump(sentiment_dist, f, ensure_ascii=False)

# MLflow local logging for Railway compatibility
mlflow.set_tracking_uri("file:/tmp/mlruns")
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

# Final predictions
data['combined_text'] = data['title'] + " " + data['content']
data_tfidf = vectorizer.transform(data['combined_text'])
data['predicted_label'] = clf.predict(data_tfidf)

results_df = data[['title', 'content', 'predicted_label', 'predicted_sentiment']].copy()

# Save predictions to PostgreSQL
conn = psycopg2.connect(**db_params)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    predicted_label TEXT,
    predicted_sentiment TEXT
);
""")
conn.commit()

for _, row in results_df.iterrows():
    cursor.execute("""
        INSERT INTO predictions (title, content, predicted_label, predicted_sentiment)
        VALUES (%s, %s, %s, %s);
    """, (row['title'], row['content'], row['predicted_label'], row['predicted_sentiment']))
conn.commit()
cursor.close()
conn.close()
print("✅ Predictions saved to PostgreSQL.")


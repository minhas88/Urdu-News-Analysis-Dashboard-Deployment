from fastapi import FastAPI
import pandas as pd
import psycopg2
import os

app = FastAPI()

# Load environment variables
DB_NAME = os.getenv("POSTGRES_DB", "news_db")
DB_USER = os.getenv("POSTGRES_USER", "affan")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pass123")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

@app.get("/cleaned-news")
def get_cleaned_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM cleaned_articles ORDER BY id DESC LIMIT 100;", conn)
    conn.close()
    return df.to_dict(orient="records")

@app.get("/inference")
def get_predictions():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM predictions ORDER BY RANDOM() LIMIT 100;", conn)
    conn.close()
    return df.to_dict(orient="records")


CREATE DATABASE news_db;
CREATE DATABASE airflow_db;

GRANT ALL PRIVILEGES ON DATABASE news_db TO affan;
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO affan;

\connect news_db
CREATE TABLE IF NOT EXISTS labeled_articles (
    title TEXT,
    content TEXT,
    gold_label TEXT,
    source TEXT,
    timestamp TIMESTAMP
);
GRANT ALL PRIVILEGES ON labeled_articles TO affan;

-- Table for cleaned and preprocessed articles
CREATE TABLE IF NOT EXISTS cleaned_articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    label INT,
    source TEXT,
    timestamp TIMESTAMP
);

-- Table for model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    predicted_label TEXT,
    predicted_sentiment TEXT
);



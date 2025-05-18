-- Create tables directly:
CREATE TABLE IF NOT EXISTS labeled_articles (
    title TEXT,
    content TEXT,
    gold_label TEXT,
    source TEXT,
    timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cleaned_articles (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    label INT,
    source TEXT,
    timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    title TEXT,
    content TEXT,
    predicted_label TEXT,
    predicted_sentiment TEXT
);

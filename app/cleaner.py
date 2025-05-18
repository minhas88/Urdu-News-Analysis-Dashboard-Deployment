import pandas as pd
import psycopg2
import stanza
import os

nlp = stanza.Pipeline(lang='ur', processors='tokenize,mwt,pos,lemma')

category_mapping = {
    'entertainment': 'entertainment',
    'business': 'business',
    'sports': 'sports',
    'science': 'science-technology',
    'technology': 'science-technology',
    'science-technology': 'science-technology',
    'world': 'international',
    'international': 'international'
}

stopwords_file = "app/stopwords-ur.txt"
with open(stopwords_file, 'r', encoding='utf-8') as f:
    stop_words = set(f.read().splitlines())

def remove_stopwords(text):
    if not isinstance(text, str):
        return text
    tokens = text.split()
    filtered = [tok for tok in tokens if tok not in stop_words]
    return " ".join(filtered)

def urdu_preprocess(text):
    if not isinstance(text, str):
        return text
    doc = nlp(text)
    lemmatized_text = ' '.join([word.lemma for sent in doc.sentences for word in sent.words])
    return remove_stopwords(lemmatized_text)

class DataCleaner:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "news_db"),
            user=os.getenv("POSTGRES_USER", "affan"),
            password=os.getenv("POSTGRES_PASSWORD", "pass123"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        self.cursor = self.conn.cursor()

    def clean_data(self):
        print("[LOADING RAW DATA]")
        df = pd.read_sql("SELECT * FROM labeled_articles;", self.conn)
        print(f"[DATA LOADED] Rows: {len(df)}")

        print("[CLEANING DATA]")
        df['gold_label'] = df['gold_label'].str.lower().map(category_mapping)
        df.dropna(subset=['content', 'gold_label'], inplace=True)

        chunk_size = 100
        cleaned_chunks = []
        for i in range(0, len(df), chunk_size):
            print(f"[PROCESSING CHUNK {i//chunk_size+1}]")
            chunk = df.iloc[i:i+chunk_size].copy()
            chunk['title'] = chunk['title'].apply(urdu_preprocess)
            chunk['content'] = chunk['content'].apply(urdu_preprocess)
            cleaned_chunks.append(chunk)

        df = pd.concat(cleaned_chunks, ignore_index=True)
        df['label'] = pd.Categorical(
            df['gold_label'],
            categories=['entertainment', 'business', 'sports', 'science-technology', 'international'],
            ordered=True
        ).codes + 1
        df.drop(columns=['gold_label'], inplace=True)

        print("[INSERTING CLEANED DATA]")
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cleaned_articles (
                title TEXT,
                content TEXT,
                label INT,
                source TEXT,
                timestamp TIMESTAMP
            );
        """)
        for _, row in df.iterrows():
            self.cursor.execute("""
                INSERT INTO cleaned_articles (title, content, label, source, timestamp)
                VALUES (%s, %s, %s, %s, %s);
            """, (row['title'], row['content'], int(row['label']), row['source'], row['timestamp']))
        self.conn.commit()
        self.cursor.close()
        self.conn.close()
        print("[CLEANING COMPLETE]")

if __name__ == "__main__":
    try:
        cleaner = DataCleaner()
        cleaner.clean_data()
        print("[SUCCESS] Cleaning pipeline finished.")
    except Exception as e:
        print(f"[FATAL ERROR] Cleaning failed: {e}")


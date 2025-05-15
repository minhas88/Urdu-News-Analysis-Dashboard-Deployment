#!/bin/bash
# Make sure PostgreSQL is ready
echo "⏳ Waiting for DB..."
bash app/wait-for-postgres.sh

# Optional: run scraper and training once
echo "🚀 Running scraper and cleaner..."
python3 app/scrapper.py
python3 app/cleaner.py

echo "🧠 Running ML model training..."
python3 app/train_model.py

# Launch Streamlit app
echo "🌐 Starting Streamlit app..."
streamlit run app/streamlit_app.py --server.port=8000 --server.address=0.0.0.0

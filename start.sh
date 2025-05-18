#!/bin/bash

# Wait for DB
echo "⏳ Waiting for DB..."
bash app/wait-for-postgres.sh

# Run scraping and training
echo "🚀 Scraping + Cleaning..."
python3 app/scrapper.py
python3 app/cleaner.py

echo "🧠 Training model..."
python3 app/train_model.py

# Run FastAPI in background
echo "🚀 Starting FastAPI (background)..."
uvicorn app.api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend (port 8080 will be exposed)
echo "🌐 Starting Streamlit (public)..."
streamlit run app/streamlit_app.py --server.port=8080 --server.address=0.0.0.0


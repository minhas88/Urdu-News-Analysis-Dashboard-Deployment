FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir mlflow psycopg2-binary

# Expose default MLflow tracking server port
EXPOSE 5000

# Set default command to run MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "postgresql://affan:pass123@postgres/news_db", "--default-artifact-root", "/mlruns"]


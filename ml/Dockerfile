FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for psycopg2 and some ML libs
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML training script
COPY train_model.py .

# Expose port if needed (usually ML training does not expose ports)
# EXPOSE 5000

# Default command can be overridden; typically you run training via CLI or Airflow
CMD ["python", "train_model.py"]


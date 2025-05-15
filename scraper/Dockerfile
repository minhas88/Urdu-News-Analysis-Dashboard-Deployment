FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by psycopg2 and others
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy scraper script
COPY scrapper.py .

# Expose port if needed (usually scraper does not expose ports)
# EXPOSE 8000

# Default command to run the scraper script (can be overridden)
CMD ["python", "scrapper.py"]


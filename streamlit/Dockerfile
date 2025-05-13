FROM python:3.10-slim

# Create dedicated user and config directory
RUN useradd -m -U appuser && \
    mkdir -p /home/appuser/.streamlit && \
    chown -R appuser:appuser /home/appuser

WORKDIR /app

# Install dependencies first for caching
COPY requirements.streamlit.txt .
RUN pip install --no-cache-dir -r requirements.streamlit.txt

# Copy app files
COPY streamlit_app.py .

# Set ownership and permissions
RUN chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Configure Streamlit environment
ENV STREAMLIT_CONFIG_DIR=/home/appuser/.streamlit
ENV STREAMLIT_DATA_DIR=/home/appuser/.streamlit/data
ENV STREAMLIT_GLOBAL_DEVELOPMENT_MODE=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Switch to non-root user
USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]

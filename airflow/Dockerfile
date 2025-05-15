FROM apache/airflow:2.7.1-python3.11

USER root

# Install additional system dependencies if needed (e.g., git, build-essential)
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy DAGs and scripts into the container
COPY dags /opt/airflow/dags
COPY scripts /opt/airflow/scripts

# Change ownership to airflow user
RUN chown -R airflow: /opt/airflow/dags /opt/airflow/scripts

USER airflow

# Set working directory (optional)
WORKDIR /opt/airflow

# Default entrypoint provided by airflow base image; start webserver and scheduler via docker-compose or Railway service config
# CMD can be overridden in docker-compose or Railway YAML if needed


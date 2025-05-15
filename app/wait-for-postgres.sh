#!/bin/bash
until pg_isready -h postgres -p 5432 -U affan -d airflow_db; do
  echo "Waiting for PostgreSQL..."
  sleep 2
done
exec "$@"

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    "start_date": datetime(2025, 5, 10),
    "catchup": False
}

with DAG("scrape_labeled_articles",
         default_args=default_args,
         schedule_interval="@daily",
         catchup=False,
         description="Scrape and clean Urdu news articles") as dag:
    
    run_scraper = BashOperator(
        task_id="run_scraper_labeled",
        bash_command="python /opt/airflow/scripts/scrapper.py"
    )

    run_cleaner = BashOperator(
        task_id="run_cleaner",
        bash_command="python /opt/airflow/scripts/cleaner.py"
    )

    run_ml_training = BashOperator(
        task_id="run_ml_training",
        bash_command="python /opt/airflow/scripts/train_model.py"
    )

    run_scraper >> run_cleaner >> run_ml_training


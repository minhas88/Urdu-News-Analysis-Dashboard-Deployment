# 📰 Urdu News Classification Pipeline

A fully containerized ETL + ML + Dashboard pipeline to scrape Urdu news articles, clean them, classify them using an ML model, and visualize predictions with a live dashboard. The project uses Apache Airflow for orchestration, BeautifulSoup and Requests for scraping, UrduHack for text preprocessing, Scikit-learn and TensorFlow for classification, MLflow for experiment tracking, FastAPI for serving predictions, PostgreSQL for data storage, and Streamlit with Plotly for interactive visualization. All components run seamlessly inside Docker containers managed with Docker Compose.

---

## 🚀 Architecture Overview

```
Airflow DAG (scraper → cleaner → model trainer → streamlit restart)
      |
      v
🗃️ PostgreSQL (stores raw, cleaned, and predicted data)
      |
      +--> ⚡ FastAPI (serves ML predictions via /cleaned-news and /inference)
      |
      +--> 📊 Streamlit (visualizes predictions and sentiment trends)
      +--> 📈 MLflow (logs model parameters, metrics, and artifacts)
```

---

## 📦 Tech Stack

| Component     | Tech                     |
|---------------|--------------------------|
| Scraper/ETL   | Apache Airflow + Python  |
| Data Storage  | PostgreSQL (via Docker)  |
| Model Serving | FastAPI + MLflow         |
| Visualization | Streamlit + Plotly       |
| Orchestration | Docker Compose           |

---

## 🛠️ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/urdu-news-classification-pipeline.git
cd urdu-news-classification-pipeline
```

### 2. Set execution permission on required files

```bash
chmod +x wait-for-postgres.sh
```

### 3. Create external Docker network (only once)

```bash
docker network create --driver bridge scraper_network
```

### 4. Build and run all services

```bash
docker-compose up --build
```

### 5. Access Services

- Airflow UI → [http://localhost:8080](http://localhost:8080) (default: admin/admin)
- FastAPI → [http://localhost:8000/inference](http://localhost:8000/inference)
- Streamlit → [http://localhost:8501](http://localhost:8501)
- MLflow UI → [http://localhost:5000](http://localhost:5000)

---

## ⚙️ Manual Airflow Usage

### Trigger a DAG run manually:
- Go to Airflow UI → DAGs → `scrape_labeled_articles`
- Click ▶️ to trigger manually

---

## 🧰 Common Fixes

### 🧪 MLflow Folder Permission Issue
If MLflow UI shows "no runs logged" but logs show tracking succeeded, fix volume permissions:
```bash
sudo rm -rf mlruns
mkdir mlruns
sudo chown -R $USER:$USER mlruns
```

### 🔐 Docker Socket Permissions
If you see `Permission denied` when Airflow tries to use DockerOperator:
```bash
sudo usermod -aG docker $USER
newgrp docker
sudo chmod 666 /var/run/docker.sock
```

### 📁 Airflow Log Permissions
If Airflow cannot write logs:
```bash
sudo rm -rf ./airflow/logs && mkdir -p ./airflow/logs
sudo chown -R 50000:0 ./airflow/logs
```

---

## 📁 Directory Structure

```
.
├── airflow/
│   ├── dags/
│   │   └── scrape_dag.py
│   └── logs/               # Airflow logs auto-generated
├── docker-compose.yml      # Service definitions
├── Dockerfile.*            # Docker build files for each service
├── fastAPI/
│   └── api.py              # FastAPI app
├── init/
│   └── postgres-init.sql   # PostgreSQL schema init
├── ml_model/
│   └── train_model.py      # ML training and prediction
├── mlruns/                 # MLflow experiment logs and models
├── scraper/
│   ├── scrapper.py         # News scraping logic
│   ├── cleaner.py          # Text normalization and preprocessing
│   └── stopwords-ur.txt    # Urdu stopword list
├── streamlit/
│   └── streamlit_app.py    # Streamlit dashboard
├── wait-for-postgres.sh    # Readiness check for Postgres
├── requirements.*.txt      # Service-specific dependencies
└── README.md
```

---

## 🧠 Contributors

* **Affan** – Streamlit dashboard & PostgreSQL integration
* **Mudasser** – Airflow ETL & news scraping
* **Usama** – ML model pipeline & FastAPI integration

---

## 📜 License

MIT License – Use, modify, and distribute freely.


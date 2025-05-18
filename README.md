# 📰 Urdu News Classification Dashboard

A streamlined ETL + ML + Dashboard pipeline for Urdu news classification. It scrapes Urdu-language news articles from Pakistani media, cleans the text using NLP (Stanza), classifies them using a logistic regression model with TF-IDF features, and visualizes results with an interactive Streamlit dashboard.

Deployed serverlessly on [Railway](https://railway.app/) using Nixpacks — no Docker needed.

---

## 🚀 Architecture Overview

```
Scraper (news) ──► Cleaner (Stanza) ──► Model (TF-IDF + LogisticRegression)
       │
       ▼
PostgreSQL DB
   ├── labeled_articles
   ├── cleaned_articles
   └── predictions
       ▲
       │
FastAPI (inference API) ◄── Streamlit (dashboard UI)
```

---

## 📦 Tech Stack

| Component     | Technology             |
|---------------|------------------------|
| Scraping      | BeautifulSoup + Requests |
| NLP Preproc   | Stanza (Urdu) + Stopwords |
| Model         | TF-IDF + LogisticRegression |
| Serving       | FastAPI                 |
| Visualization | Streamlit + Plotly     |
| Storage       | PostgreSQL (managed by Railway) |
| Infra         | Railway + Nixpacks     |

---

## 🛠️ Getting Started (Locally)

### 1. Clone the repo

```bash
git clone https://github.com/your-username/Urdu-News-Analysis-Dashboard-Deployment.git
cd Urdu-News-Analysis-Dashboard-Deployment
```

### 2. Set up Python environment and install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file with your Railway PostgreSQL credentials:

```env
POSTGRES_DB=railway
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password_here
POSTGRES_HOST=postgres.railway.internal
POSTGRES_PORT=5432
```

> 🚨 Do **not** commit `.env` to GitHub.

---

## 🔃 Pipeline Usage

```bash
# Step 1: Ensure PostgreSQL is running (Railway or local)
# Step 2: Run everything through the orchestration script
bash start.sh
```

`start.sh` performs:
- DB readiness check
- Web scraping from 4 news sites
- Urdu text cleaning
- Model training and evaluation
- Starts FastAPI on port 8000 and Streamlit on port 8080

---

## 🌐 Accessing Services (During Local Dev)

| Service    | URL                     |
|------------|--------------------------|
| FastAPI    | http://localhost:8000/inference |
| Streamlit  | http://localhost:8080    |

---

## 📁 Folder Structure

```
.
├── app/
│   ├── api.py              # FastAPI app
│   ├── cleaner.py          # Text preprocessing (stanza + stopwords)
│   ├── scrapper.py         # Scrapes Urdu news articles
│   ├── streamlit_app.py    # Streamlit dashboard
│   ├── train_model.py      # ML pipeline (TF-IDF + LogisticRegression)
│   ├── stopwords-ur.txt    # Urdu stopword list
│   └── wait-for-postgres.sh # DB readiness script
├── postgres-init.sql       # Optional schema bootstrap
├── requirements.txt        # Python dependencies
├── start.sh                # Run scraper → clean → train → launch
└── README.md
```

---

## 📜 License

MIT License – Free for personal and commercial use.

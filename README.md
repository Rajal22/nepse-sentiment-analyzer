# NEPSE News Sentiment Analyzer

A full ML pipeline for tracking market sentiment from Nepali financial news, built for retail investors, brokers, and analysts who currently lack tools to gauge NEPSE sentiment from local news sources.

## Problem

NEPSE is highly sentiment-driven, and retail investors often get trapped by market swings they didn't see coming because there's no accessible way to track sentiment from local financial news. This project builds that missing layer.

## What it does

- Scrapes financial news from ShareSansar (and a small MeroLagani sample)
- Classifies each article's sentiment (positive / negative / neutral) using a fine-tuned mBERT model
- Aggregates sentiment into a daily score, smoothed with a 7-day rolling average
- Correlates sentiment against the real NEPSE index to test whether news sentiment tracks market movement
- Serves live predictions and historical data through a FastAPI backend
- Visualizes everything in an interactive Streamlit dashboard, including stock-specific news search

## Tech stack

| Component | Tech |
|---|---|
| Scraping | Python, `requests`, `BeautifulSoup` |
| Labeling | Gemini API (LLM-assisted sentiment labeling) |
| Baseline model | Naive Bayes (`scikit-learn`, TF-IDF) |
| Deep learning | Fine-tuned `bert-base-multilingual-cased` (mBERT), trained on Google Colab (GPU) |
| Backend | FastAPI |
| Frontend | Streamlit + Plotly |

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes (baseline) | 55% | 0.55 |
| Fine-tuned mBERT | **79%** | **0.75** |

Sentiment-vs-NEPSE-index correlation was found to be weak (r < 0.1 across same-day, next-day, and weekly windows) — a finding that held consistent across both the Naive Bayes and mBERT model outputs, suggesting the limitation lies in single-source/low daily article volume rather than model quality. See `notebooks/` for the full analysis.

## Project structure

```
nepse-sentiment-analyzer/
├── backend/
│   └── main.py                 # FastAPI app - model inference + data endpoints
├── frontend/
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   ├── scrape_sharesansar.ipynb
│   ├── scrape_merolagani.ipynb
│   ├── clean_data.ipynb
│   ├── label_sentiment.ipynb
│   ├── train_naive_bayes.ipynb
│   └── nepse_mbert_finetuning.ipynb   # run on Google Colab (GPU)
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1      # Windows
   pip install -r requirements.txt
   ```

2. Run the notebooks in `notebooks/` in order to regenerate the dataset (scraping → cleaning → labeling → training). Raw data and trained model weights are not included in this repo (see `.gitignore`) due to size — you'll need to run the pipeline yourself, or reach out for the dataset.

3. Fine-tune mBERT on Google Colab using `nepse_mbert_finetuning.ipynb` (requires GPU runtime), then download the resulting model folder into `backend/nepse_mbert_final/`.

4. Start the backend:
   ```
   cd backend
   uvicorn main:app --reload --port 8000
   ```

5. Start the dashboard (in a separate terminal):
   ```
   cd frontend
   streamlit run app.py
   ```

## Data sources

- [ShareSansar](https://www.sharesansar.com) - NEPSE news, IPO/FPO news, mutual fund news, and market data
- [MeroLagani](https://merolagani.com) - supplementary Nepali-language financial news
- NEPSE index and market capitalization historical data

## Limitations & future scope

- MeroLagani scraping is currently limited to first-page-per-category due to AJAX-based pagination; full pagination support is a planned improvement
- Facebook post sentiment (originally scoped) was excluded due to platform scraping restrictions
- Weak sentiment-index correlation suggests future work should focus on increasing daily article volume and adding more news sources before building automated trading signals
- Planned: automated trading signal generation as a longer-term extension

## Author

Rajal Maharjan — Computer Science (Data Science), Taylor's University
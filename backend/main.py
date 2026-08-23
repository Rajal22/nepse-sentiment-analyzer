"""
NEPSE Sentiment Analyzer - FastAPI Backend

Run with: uvicorn main:app --reload --port 8000
Requires: nepse_mbert_final/ (your downloaded model folder) in the same directory,
          sharesansar_mbert_predictions.csv, and the NEPSE index Excel file.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import os

app = FastAPI(title="NEPSE Sentiment Analyzer API")

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = "nepse_mbert_final"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

tokenizer = None
model = None

# ---------------------------------------------------------------------------
# Load precomputed data once at startup
# ---------------------------------------------------------------------------
ARTICLES_FILE = "dataset/sharesansar_mbert_predictions.csv"
NEPSE_INDEX_FILE = "dataset/NEPSE-index-and-market-capitalization (1) (1).xlsx"

articles_df = None
daily_sentiment_df = None
nepse_index_df = None


@app.on_event("startup")
def load_resources():
    global tokenizer, model, articles_df, daily_sentiment_df, nepse_index_df

    print("Loading mBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded.")

    print("Loading articles and computing daily sentiment...")
    articles_df = pd.read_csv(ARTICLES_FILE)
    articles_df["date"] = pd.to_datetime(articles_df["date"], errors="coerce")

    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    articles_df["sentiment_score"] = articles_df["mbert_sentiment"].map(sentiment_map)

    dated = articles_df.dropna(subset=["date"])
    daily = dated.groupby(dated["date"].dt.date).agg(
        avg_sentiment=("sentiment_score", "mean"),
        article_count=("sentiment_score", "count"),
        positive_count=("mbert_sentiment", lambda x: (x == "positive").sum()),
        negative_count=("mbert_sentiment", lambda x: (x == "negative").sum()),
        neutral_count=("mbert_sentiment", lambda x: (x == "neutral").sum()),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["sentiment_7day_avg"] = daily["avg_sentiment"].rolling(window=7, min_periods=1).mean()
    daily_sentiment_df = daily
    print(f"Daily sentiment computed for {len(daily)} days.")

    print("Loading NEPSE index data...")
    if os.path.exists(NEPSE_INDEX_FILE):
        nepse_raw = pd.read_excel(NEPSE_INDEX_FILE, sheet_name="NEPSE Index", header=1)
        nepse_clean = nepse_raw[["Date/Month", "Nepse"]].copy()
        nepse_clean.columns = ["date", "nepse_index"]
        nepse_clean["date"] = pd.to_datetime(nepse_clean["date"], errors="coerce")
        nepse_index_df = nepse_clean.dropna(subset=["date", "nepse_index"])
        print(f"NEPSE index loaded: {len(nepse_index_df)} rows.")
    else:
        nepse_index_df = pd.DataFrame(columns=["date", "nepse_index"])
        print("WARNING: NEPSE index file not found, overlay data will be empty.")


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str
    title: Optional[str] = None


class PredictResponse(BaseModel):
    sentiment: str
    confidence_scores: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "NEPSE Sentiment Analyzer API"}


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(req: PredictRequest):
    """Predict sentiment for a new piece of text using fine-tuned mBERT."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = req.text[:2000]
    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = torch.argmax(probs).item()
    confidence_scores = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(len(ID2LABEL))}

    return PredictResponse(sentiment=ID2LABEL[pred_id], confidence_scores=confidence_scores)


@app.get("/daily-sentiment")
def get_daily_sentiment(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Return the daily sentiment score time series, optionally filtered by date range."""
    df = daily_sentiment_df.copy()
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.get("/nepse-index")
def get_nepse_index(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Return the real NEPSE index time series, optionally filtered by date range."""
    df = nepse_index_df.copy()
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")


@app.get("/articles")
def get_articles(
    date: Optional[str] = None,
    sentiment: Optional[str] = None,
    stock_keyword: Optional[str] = None,
    limit: int = 50,
):
    """
    Return recent articles, optionally filtered by:
    - date (YYYY-MM-DD)
    - sentiment (positive/negative/neutral)
    - stock_keyword (matches in title or text - powers 'stock-specific alerts')
    """
    df = articles_df.copy()

    if date:
        target_date = pd.to_datetime(date).date()
        df = df[df["date"].dt.date == target_date]
    if sentiment:
        df = df[df["mbert_sentiment"] == sentiment]
    if stock_keyword:
        mask = (
            df["title"].str.contains(stock_keyword, case=False, na=False)
            | df["text"].str.contains(stock_keyword, case=False, na=False)
        )
        df = df[mask]

    df = df.sort_values("date", ascending=False).head(limit)
    result = df[["title", "url", "date", "category", "mbert_sentiment"]].copy()
    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result = result.rename(columns={"mbert_sentiment": "sentiment"})

    return result.to_dict(orient="records")


@app.get("/summary")
def get_summary():
    """Overall dataset summary stats for the dashboard header."""
    total_articles = len(articles_df)
    sentiment_counts = articles_df["mbert_sentiment"].value_counts().to_dict()
    date_min = articles_df["date"].min()
    date_max = articles_df["date"].max()
    latest_sentiment = daily_sentiment_df.iloc[-1]["sentiment_7day_avg"] if len(daily_sentiment_df) else None

    return {
        "total_articles": total_articles,
        "sentiment_distribution": sentiment_counts,
        "date_range": {
            "start": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else None,
            "end": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else None,
        },
        "latest_7day_sentiment": round(latest_sentiment, 3) if latest_sentiment is not None else None,
    }

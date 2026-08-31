"""
NEPSE Sentiment Analyzer - FastAPI Backend (MongoDB-backed)

Run with: uvicorn main:app --reload --port 8000
Requires: nepse_mbert_final/ (your downloaded model folder) in the same directory,
          and a MONGODB_URI + GEMINI_API_KEY set in a .env file (or as real
          environment variables when deployed).
"""

import os
import certifi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pymongo import MongoClient
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="NEPSE Sentiment Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "nepse_mbert_final"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI not set - check your .env file or environment variables")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not set - check your .env file")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-flash-lite-latest")

tokenizer = None
model = None
mongo_client = None
db = None


@app.on_event("startup")
def load_resources():
    global tokenizer, model, mongo_client, db

    print("Connecting to MongoDB...")
    mongo_client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
    db = mongo_client["nepse_sentiment"]
    print(f"Connected. Collections: {db.list_collection_names()}")

    print("Loading mBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    print("Model loaded.")


@app.on_event("shutdown")
def close_mongo():
    if mongo_client:
        mongo_client.close()


class PredictRequest(BaseModel):
    text: str
    title: Optional[str] = None


class PredictResponse(BaseModel):
    sentiment: str
    confidence_scores: dict


class SummarizeRequest(BaseModel):
    text: str
    title: Optional[str] = None


@app.get("/")
def root():
    return {"status": "ok", "message": "NEPSE Sentiment Analyzer API"}


@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(req: PredictRequest):
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

@app.post("/summarize")
def summarize_article(req: SummarizeRequest):
    """Generate a short summary of an article using Gemini."""
    prompt = f"""Summarize this NEPSE/financial news article in 2-3 concise sentences.
Keep it factual and neutral. If the article is in Nepali, respond in English.

Title: {req.title or ''}
Article: {req.text[:3000]}

Summary:"""

    try:
        response = gemini_model.generate_content(prompt)
        return {"summary": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {str(e)}")


@app.get("/daily-sentiment")
def get_daily_sentiment(start_date: Optional[str] = None, end_date: Optional[str] = None):
    query = {}
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = pd.to_datetime(start_date).to_pydatetime()
        if end_date:
            date_filter["$lte"] = pd.to_datetime(end_date).to_pydatetime()
        query["date"] = date_filter

    docs = list(db.daily_sentiment.find(query, {"_id": 0}).sort("date", 1))
    for d in docs:
        d["date"] = d["date"].strftime("%Y-%m-%d")
    return docs


@app.get("/nepse-index")
def get_nepse_index(start_date: Optional[str] = None, end_date: Optional[str] = None):
    query = {}
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = pd.to_datetime(start_date).to_pydatetime()
        if end_date:
            date_filter["$lte"] = pd.to_datetime(end_date).to_pydatetime()
        query["date"] = date_filter

    docs = list(db.nepse_index.find(query, {"_id": 0}).sort("date", 1))
    for d in docs:
        d["date"] = d["date"].strftime("%Y-%m-%d")
    return docs


@app.get("/articles")
def get_articles(
    date: Optional[str] = None,
    sentiment: Optional[str] = None,
    stock_keyword: Optional[str] = None,
    limit: int = 50,
):
    query = {}

    if date:
        target_date = pd.to_datetime(date)
        next_day = target_date + pd.Timedelta(days=1)
        query["date"] = {"$gte": target_date.to_pydatetime(), "$lt": next_day.to_pydatetime()}

    if sentiment:
        query["mbert_sentiment"] = sentiment

    if stock_keyword:
        query["$text"] = {"$search": stock_keyword}

    projection = {"_id": 0, "title": 1, "url": 1, "date": 1, "category": 1, "mbert_sentiment": 1, "text": 1}
    docs = list(db.articles.find(query, projection).sort("date", -1).limit(limit))

    for d in docs:
        if d.get("date"):
            d["date"] = d["date"].strftime("%Y-%m-%d")
        d["sentiment"] = d.pop("mbert_sentiment", None)

    return docs


@app.get("/summary")
def get_summary():
    total_articles = db.articles.count_documents({})

    pipeline = [{"$group": {"_id": "$mbert_sentiment", "count": {"$sum": 1}}}]
    sentiment_counts = {doc["_id"]: doc["count"] for doc in db.articles.aggregate(pipeline)}

    date_range = list(db.articles.aggregate([
        {"$group": {"_id": None, "min_date": {"$min": "$date"}, "max_date": {"$max": "$date"}}}
    ]))

    latest_sentiment_doc = db.daily_sentiment.find_one(sort=[("date", -1)])
    latest_sentiment = latest_sentiment_doc.get("sentiment_7day_avg") if latest_sentiment_doc else None

    return {
        "total_articles": total_articles,
        "sentiment_distribution": sentiment_counts,
        "date_range": {
            "start": date_range[0]["min_date"].strftime("%Y-%m-%d") if date_range else None,
            "end": date_range[0]["max_date"].strftime("%Y-%m-%d") if date_range else None,
        },
        "latest_7day_sentiment": round(latest_sentiment, 3) if latest_sentiment is not None else None,
    }
"""
Load scraped/labeled NEPSE data into MongoDB Atlas.
Run once locally: python load_to_mongodb.py
"""

import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MONGODB_URI not found - check your .env file")

client = MongoClient(MONGODB_URI)
db = client["nepse_sentiment"]

# ---------------------------------------------------------------------------
# Load articles collection
# ---------------------------------------------------------------------------
print("Loading articles...")
articles_df = pd.read_csv("dataset/sharesansar_mbert_predictions.csv")
articles_df["date"] = pd.to_datetime(articles_df["date"], errors="coerce")

articles_collection = db["articles"]
articles_collection.delete_many({})  # clear existing data for a clean reload

records = articles_df.to_dict(orient="records")
for r in records:
    if pd.notna(r.get("date")):
        r["date"] = r["date"].to_pydatetime()
    else:
        r["date"] = None

articles_collection.insert_many(records)
print(f"Inserted {len(records)} articles")

# create indexes for common query patterns
articles_collection.create_index("date")
articles_collection.create_index("mbert_sentiment")
articles_collection.create_index([("title", "text"), ("text", "text")])
print("Created indexes on articles collection")

# ---------------------------------------------------------------------------
# Compute and load daily sentiment collection
# ---------------------------------------------------------------------------
print("\nComputing daily sentiment...")
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

daily_collection = db["daily_sentiment"]
daily_collection.delete_many({})

daily_records = daily.to_dict(orient="records")
for r in daily_records:
    r["date"] = r["date"].to_pydatetime()

daily_collection.insert_many(daily_records)
daily_collection.create_index("date")
print(f"Inserted {len(daily_records)} daily sentiment records")

# ---------------------------------------------------------------------------
# Load NEPSE index collection
# ---------------------------------------------------------------------------
print("\nLoading NEPSE index data...")
nepse_raw = pd.read_excel(
    "dataset/NEPSE-index-and-market-capitalization (1) (1).xlsx",
    sheet_name="NEPSE Index",
    header=1
)
nepse_clean = nepse_raw[["Date/Month", "Nepse"]].copy()
nepse_clean.columns = ["date", "nepse_index"]
nepse_clean["date"] = pd.to_datetime(nepse_clean["date"], errors="coerce")
nepse_clean = nepse_clean.dropna(subset=["date", "nepse_index"])

nepse_collection = db["nepse_index"]
nepse_collection.delete_many({})

nepse_records = nepse_clean.to_dict(orient="records")
for r in nepse_records:
    r["date"] = r["date"].to_pydatetime()

nepse_collection.insert_many(nepse_records)
nepse_collection.create_index("date")
print(f"Inserted {len(nepse_records)} NEPSE index records")

print("\nAll data loaded successfully.")
print(f"Collections in database: {db.list_collection_names()}")

client.close()
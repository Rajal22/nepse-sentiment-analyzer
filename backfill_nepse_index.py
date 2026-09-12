"""
One-time backfill: extract NEPSE index closing values from already-scraped
ShareSansar articles (they mention the closing index in the text), and add
them to the nepse_index collection to fill the gap after the static Excel
data (which stops Dec 2025).

Run once: python backfill_nepse_index.py
"""

import os
import re
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client["nepse_sentiment"]


def extract_nepse_index(text):
    """Find a 'closed at X,XXX.XX' style mention of the NEPSE index in article text."""
    if not text:
        return None
    match = re.search(r"clos(?:e|ed)\s+at\s+([\d,]+\.\d+)", text, re.IGNORECASE)
    if match:
        value_str = match.group(1).replace(",", "")
        try:
            value = float(value_str)
            # sanity check - NEPSE index has historically been in a reasonable range
            if 500 < value < 10000:
                return value
        except ValueError:
            pass
    return None


def main():
    # only look at market-recap style categories, and only dates after our
    # static Excel data ends, to avoid overwriting good historical data
    candidates = db.articles.find({
        "category": {"$in": ["nepse-news", "stock-market"]},
        "date": {"$gte": __import__("datetime").datetime(2025, 12, 16)},
    })

    extracted = {}  # date -> value, dedup by keeping first found per day
    count_checked = 0
    count_found = 0

    for article in candidates:
        count_checked += 1
        value = extract_nepse_index(article.get("text", ""))
        if value is not None and article.get("date"):
            day = article["date"].date()
            if day not in extracted:
                extracted[day] = value
                count_found += 1

    print(f"Checked {count_checked} candidate articles")
    print(f"Extracted NEPSE index values for {count_found} distinct days")

    if not extracted:
        print("Nothing to backfill.")
        return

    from datetime import datetime
    inserted = 0
    updated = 0

    for day, value in extracted.items():
        day_dt = datetime.combine(day, datetime.min.time())
        result = db.nepse_index.update_one(
            {"date": day_dt},
            {"$set": {"date": day_dt, "nepse_index": value}},
            upsert=True,
        )
        if result.upserted_id:
            inserted += 1
        elif result.modified_count:
            updated += 1

    print(f"Inserted {inserted} new NEPSE index records")
    print(f"Updated {updated} existing records")

    total = db.nepse_index.count_documents({})
    print(f"\nTotal nepse_index records now: {total}")

    latest = db.nepse_index.find_one(sort=[("date", -1)])
    print(f"Latest date in nepse_index: {latest['date'] if latest else 'none'}")


if __name__ == "__main__":
    main()

"""
Scrape new ShareSansar articles, label them with Gemini, and insert into MongoDB.
Only processes articles not already in the database (dedup by URL).

Run manually: python update_news.py
Or scheduled via GitHub Actions (see .github/workflows/update_news.yml)

Requires environment variables: MONGODB_URI, GEMINI_API_KEY
"""

import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import certifi
import google.generativeai as genai
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "https://www.sharesansar.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
DELAY_SECONDS = 1.5
MAX_PAGES_PER_CATEGORY = 2  # only check the newest couple pages - we only want NEW articles

CATEGORIES = ["nepse-news", "stock-market", "share-listed", "ipo-fpo-news", "mutual-fund", "featured"]

MONGODB_URI = os.environ["MONGODB_URI"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-flash-lite-latest")

client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where())
db = client["nepse_sentiment"]


# ---------------------------------------------------------------------------
# Scraping (reused logic from the original scraper)
# ---------------------------------------------------------------------------
def get_article_links(category, max_pages=MAX_PAGES_PER_CATEGORY):
    seen = {}
    url = f"{BASE}/category/{category}"
    page_count = 0
    article_pattern = re.compile(r"/newsdetail/.*-\d{4}-\d{2}-\d{2}$")

    while url and page_count < max_pages:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=article_pattern)

        for link in links:
            title = link.get_text(strip=True)
            href = link["href"]
            if not href.startswith("http"):
                href = BASE + href
            if not title:
                continue
            if href not in seen or len(title) > len(seen[href]):
                seen[href] = title

        next_link = soup.find("a", string=re.compile("Next", re.IGNORECASE))
        if next_link and next_link.get("href"):
            next_href = next_link["href"]
            if next_href.startswith("http"):
                url = next_href
            elif next_href.startswith("?"):
                base_path = url.split("?")[0]
                url = base_path + next_href
            else:
                url = BASE + next_href
        else:
            url = None

        page_count += 1
        time.sleep(DELAY_SECONDS)

    articles = []
    for href, title in seen.items():
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})$", href)
        date = date_match.group(1) if date_match else None
        articles.append({"title": title, "url": href, "date": date})

    return articles


def get_article_body(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        body_lines = []
        skip_markers = [
            "Dhalko Linkroad", "sharesansar@gmail.com", "Regd No",
            "Share on Facebook", "Share on Twitter", "IMS Investment",
        ]

        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) < 20:
                continue
            if any(marker in text for marker in skip_markers):
                continue
            body_lines.append(text)

        return " ".join(body_lines)

    except requests.RequestException as e:
        print(f"  Error fetching {url}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
BATCH_PROMPT = """You are labeling financial news for a NEPSE (Nepal Stock Exchange) sentiment analysis dataset.

For EACH article below, classify its likely impact on investor sentiment toward NEPSE or the specific companies/sector mentioned.

Respond with ONLY a JSON array, no other text, one object per article in the same order:
[{{"label": "positive", "confidence": "high"}}, {{"label": "neutral", "confidence": "medium"}}, ...]

label must be exactly one of: positive, negative, neutral
confidence must be exactly one of: high, medium, low

Rules:
- positive = bullish news (profit growth, gains, positive reforms, good IPO news, index rising)
- negative = bearish news (losses, declines, fraud, suspensions, negative regulation, index falling)
- neutral = purely factual/informational with no clear positive or negative slant

Articles:
{articles}
"""


def label_batch(rows, max_chars=800):
    articles_text = ""
    for i, row in enumerate(rows):
        articles_text += f"\n[Article {i+1}]\nTitle: {row['title']}\nText: {row['text'][:max_chars]}\n"

    prompt = BATCH_PROMPT.format(articles=articles_text)

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip().replace("```json", "").replace("```", "").strip()
        results = json.loads(raw)
        if len(results) != len(rows):
            print(f"  Warning: expected {len(rows)} labels, got {len(results)}")
        return results
    except Exception as e:
        print(f"  Error labeling batch: {e}")
        return [{"label": "error", "confidence": "error"}] * len(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[{datetime.now()}] Checking for new articles...")

    existing_urls = set(db.articles.distinct("url"))
    print(f"Database currently has {len(existing_urls)} articles")

    all_candidates = []
    for category in CATEGORIES:
        print(f"Checking category: {category}")
        links = get_article_links(category)
        for link in links:
            link["category"] = category
        all_candidates.extend(links)

    # dedupe and filter to only genuinely new articles
    seen_urls = set()
    new_articles = []
    for a in all_candidates:
        if a["url"] in existing_urls or a["url"] in seen_urls:
            continue
        seen_urls.add(a["url"])
        new_articles.append(a)

    print(f"Found {len(new_articles)} new articles")

    if not new_articles:
        print("Nothing new to add. Exiting.")
        return

    # fetch bodies
    print("Fetching article bodies...")
    for article in new_articles:
        article["text"] = get_article_body(article["url"])
        time.sleep(DELAY_SECONDS)

    # drop any with empty text
    new_articles = [a for a in new_articles if len(a.get("text", "")) > 20]
    print(f"{len(new_articles)} articles with valid text")

    if not new_articles:
        print("No valid articles after body fetch. Exiting.")
        return

    # label in batches of 50
    print("Labeling with Gemini...")
    BATCH_SIZE = 50
    for i in range(0, len(new_articles), BATCH_SIZE):
        batch = new_articles[i:i + BATCH_SIZE]
        results = label_batch(batch)
        for article, result in zip(batch, results):
            article["mbert_sentiment"] = result.get("label", "error")  # keep field name consistent with existing data
            article["confidence"] = result.get("confidence", "error")
        time.sleep(2)

    # drop labeling errors
    new_articles = [a for a in new_articles if a["mbert_sentiment"] != "error"]
    print(f"{len(new_articles)} articles successfully labeled")

    # convert date strings to datetime objects for MongoDB
    for a in new_articles:
        if a.get("date"):
            try:
                a["date"] = datetime.strptime(a["date"], "%Y-%m-%d")
            except ValueError:
                a["date"] = None

    if new_articles:
        db.articles.insert_many(new_articles)
        print(f"Inserted {len(new_articles)} new articles into MongoDB")

    # recompute daily_sentiment from scratch (dataset is small enough this is cheap and always correct)
    print("Recomputing daily sentiment...")
    all_docs = list(db.articles.find({}, {"date": 1, "mbert_sentiment": 1}))

    from collections import defaultdict
    daily_data = defaultdict(list)
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}

    for doc in all_docs:
        if doc.get("date") and doc.get("mbert_sentiment") in sentiment_map:
            day = doc["date"].date()
            daily_data[day].append(doc["mbert_sentiment"])

    daily_records = []
    for day in sorted(daily_data.keys()):
        sentiments = daily_data[day]
        scores = [sentiment_map[s] for s in sentiments]
        daily_records.append({
            "date": datetime.combine(day, datetime.min.time()),
            "avg_sentiment": sum(scores) / len(scores),
            "article_count": len(sentiments),
            "positive_count": sentiments.count("positive"),
            "negative_count": sentiments.count("negative"),
            "neutral_count": sentiments.count("neutral"),
        })

    # compute 7-day rolling average
    for i, record in enumerate(daily_records):
        window = daily_records[max(0, i - 6):i + 1]
        record["sentiment_7day_avg"] = sum(r["avg_sentiment"] for r in window) / len(window)

    db.daily_sentiment.delete_many({})
    db.daily_sentiment.insert_many(daily_records)
    print(f"Recomputed {len(daily_records)} days of sentiment data")

    print(f"[{datetime.now()}] Update complete.")


if __name__ == "__main__":
    main()

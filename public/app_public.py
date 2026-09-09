"""
NEPSE Sentiment Analyzer - Public Streamlit Dashboard

Connects directly to MongoDB Atlas (no FastAPI backend required).
Deploy on Streamlit Community Cloud: share.streamlit.io

Secrets required (set in Streamlit Cloud's "Secrets" settings, not in this file):
MONGODB_URI = "your connection string"
GEMINI_API_KEY = "your gemini key"
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
import google.generativeai as genai

st.set_page_config(page_title="NEPSE Sentiment Analyzer", layout="wide")

# ---------------------------------------------------------------------------
# Connect to MongoDB and Gemini using Streamlit secrets
# ---------------------------------------------------------------------------
@st.cache_resource
def get_db_connection():
    client = MongoClient(st.secrets["MONGODB_URI"])
    return client["nepse_sentiment"]


@st.cache_resource
def get_gemini_model():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-flash-lite-latest")


db = get_db_connection()
gemini_model = get_gemini_model()

st.title("📈 NEPSE Stock Market News Sentiment Analyzer")
st.caption("Daily sentiment tracking from Nepali financial news, powered by fine-tuned mBERT")


# ---------------------------------------------------------------------------
# Data fetching helpers (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_summary():
    total_articles = db.articles.count_documents({})

    pipeline = [{"$group": {"_id": "$mbert_sentiment", "count": {"$sum": 1}}}]
    sentiment_counts = {doc["_id"]: doc["count"] for doc in db.articles.aggregate(pipeline)}

    date_range = list(db.articles.aggregate([
        {"$group": {"_id": None, "min_date": {"$min": "$date"}, "max_date": {"$max": "$date"}}}
    ]))

    latest_doc = db.daily_sentiment.find_one(sort=[("date", -1)])
    latest_sentiment = latest_doc.get("sentiment_7day_avg") if latest_doc else None

    return {
        "total_articles": total_articles,
        "sentiment_distribution": sentiment_counts,
        "date_range": {
            "start": date_range[0]["min_date"].strftime("%Y-%m-%d") if date_range else None,
            "end": date_range[0]["max_date"].strftime("%Y-%m-%d") if date_range else None,
        },
        "latest_7day_sentiment": round(latest_sentiment, 3) if latest_sentiment is not None else None,
    }


@st.cache_data(ttl=300)
def fetch_daily_sentiment():
    docs = list(db.daily_sentiment.find({}, {"_id": 0}).sort("date", 1))
    df = pd.DataFrame(docs)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def fetch_nepse_index():
    docs = list(db.nepse_index.find({}, {"_id": 0}).sort("date", 1))
    df = pd.DataFrame(docs)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def fetch_articles(date=None, sentiment=None, stock_keyword=None, limit=50):
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

    df = pd.DataFrame(docs)
    if len(df):
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["sentiment"] = df.pop("mbert_sentiment")
    return df


def summarize_article(text, title=None):
    prompt = f"""Summarize this NEPSE/financial news article in 2-3 concise sentences.
Keep it factual and neutral. If the article is in Nepali, respond in English.

Title: {title or ''}
Article: {text[:3000]}

Summary:"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate summary: {e}"


# ---------------------------------------------------------------------------
# Top summary metrics
# ---------------------------------------------------------------------------
summary = fetch_summary()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Articles", f"{summary['total_articles']:,}")

with col2:
    dist = summary["sentiment_distribution"]
    pos_pct = dist.get("positive", 0) / summary["total_articles"] * 100
    st.metric("Positive Articles", f"{dist.get('positive', 0):,}", f"{pos_pct:.0f}%")

with col3:
    neg_pct = dist.get("negative", 0) / summary["total_articles"] * 100
    st.metric("Negative Articles", f"{dist.get('negative', 0):,}", f"{neg_pct:.0f}%")

with col4:
    latest = summary.get("latest_7day_sentiment")
    if latest is not None:
        mood = "Bullish 🟢" if latest > 0.15 else "Bearish 🔴" if latest < -0.15 else "Neutral 🟡"
        st.metric("Current 7-Day Sentiment", f"{latest:+.2f}", mood)

st.caption(f"Data range: {summary['date_range']['start']} to {summary['date_range']['end']}")

st.divider()

# ---------------------------------------------------------------------------
# Sentiment vs NEPSE Index overlay
# ---------------------------------------------------------------------------
st.subheader("Sentiment Trend vs NEPSE Index")

daily_df = fetch_daily_sentiment()
nepse_df = fetch_nepse_index()

if len(daily_df) and len(nepse_df):
    merged = pd.merge(daily_df, nepse_df, on="date", how="inner").sort_values("date")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=merged["date"], y=merged["sentiment_7day_avg"],
                   name="Sentiment (7-day avg)", line=dict(color="#1f77b4", width=2)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=merged["date"], y=merged["nepse_index"],
                   name="NEPSE Index", line=dict(color="#d62728", width=1.5), opacity=0.6),
        secondary_y=True,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4, secondary_y=False)
    fig.update_layout(
        height=450, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=False)
    fig.update_yaxes(title_text="NEPSE Index", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("About this correlation"):
        corr = merged["sentiment_7day_avg"].corr(merged["nepse_index"].diff())
        st.write(
            f"Same-day correlation between 7-day sentiment and NEPSE index change: **{corr:.3f}**. "
            "This is a weak correlation, reflecting that single-source news sentiment is just one of many "
            "factors driving NEPSE movement."
        )
else:
    st.warning("Not enough overlapping data to plot.")

st.divider()

# ---------------------------------------------------------------------------
# Distribution + volume
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
    dist = summary["sentiment_distribution"]
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(dist.keys()), values=list(dist.values()),
        marker=dict(colors=["#2ca02c", "#7f7f7f", "#d62728"]), hole=0.4,
    )])
    fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Daily Article Volume")
    if len(daily_df):
        fig_vol = go.Figure(data=[go.Bar(x=daily_df["date"], y=daily_df["article_count"], marker_color="#1f77b4")])
        fig_vol.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Stock-specific search
# ---------------------------------------------------------------------------
st.subheader("🔍 Stock-Specific News Search")
st.caption("Search for news mentioning a specific company or stock symbol")

search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    stock_query = st.text_input("Company or keyword", placeholder="e.g. Nabil Bank, NEPSE, hydropower")
with search_col2:
    sentiment_filter = st.selectbox("Filter by sentiment", ["All", "positive", "negative", "neutral"])

if stock_query:
    filter_sentiment = None if sentiment_filter == "All" else sentiment_filter
    results = fetch_articles(stock_keyword=stock_query, sentiment=filter_sentiment, limit=30)

    if len(results):
        st.write(f"Found {len(results)} articles")
        for idx, row in results.iterrows():
            sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(row["sentiment"], "")
            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown(f"{sentiment_emoji} **[{row['title']}]({row['url']})** — {row['date']} · {row['category']}")
            with col_b:
                if st.button("Summarize", key=f"summarize_search_{idx}"):
                    with st.spinner("Summarizing..."):
                        summary_text = summarize_article(row.get("text", ""), row["title"])
                    st.info(summary_text)
    else:
        st.info("No articles found matching that search.")

st.divider()

# ---------------------------------------------------------------------------
# Note about the live model demo (not included in this public deployment)
# ---------------------------------------------------------------------------
st.info(
    "💡 A live sentiment-prediction demo (paste any text, get instant mBERT classification) "
    "is available in the full local version with the FastAPI backend running. "
    "It's excluded here to keep this public deployment lightweight and free to host."
)

st.divider()

# ---------------------------------------------------------------------------
# Recent articles
# ---------------------------------------------------------------------------
st.subheader("📰 Recent Articles")
recent = fetch_articles(limit=20)
if len(recent):
    for idx, row in recent.iterrows():
        sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(row["sentiment"], "")
        col_a, col_b = st.columns([5, 1])
        with col_a:
            st.markdown(f"{sentiment_emoji} **[{row['title']}]({row['url']})** — {row['date']} · {row['category']}")
        with col_b:
            if st.button("Summarize", key=f"summarize_recent_{idx}"):
                with st.spinner("Summarizing..."):
                    summary_text = summarize_article(row.get("text", ""), row["title"])
                st.info(summary_text)
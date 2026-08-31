"""
NEPSE Sentiment Analyzer - Streamlit Dashboard

Run with: streamlit run app.py
Requires the FastAPI backend running at http://localhost:8000
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="NEPSE Sentiment Analyzer", layout="wide")

st.title("📈 NEPSE Stock Market News Sentiment Analyzer")
st.caption("Daily sentiment tracking from Nepali financial news, powered by fine-tuned mBERT")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_summary():
    resp = requests.get(f"{API_BASE}/summary")
    return resp.json()


@st.cache_data(ttl=300)
def fetch_daily_sentiment():
    resp = requests.get(f"{API_BASE}/daily-sentiment")
    df = pd.DataFrame(resp.json())
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def fetch_nepse_index():
    resp = requests.get(f"{API_BASE}/nepse-index")
    df = pd.DataFrame(resp.json())
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=300)
def fetch_articles(date=None, sentiment=None, stock_keyword=None, limit=50):
    params = {"limit": limit}
    if date:
        params["date"] = date
    if sentiment:
        params["sentiment"] = sentiment
    if stock_keyword:
        params["stock_keyword"] = stock_keyword
    resp = requests.get(f"{API_BASE}/articles", params=params)
    return pd.DataFrame(resp.json())


def predict_new_text(text):
    resp = requests.post(f"{API_BASE}/predict", json={"text": text})
    return resp.json()

def summarize_article(text, title=None):
    resp = requests.post(f"{API_BASE}/summarize", json={"text": text, "title": title})
    if resp.status_code == 200:
        return resp.json()["summary"]
    return "Could not generate summary."


# ---------------------------------------------------------------------------
# Check backend connectivity
# ---------------------------------------------------------------------------
try:
    summary = fetch_summary()
except requests.exceptions.ConnectionError:
    st.error("Cannot connect to the backend API. Make sure it's running: `uvicorn main:app --reload` in the backend folder.")
    st.stop()


# ---------------------------------------------------------------------------
# Top summary metrics
# ---------------------------------------------------------------------------
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
# Main chart: Sentiment vs NEPSE Index overlay
# ---------------------------------------------------------------------------
st.subheader("Sentiment Trend vs NEPSE Index")

daily_df = fetch_daily_sentiment()
nepse_df = fetch_nepse_index()

if len(daily_df) and len(nepse_df):
    merged = pd.merge(daily_df, nepse_df, on="date", how="inner").sort_values("date")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=merged["date"], y=merged["sentiment_7day_avg"],
            name="Sentiment (7-day avg)", line=dict(color="#1f77b4", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=merged["date"], y=merged["nepse_index"],
            name="NEPSE Index", line=dict(color="#d62728", width=1.5), opacity=0.6,
        ),
        secondary_y=True,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4, secondary_y=False)

    fig.update_layout(
        height=450,
        hovermode="x unified",
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
            "factors driving NEPSE movement (alongside macroeconomic conditions, liquidity, remittances, "
            "and broader market activity)."
        )
else:
    st.warning("Not enough overlapping data between sentiment and NEPSE index to plot.")

st.divider()

# ---------------------------------------------------------------------------
# Sentiment distribution pie + daily article volume
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Sentiment Distribution")
    dist = summary["sentiment_distribution"]
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(dist.keys()),
        values=list(dist.values()),
        marker=dict(colors=["#2ca02c", "#7f7f7f", "#d62728"]),
        hole=0.4,
    )])
    fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Daily Article Volume")
    if len(daily_df):
        fig_vol = go.Figure(data=[go.Bar(
            x=daily_df["date"], y=daily_df["article_count"],
            marker_color="#1f77b4",
        )])
        fig_vol.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_vol, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Stock-specific news alerts (search)
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
        for _, row in results.iterrows():
            sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(row["sentiment"], "")
            st.markdown(f"{sentiment_emoji} **[{row['title']}]({row['url']})** — {row['date']} · {row['category']}")
    else:
        st.info("No articles found matching that search.")

st.divider()

# ---------------------------------------------------------------------------
# Try it yourself - live prediction
# ---------------------------------------------------------------------------
st.subheader("✍️ Try the Sentiment Model")
st.caption("Paste any NEPSE-related news text to see how the model classifies it")

user_text = st.text_area("News text", placeholder="Paste a headline or article excerpt here...", height=100)

if st.button("Analyze Sentiment"):
    if user_text.strip():
        with st.spinner("Analyzing..."):
            result = predict_new_text(user_text)

        sentiment = result["sentiment"]
        scores = result["confidence_scores"]

        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(sentiment, "")
        st.markdown(f"### {emoji} Predicted Sentiment: **{sentiment.upper()}**")

        score_cols = st.columns(3)
        for i, (label, score) in enumerate(scores.items()):
            with score_cols[i]:
                st.metric(label.capitalize(), f"{score:.1%}")
    else:
        st.warning("Please enter some text first.")

st.divider()

# ---------------------------------------------------------------------------
# Recent articles feed
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
                    summary = summarize_article(row.get("text", ""), row["title"])
                st.info(summary)

   
    else:
        st.info("No articles found matching that search.")
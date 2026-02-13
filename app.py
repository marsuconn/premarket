"""Premarket Pulse Streamlit app.

This app discovers trending stock tickers from social posts, scores sentiment
with VADER, and surfaces a concise market context ("why") for each ticker.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Common non-equity tags often found in social chatter.
BLOCKLISTED_TAGS = {"BTC", "ETH", "AI", "USD", "EUR", "SPX", "QQQ"}

# Regex for cashtags like $NVDA, $TSLA.
CASHTAG_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")


@dataclass
class TickerInsight:
    """Container for per-ticker dashboard values."""

    ticker: str
    mention_count: int
    sentiment_score: float
    market_context: str


def get_mock_tweets(n_posts: int = 100, seed: int = 42) -> List[str]:
    """Generate mock FinTwit-style posts.

    Args:
        n_posts: Number of posts to generate.
        seed: Seed for deterministic output.

    Returns:
        A list of social posts containing cashtags and varied sentiment.
    """
    random.seed(seed)

    tickers = [
        "NVDA",
        "TSLA",
        "AAPL",
        "MSFT",
        "AMZN",
        "META",
        "PLTR",
        "AMD",
        "NFLX",
        "SMCI",
        "COIN",
        "NIO",
    ]
    blocked_noise = ["BTC", "ETH", "AI"]

    bullish_phrases = [
        "massive breakout on volume",
        "guidance raised again",
        "institutions are accumulating",
        "calls are printing hard",
        "earnings beat was huge",
        "momentum looks unstoppable",
        "new product cycle is strong",
    ]
    bearish_phrases = [
        "failed at resistance",
        "guidance cut worries me",
        "distribution all day",
        "puts are loading",
        "margin pressure is obvious",
        "macro headwinds are building",
        "valuation still too stretched",
    ]
    neutral_phrases = [
        "watching price action near key level",
        "waiting for FOMC before adding",
        "keeping position sizing tight",
        "market chop but setup is interesting",
        "tracking options flow this week",
    ]

    catalysts = [
        "ahead of earnings",
        "after analyst upgrade",
        "after analyst downgrade",
        "on AI server demand talk",
        "as yields cool",
        "as yields rise",
        "after delivery numbers",
        "on rumored partnership",
        "after data-center commentary",
    ]

    posts: List[str] = []
    for _ in range(n_posts):
        sentiment_bucket = random.choices(
            ["bull", "bear", "neutral"], weights=[0.45, 0.35, 0.20], k=1
        )[0]

        # Add 1-3 ticker mentions, sometimes repeated to mimic hype threads.
        chosen = random.sample(tickers, k=random.randint(1, 3))
        repeated = random.choice(chosen)
        cashtags = [f"${t}" for t in chosen]

        if random.random() < 0.35:
            cashtags.extend([f"${repeated}"] * random.randint(1, 2))

        # Occasionally inject noisy non-equity tags.
        if random.random() < 0.20:
            cashtags.append(f"${random.choice(blocked_noise)}")

        if sentiment_bucket == "bull":
            phrase = random.choice(bullish_phrases)
        elif sentiment_bucket == "bear":
            phrase = random.choice(bearish_phrases)
        else:
            phrase = random.choice(neutral_phrases)

        catalyst = random.choice(catalysts)
        post = f"{' '.join(cashtags)} {phrase} {catalyst}."
        posts.append(post)

    return posts


def extract_tickers(text: str) -> List[str]:
    """Extract equity-like cashtags from one post, excluding noise tags."""
    return [
        ticker
        for ticker in CASHTAG_PATTERN.findall(text.upper())
        if ticker not in BLOCKLISTED_TAGS
    ]


def get_top_tickers(posts: Sequence[str], top_n: int = 5) -> List[Tuple[str, int]]:
    """Find top mentioned tickers from posts.

    Uses regex cashtag extraction and returns the most frequent tickers.
    """
    counts: Counter = Counter()
    for post in posts:
        counts.update(extract_tickers(post))
    return counts.most_common(top_n)


def sentiment_to_percent(compound: float) -> float:
    """Map VADER compound [-1, 1] to [0, 100] bullishness percent."""
    return round((compound + 1) * 50, 2)


def summarize_why(post: str, ticker: str, word_limit: int = 12) -> str:
    """Produce a concise 12-word market context statement from a viral post."""
    cleaned = re.sub(r"\s+", " ", post.replace("\n", " ")).strip()
    cleaned = re.sub(r"\$[A-Z]{1,5}", "", cleaned).strip()
    words = cleaned.split()

    if not words:
        return f"{ticker} attracting unusual social attention on mixed signals and momentum chatter."

    summary_words = words[:word_limit]
    if len(summary_words) < word_limit:
        filler = ["driving", "attention", "across", "social", "feeds", "today"]
        for token in filler:
            if len(summary_words) >= word_limit:
                break
            summary_words.append(token)

    return " ".join(summary_words[:word_limit])


def build_ticker_insights(posts: Sequence[str], top_n: int = 5) -> List[TickerInsight]:
    """Compute mention counts, sentiment, and market context for top tickers."""
    analyzer = SentimentIntensityAnalyzer()
    top_tickers = get_top_tickers(posts, top_n=top_n)

    ticker_posts: Dict[str, List[str]] = defaultdict(list)
    for post in posts:
        present = set(extract_tickers(post))
        for ticker in present:
            ticker_posts[ticker].append(post)

    insights: List[TickerInsight] = []
    for ticker, mention_count in top_tickers:
        related_posts = ticker_posts.get(ticker, [])

        if related_posts:
            compounds = [analyzer.polarity_scores(p)["compound"] for p in related_posts]
            avg_compound = sum(compounds) / len(compounds)

            # Viral post = post containing highest number of mentions of the ticker keyword.
            viral_post = max(
                related_posts,
                key=lambda p: len(re.findall(fr"\${re.escape(ticker)}\b", p.upper())),
            )
            why = summarize_why(viral_post, ticker=ticker, word_limit=12)
        else:
            avg_compound = 0.0
            why = "Insufficient post context for this ticker in current social sample data."

        insights.append(
            TickerInsight(
                ticker=ticker,
                mention_count=mention_count,
                sentiment_score=sentiment_to_percent(avg_compound),
                market_context=why,
            )
        )

    return insights


def insights_to_dataframe(insights: Iterable[TickerInsight]) -> pd.DataFrame:
    """Convert insight objects to the requested dashboard table format."""
    rows = [
        {
            "Ticker": i.ticker,
            "Mention Count": i.mention_count,
            "Sentiment (0-100%)": i.sentiment_score,
            "Market Context": i.market_context,
        }
        for i in insights
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Streamlit entrypoint."""
    st.set_page_config(page_title="Premarket Pulse", page_icon="📈", layout="wide")
    st.title("📈 Premarket Pulse")
    st.caption("Top trending tickers from social chatter with sentiment and context.")

    n_posts = st.sidebar.slider("Mock posts", min_value=50, max_value=300, value=100, step=10)
    seed = st.sidebar.number_input("Random seed", min_value=1, max_value=9999, value=42)
    top_n = st.sidebar.slider("Top tickers to show", min_value=3, max_value=10, value=5)

    posts = get_mock_tweets(n_posts=n_posts, seed=int(seed))
    insights = build_ticker_insights(posts, top_n=top_n)
    df = insights_to_dataframe(insights)

    st.subheader("Trending Tickers")
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Sample raw posts"):
        for p in posts[:10]:
            st.write(f"- {p}")


if __name__ == "__main__":
    main()

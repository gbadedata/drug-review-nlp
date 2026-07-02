"""Load and clean the UCI Drug Review dataset.

The raw files have three quirks worth handling explicitly, because they change the
numbers if ignored:
  - ~900 rows carry a scraping artifact in `condition` ("N</span> users found this
    comment helpful.") and ~900 more have no condition at all;
  - review text is HTML-escaped (&amp;, &#039;) and wrapped in stray quotes;
  - ratings are on 1-10 but cluster hard at the extremes, so sentiment is derived with
    an explicit neutral band that is dropped rather than guessed.

`load_reviews` returns a cleaned DataFrame with a binary `sentiment` label and a set of
structured features derived alongside the text.
"""
from __future__ import annotations

import html
import re

import pandas as pd

RAW_COLUMNS = ["Unnamed: 0", "drugName", "condition", "review", "rating", "date", "usefulCount"]
_JUNK_CONDITION = re.compile(r"</span>|users found this comment helpful", re.I)


def _clean_review(text: str) -> str:
    text = html.unescape(str(text))
    text = text.strip().strip('"').strip()
    return re.sub(r"\s+", " ", text)


def load_reviews(path: str, *, pos_threshold: int = 7, neg_threshold: int = 4,
                 drop_neutral: bool = True) -> pd.DataFrame:
    sep = "," if path.lower().endswith(".csv") else "\t"
    df = pd.read_csv(path, sep=sep)
    df = df.rename(columns={"Unnamed: 0": "review_id"})
    if "review_id" not in df.columns:
        df.insert(0, "review_id", range(len(df)))

    # drop rows with missing or artifact conditions
    df = df[df["condition"].notna()].copy()
    df = df[~df["condition"].astype(str).str.contains(_JUNK_CONDITION)]

    df["review"] = df["review"].map(_clean_review)
    df = df[df["review"].str.len() > 0]

    # structured features derived before we filter on sentiment
    df["review_len"] = df["review"].str.len()
    df["word_count"] = df["review"].str.split().map(len)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["condition_freq"] = df.groupby("condition")["review_id"].transform("count")
    df["drug_freq"] = df.groupby("drugName")["review_id"].transform("count")

    # binary sentiment with an explicit, dropped neutral band
    def label(r: float) -> str | float:
        if r >= pos_threshold:
            return "positive"
        if r <= neg_threshold:
            return "negative"
        return "neutral"

    df["sentiment"] = df["rating"].map(label)
    if drop_neutral:
        df = df[df["sentiment"] != "neutral"].copy()

    return df.reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "n_reviews": int(len(df)),
        "n_conditions": int(df["condition"].nunique()),
        "n_drugs": int(df["drugName"].nunique()),
        "sentiment_distribution": df["sentiment"].value_counts().to_dict(),
        "positive_base_rate": round(float((df["sentiment"] == "positive").mean()), 4),
        "median_word_count": int(df["word_count"].median()),
        "mean_rating": round(float(df["rating"].mean()), 3),
    }

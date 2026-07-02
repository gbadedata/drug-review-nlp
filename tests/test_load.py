import os

import pandas as pd

from drugnlp.load import _clean_review, load_reviews
from drugnlp.features import StructuredFeaturizer, TextFeaturizer, combine

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_reviews.csv")


def test_clean_review_unescapes_and_trims():
    assert _clean_review('"it&#039;s   fine &amp; safe"') == "it's fine & safe"


def test_sample_loads_with_binary_sentiment():
    df = load_reviews(SAMPLE)
    assert len(df) > 0
    assert set(df["sentiment"].unique()) <= {"positive", "negative"}
    # derived structured columns are present
    for col in ("review_len", "word_count", "year", "condition_freq", "drug_freq"):
        assert col in df.columns


def test_neutral_band_is_dropped():
    df = load_reviews(SAMPLE)
    assert not ((df["rating"] >= 5) & (df["rating"] <= 6)).any()


def test_feature_matrices_align():
    df = load_reviews(SAMPLE)
    text = TextFeaturizer(max_features=500, min_df=1).fit_transform(df["review"])
    struct = StructuredFeaturizer().fit_transform(df)
    both = combine(text, struct)
    assert text.shape[0] == struct.shape[0] == both.shape[0] == len(df)
    assert both.shape[1] == text.shape[1] + struct.shape[1]

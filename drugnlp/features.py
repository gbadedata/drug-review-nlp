"""Feature builders for the review models.

Three views of each review, so the contribution of each can be isolated in the
evaluation: the text (TF-IDF over 1-2 grams), a small block of structured signals
(length, usefulness votes, how common the drug/condition is, the year), and the two
stacked together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

STRUCTURED_COLS = ["review_len", "word_count", "usefulCount",
                   "condition_freq", "drug_freq", "year"]


class TextFeaturizer:
    def __init__(self, max_features: int = 50000, ngram_range=(1, 2), min_df: int = 5):
        self.vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
                                   min_df=min_df, sublinear_tf=True, stop_words="english")

    def fit_transform(self, texts):
        return self.vec.fit_transform(texts)

    def transform(self, texts):
        return self.vec.transform(texts)


class StructuredFeaturizer:
    def __init__(self, cols=STRUCTURED_COLS):
        self.cols = list(cols)
        self.scaler = StandardScaler()

    def _frame(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols].copy()
        x["year"] = x["year"].fillna(x["year"].median())
        return x.to_numpy(dtype="float64")

    def fit_transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.csr_matrix(self.scaler.fit_transform(self._frame(df)))

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        return sparse.csr_matrix(self.scaler.transform(self._frame(df)))


def combine(text_mat, struct_mat) -> sparse.csr_matrix:
    return sparse.hstack([text_mat, struct_mat]).tocsr()

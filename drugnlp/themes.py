"""What do patients complain about, per condition?

For each high-volume condition, this ranks the terms that most distinguish its negative
reviews from negative reviews of every other condition, using the weighted log-odds
ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008). That method
is used instead of raw frequency or plain TF-IDF because it corrects for how common a
word is overall and for the sampling variance of rare words, so the terms it surfaces
are genuinely characteristic rather than just frequent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def _log_odds(count_i: np.ndarray, count_j: np.ndarray, alpha0: float = 1000.0):
    total = count_i + count_j
    n = float(total.sum())
    alpha = alpha0 * (total / n)                       # informative prior from background freq
    ni, nj, a0 = count_i.sum(), count_j.sum(), alpha.sum()

    def logit(y, ntot):
        return np.log((y + alpha) / (ntot + a0 - y - alpha))

    delta = logit(count_i, ni) - logit(count_j, nj)
    var = 1.0 / (count_i + alpha) + 1.0 / (count_j + alpha)
    return delta / np.sqrt(var)


def distinctive_negative_terms(df: pd.DataFrame, *, top_conditions: int = 8,
                               terms_per_condition: int = 12, min_df: int = 20) -> dict:
    """Return {condition: [distinctive negative-review terms]} for the busiest conditions."""
    neg = df[df["sentiment"] == "negative"]
    top = neg["condition"].value_counts().head(top_conditions).index.tolist()

    vec = CountVectorizer(ngram_range=(1, 2), min_df=min_df, stop_words="english")
    X = vec.fit_transform(neg["review"])
    vocab = np.array(vec.get_feature_names_out())
    cond = neg["condition"].to_numpy()

    out: dict[str, list[str]] = {}
    for c in top:
        mask = cond == c
        count_i = np.asarray(X[mask].sum(axis=0)).ravel()
        count_j = np.asarray(X[~mask].sum(axis=0)).ravel()
        z = _log_odds(count_i, count_j)
        order = np.argsort(z)[::-1][:terms_per_condition]
        out[c] = vocab[order].tolist()
    return out


def satisfaction_by_condition(df: pd.DataFrame, *, top_n: int = 12,
                              min_reviews: int = 300) -> pd.DataFrame:
    """Mean rating and negative share for the most-reviewed conditions."""
    g = df.groupby("condition")
    tab = pd.DataFrame({
        "n_reviews": g.size(),
        "mean_rating": g["rating"].mean().round(2),
        "negative_share": (g["sentiment"].apply(lambda s: (s == "negative").mean())).round(3),
    })
    tab = tab[tab["n_reviews"] >= min_reviews]
    return tab.sort_values("mean_rating").head(top_n)

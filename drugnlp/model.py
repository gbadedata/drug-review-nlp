"""Train and evaluate the review-sentiment models.

Uses the dataset's own train/test split (not a re-shuffle), so the evaluation is on
genuinely held-out reviews. Three feature views (text, structured, combined) are each
scored against the majority-class baseline on accuracy, macro-F1 and ROC-AUC, so the
lift from the text and from the structured block can be read off directly.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .features import StructuredFeaturizer, TextFeaturizer, combine

POS = "positive"


def _y(df: pd.DataFrame) -> np.ndarray:
    return (df["sentiment"] == POS).astype(int).to_numpy()


def _score(y_true, pred, proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "macro_f1": round(float(f1_score(y_true, pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }


def _fit_eval(xtr, ytr, xte, yte) -> dict:
    clf = LogisticRegression(max_iter=1000, C=4.0, solver="liblinear")
    clf.fit(xtr, ytr)
    proba = clf.predict_proba(xte)[:, 1]
    return _score(yte, clf.predict(xte), proba)


def run(train_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str = "results") -> dict:
    ytr, yte = _y(train_df), _y(test_df)

    # majority-class baseline
    major = int(round(ytr.mean()))
    maj_pred = np.full_like(yte, major)
    baseline = {
        "accuracy": round(float(accuracy_score(yte, maj_pred)), 4),
        "macro_f1": round(float(f1_score(yte, maj_pred, average="macro")), 4),
        "roc_auc": 0.5,
        "predicts": POS if major == 1 else "negative",
    }

    text = TextFeaturizer()
    xtr_t = text.fit_transform(train_df["review"])
    xte_t = text.transform(test_df["review"])

    struct = StructuredFeaturizer()
    xtr_s = struct.fit_transform(train_df)
    xte_s = struct.transform(test_df)

    results = {
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "positive_base_rate_test": round(float(yte.mean()), 4),
        "majority_baseline": baseline,
        "text_only": _fit_eval(xtr_t, ytr, xte_t, yte),
        "structured_only": _fit_eval(xtr_s, ytr, xte_s, yte),
        "text_plus_structured": _fit_eval(combine(xtr_t, xtr_s), ytr,
                                          combine(xte_t, xte_s), yte),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sentiment_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    return results

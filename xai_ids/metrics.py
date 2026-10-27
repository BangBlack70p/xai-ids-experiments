"""Threshold selection, detection metrics and statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


# ---------------------------------------------------------------- threshold
def select_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    """theta* = argmax_theta F1(theta; validation partition)."""
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.nanargmax(f1))])


# ---------------------------------------------------------------- metrics
def detection_metrics(y: np.ndarray, score: np.ndarray, theta: float) -> dict:
    pred = (score >= theta).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    out = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "far": fp / max(1, fp + tn),
        "accuracy": float((pred == y).mean()),
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "n": int(len(y)), "positives": int(y.sum()), "theta": theta,
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, score)
        out["pr_auc"] = average_precision_score(y, score)
    else:
        out["roc_auc"] = out["pr_auc"] = np.nan
    return out


def per_class_recall(cls: pd.Series, y: np.ndarray, score: np.ndarray, theta: float) -> pd.Series:
    pred = score >= theta
    df = pd.DataFrame({"cls": cls.values, "y": y, "pred": pred})
    att = df[df.y == 1].groupby("cls")["pred"].agg(["mean", "size"])
    ben = df[df.y == 0]["pred"]
    att.loc["(benign specificity)"] = [1 - ben.mean() if len(ben) else np.nan, len(ben)]
    return att.rename(columns={"mean": "recall", "size": "n"})


# ---------------------------------------------------------------- tests
def bootstrap_ci(y, score, theta, metric="f1", n=1000, seed=0, alpha=0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0:
            continue
        vals.append(detection_metrics(y[b], score[b], theta)[metric])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def mcnemar(y, pred_a, pred_b) -> dict:
    """Exact McNemar test on paired correctness of two classifiers."""
    a_ok, b_ok = pred_a == y, pred_b == y
    n01 = int((a_ok & ~b_ok).sum())
    n10 = int((~a_ok & b_ok).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


def _midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray) -> dict:
    """DeLong test for two correlated ROC AUCs (Sun & Xu fast algorithm)."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    preds = np.vstack([s_a, s_b])
    aucs, v01, v10 = [], [], []
    for s in preds:
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.concatenate([s[pos], s[neg]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v01.append((tz[:m] - tx) / n)
        v10.append(1 - (tz[m:] - ty) / m)
    v01, v10 = np.array(v01), np.array(v10)
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-18))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p_value": float(2 * stats.norm.sf(abs(z)))}

"""Deduplication, partitioning and leakage-safe preprocessing.

Leakage rules implemented here (Section 4.2 of the paper):
1. duplicates removed BEFORE partitioning; no feature vector in two partitions;
2. identifiers/timestamps never enter X (handled in data.py);
3. partition first, then fit imputer/encoder/scaler on the training part only;
4. resampling (if any) on the training part only (see models.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import DAY_COL, TIME_COL, FlowData


def row_hash(X: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(X, index=False)


def deduplicate(data: FlowData) -> tuple[FlowData, dict]:
    """Drop vectors that carry conflicting labels, then keep one copy of each vector."""
    h = row_hash(data.X)
    n0 = len(h)
    labels_per_hash = data.y.groupby(h.values).nunique()
    conflicting = set(labels_per_hash[labels_per_hash > 1].index)
    keep = ~h.isin(conflicting)
    keep &= ~h.duplicated(keep="first")
    out = data.subset(data.y.index[keep.values])
    stats = {"rows_before": n0, "conflicting_vectors": len(conflicting),
             "rows_after": len(out)}
    return out, stats


def chronological_split(data: FlowData, fractions: dict, by_class: bool = True) -> dict[str, FlowData]:
    """Cut each group in time order: first `train`, next `val`, last `test`.

    Groups are (day, class) by default. CIC-IDS2017 attacks run in bursts of an
    hour or so; cutting a whole day by time would put an afternoon attack
    entirely in the test partition. Cutting each (day, class) stream keeps time
    order inside every attack episode and places every class in all partitions.
    ``by_class=False`` cuts whole days (stricter, used as a robustness check).
    """
    parts = {"train": [], "val": [], "test": []}
    f_tr, f_va = fractions["train"], fractions["val"]
    keys = [data.meta[DAY_COL], data.cls] if by_class else [data.meta[DAY_COL]]
    for _, grp in data.meta.groupby(keys, sort=True):
        order = grp.sort_values(TIME_COL, kind="stable").index if grp[TIME_COL].notna().any() \
            else grp.index
        n = len(order)
        a, b = int(round(n * f_tr)), int(round(n * (f_tr + f_va)))
        parts["train"].extend(order[:a])
        parts["val"].extend(order[a:b])
        parts["test"].extend(order[b:])
    return {k: data.subset(pd.Index(v)) for k, v in parts.items()}


def random_split(data: FlowData, fractions: dict, seed: int) -> dict[str, FlowData]:
    """Conventional stratified random split (used only as the E1 contrast)."""
    idx = data.y.index
    test_size = fractions["test"]
    tr_va, te = train_test_split(idx, test_size=test_size, stratify=data.y, random_state=seed)
    val_rel = fractions["val"] / (fractions["train"] + fractions["val"])
    tr, va = train_test_split(tr_va, test_size=val_rel, stratify=data.y.loc[tr_va],
                              random_state=seed)
    return {"train": data.subset(tr), "val": data.subset(va), "test": data.subset(te)}


def leave_day_out(data: FlowData, day: str, val_frac: float = 0.2) -> dict[str, FlowData]:
    """Test = one whole day; train/val = the other days, cut chronologically per day."""
    test_idx = data.meta.index[data.meta[DAY_COL] == day]
    rest = data.subset(data.meta.index[data.meta[DAY_COL] != day])
    parts = chronological_split(rest, {"train": 1 - val_frac, "val": val_frac, "test": 0.0})
    return {"train": parts["train"], "val": parts["val"], "test": data.subset(test_idx)}


def leakage_report(parts: dict[str, FlowData]) -> dict:
    """Count feature vectors shared between partitions (should be 0)."""
    hashes = {k: set(row_hash(v.X).values) for k, v in parts.items()}
    report = {}
    keys = list(parts)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            report[f"overlap_{a}_{b}"] = len(hashes[a] & hashes[b])
    return report


def class_counts(parts: dict[str, FlowData]) -> pd.DataFrame:
    """Per-class counts per partition (reported in the paper)."""
    return pd.DataFrame({k: v.cls.value_counts() for k, v in parts.items()}).fillna(0).astype(int)


class Preprocessor:
    """Imputer (+ one-hot for categoricals, + optional scaler), fitted on train only."""

    def __init__(self, categorical: list[str], scale: bool = False):
        self.categorical = categorical
        self.scale = scale
        self.ct: ColumnTransformer | None = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        num = [c for c in X.columns if c not in self.categorical]
        num_steps = [("impute", SimpleImputer(strategy="median"))]
        if self.scale:
            num_steps.append(("scale", StandardScaler()))
        transformers = [("num", Pipeline(num_steps), num)]
        if self.categorical:
            transformers.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=0.001, sparse_output=False)),
            ]), self.categorical))
        self.ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
        self.ct.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.ct.transform(X), dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        return list(self.ct.get_feature_names_out())

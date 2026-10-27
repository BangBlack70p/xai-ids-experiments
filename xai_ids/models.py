"""Detector factory.

Every detector exposes ``fit(X, y)`` and ``score(X)``; a higher score means
"more likely a threat". Supervised models return P(Y = Threat | x).
Unsupervised models are trained on benign training flows only and return an
anomaly score, so the same validation-selected threshold logic applies.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

SUPERVISED = ["lr", "dt", "rf", "xgb"]
UNSUPERVISED = ["iforest", "autoencoder"]
NEEDS_SCALING = {"lr", "autoencoder"}


def _resample(X, y, method: str | None, seed: int):
    """Optional resampling, applied to the TRAINING partition only."""
    if not method:
        return X, y
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install imbalanced-learn to use resampling") from exc
    sampler = SMOTETomek(random_state=seed) if method == "smote_tomek" else SMOTE(random_state=seed)
    return sampler.fit_resample(X, y)


class Detector:
    def __init__(self, name: str, seed: int = 0, params: dict | None = None,
                 resample: str | None = None):
        self.name, self.seed, self.params, self.resample = name, seed, params or {}, resample
        self.model = None
        self._ae_scale = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Detector":
        p, s = self.params, self.seed
        if self.name in SUPERVISED:
            X, y = _resample(X, y, self.resample, s)
        if self.name == "rf":
            self.model = RandomForestClassifier(random_state=s, **p)
        elif self.name == "dt":
            self.model = DecisionTreeClassifier(random_state=s, class_weight="balanced", **p)
        elif self.name == "lr":
            self.model = LogisticRegression(max_iter=2000, class_weight="balanced", **p)
        elif self.name == "xgb":
            from xgboost import XGBClassifier
            pos = max(1, int((y == 1).sum()))
            self.model = XGBClassifier(n_estimators=p.get("n_estimators", 300),
                                       max_depth=p.get("max_depth", 8),
                                       learning_rate=p.get("learning_rate", 0.1),
                                       subsample=0.8, colsample_bytree=0.8,
                                       scale_pos_weight=((y == 0).sum() / pos),
                                       tree_method="hist", n_jobs=-1, random_state=s,
                                       eval_metric="logloss")
        elif self.name == "iforest":
            self.model = IsolationForest(n_estimators=p.get("n_estimators", 200),
                                         random_state=s, n_jobs=-1)
            X = X[y == 0]
        elif self.name == "autoencoder":
            X = X[y == 0]
            if len(X) > p.get("max_rows", 200_000):
                X = X[np.random.default_rng(s).choice(len(X), p.get("max_rows", 200_000),
                                                      replace=False)]
            d = X.shape[1]
            hidden = p.get("hidden", (max(8, d // 2), max(4, d // 4), max(8, d // 2)))
            self.model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=p.get("max_iter", 50),
                                      early_stopping=True, random_state=s)
        else:
            raise ValueError(f"Unknown model {self.name}")
        if self.name == "autoencoder":
            self.model.fit(X, X)
        elif self.name == "iforest":
            self.model.fit(X)
        else:
            self.model.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.name in SUPERVISED:
            return self.model.predict_proba(X)[:, 1]
        if self.name == "iforest":
            return -self.model.score_samples(X)
        recon = self.model.predict(X)
        return np.mean((X - recon) ** 2, axis=1)

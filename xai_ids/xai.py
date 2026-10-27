"""Explanation layer and its quantitative evaluation (E5).

* TreeSHAP (exact, tree-path-dependent) for local and global attributions
* LIME tabular surrogate for local attributions
* impurity and permutation importance as global baselines
* agreement   : Kendall tau-b, top-k overlap, sign agreement
* faithfulness: deletion curves (top-k features replaced by training medians)
* stability   : LIME repeated with different seeds
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.inspection import permutation_importance


# ------------------------------------------------------------ attributions
def tree_shap(model, X: np.ndarray) -> np.ndarray:
    """SHAP values for the threat class, shape (n, d)."""
    import shap
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    vals = explainer.shap_values(X, check_additivity=False)
    if isinstance(vals, list):           # older shap: one array per class
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:                   # newer shap: (n, d, classes)
        vals = vals[:, :, 1]
    return vals


def lime_weights(predict_proba, X_ref: np.ndarray, X: np.ndarray, feature_names: list[str],
                 num_samples: int = 2000, seed: int = 0) -> np.ndarray:
    """LIME weights for the threat class for every row of X, shape (n, d)."""
    from lime.lime_tabular import LimeTabularExplainer
    d = X.shape[1]
    explainer = LimeTabularExplainer(X_ref, feature_names=feature_names, mode="classification",
                                     discretize_continuous=False, random_state=seed)
    out = np.zeros((len(X), d))
    for i, x in enumerate(X):
        exp = explainer.explain_instance(x, predict_proba, labels=(1,), num_features=d,
                                         num_samples=num_samples)
        for j, w in exp.as_map()[1]:
            out[i, j] = w
    return out


def global_importances(model, X_val: np.ndarray, y_val: np.ndarray, shap_vals: np.ndarray,
                       feature_names: list[str], n_perm: int = 50000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_val), min(n_perm, len(X_val)), replace=False)
    perm = permutation_importance(model, X_val[idx], y_val[idx], scoring="average_precision",
                                  n_repeats=3, random_state=seed, n_jobs=-1)
    return pd.DataFrame({
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        "impurity": model.feature_importances_,
        "permutation": perm.importances_mean,
    }, index=feature_names)


# ------------------------------------------------------------ agreement
def top_k(v: np.ndarray, k: int) -> set[int]:
    return set(np.argsort(-np.abs(v))[:k])


def rank_agreement(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    tau = kendalltau(np.abs(a), np.abs(b)).statistic
    ta, tb = top_k(a, k), top_k(b, k)
    shared = list(ta & tb)
    sign = float(np.mean(np.sign(a[shared]) == np.sign(b[shared]))) if shared else np.nan
    return {"kendall_tau": float(tau) if tau == tau else np.nan,
            "topk_overlap": len(ta & tb) / k, "sign_agreement": sign}


def global_agreement(imp: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(imp.columns, 2):
        r = rank_agreement(imp[a].to_numpy(), imp[b].to_numpy(), k)
        rows.append({"method_a": a, "method_b": b, "kendall_tau": r["kendall_tau"],
                     "topk_overlap": r["topk_overlap"]})
    return pd.DataFrame(rows)


def local_agreement(A: np.ndarray, B: np.ndarray, k: int) -> pd.DataFrame:
    return pd.DataFrame([rank_agreement(a, b, k) for a, b in zip(A, B)])


# ------------------------------------------------------------ faithfulness
def deletion_curve(predict_proba, X: np.ndarray, attributions: np.ndarray | None,
                   background: np.ndarray, max_k: int, seed: int = 0,
                   n_random: int = 5) -> np.ndarray:
    """Mean P(threat) after replacing the top-k features (k = 0..max_k).

    ``attributions=None`` gives the random-order baseline (averaged over n_random orders).
    """
    rng = np.random.default_rng(seed)
    n, d = X.shape
    orders = []
    if attributions is None:
        for _ in range(n_random):
            orders.append(np.array([rng.permutation(d) for _ in range(n)]))
    else:
        # Only features that push towards "threat" are removed first.
        orders.append(np.argsort(-attributions, axis=1))
    curves = []
    for order in orders:
        curve = []
        Xk = X.copy()
        curve.append(predict_proba(Xk)[:, 1].mean())
        for k in range(max_k):
            cols = order[:, k]
            Xk[np.arange(n), cols] = background[cols]
            curve.append(predict_proba(Xk)[:, 1].mean())
        curves.append(curve)
    return np.mean(curves, axis=0)


def curve_area(curve: np.ndarray) -> float:
    """Normalised area under a deletion curve (lower = more faithful)."""
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2 compatibility
    return float(trapz(curve, dx=1.0) / (len(curve) - 1))


# ------------------------------------------------------------ stability
def lime_stability(predict_proba, X_ref, X, feature_names, runs, num_samples, k) -> pd.DataFrame:
    per_run = [lime_weights(predict_proba, X_ref, X, feature_names, num_samples, seed=r)
               for r in range(runs)]
    rows = []
    for i in range(len(X)):
        taus, overlaps = [], []
        for a, b in itertools.combinations(range(runs), 2):
            r = rank_agreement(per_run[a][i], per_run[b][i], k)
            taus.append(r["kendall_tau"])
            overlaps.append(r["topk_overlap"])
        rows.append({"mean_pairwise_tau": np.nanmean(taus), "mean_topk_overlap": np.mean(overlaps)})
    return pd.DataFrame(rows)

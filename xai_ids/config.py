"""Configuration loading.

All experiment settings live in one YAML file (see configs/example.yaml).
Paths inside the file are resolved relative to the YAML file's folder.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "output_dir": "results",
    "seeds": [0, 1, 2, 3, 4],
    "shared_features": None,  # None = intersection of columns across datasets
    "split": {"train": 0.6, "val": 0.2, "test": 0.2},
    "model": {
        "rf": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1,
               "class_weight": "balanced", "n_jobs": -1},
    },
    "xai": {
        "n_alerts": 500,          # local explanations (TreeSHAP vs LIME)
        "top_k": 5,
        "deletion_max_k": 10,
        "lime_num_samples": 2000,
        "lime_stability_alerts": 50,
        "lime_stability_runs": 10,
        "permutation_sample": 50000,
        "shap_background": 200,
    },
    "bootstrap": {"n": 1000},
    "datasets": {},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _merge(DEFAULTS, raw)
    base = path.parent
    cfg["output_dir"] = str((base / cfg["output_dir"]).resolve())
    for ds in cfg["datasets"].values():
        for key in ("path", "train_path", "test_path", "noc_pseudonyms"):
            if ds.get(key):
                ds[key] = str((base / ds[key]).resolve())
    return cfg

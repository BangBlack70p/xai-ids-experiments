"""Dataset loading.

Every dataset is turned into one ``FlowData`` object:

* ``X``      - numeric/categorical flow features only (no identifiers, no time)
* ``y``      - binary target, 1 = threat, 0 = benign
* ``cls``    - original class name (for per-class recall)
* ``meta``   - identifiers and timestamp, kept OUT of X (used for splitting and
               for entity-level analysis in later modules)

The loader is generic so that CIC-IDS2017, UNSW-NB15 and PTMA re-extracted with
the same flow extractor (CICFlowMeter / NFStream) share one code path.
Column names are normalised (lower case, non-alphanumerics -> "_") so that
"Flow Duration", "flow_duration" and "Flow_Duration" all match.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TIME_COL = "_timestamp"
DAY_COL = "_day"


def norm(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return name.strip("_")


@dataclass
class FlowData:
    name: str
    X: pd.DataFrame
    y: pd.Series
    cls: pd.Series
    meta: pd.DataFrame
    categorical: list[str] = field(default_factory=list)

    def subset(self, idx) -> "FlowData":
        return FlowData(self.name, self.X.loc[idx], self.y.loc[idx], self.cls.loc[idx],
                        self.meta.loc[idx], list(self.categorical))

    def restrict_features(self, features: list[str]) -> "FlowData":
        cats = [c for c in self.categorical if c in features]
        return FlowData(self.name, self.X[features], self.y, self.cls, self.meta, cats)

    def __len__(self) -> int:
        return len(self.y)


def _protected(cfg: dict) -> set[str]:
    """Columns that must stay as text (labels, time, identifiers, categoricals)."""
    keys = [cfg.get("label_col", "label"), cfg.get("class_col"), cfg.get("timestamp_col")]
    keys += cfg.get("meta_cols", []) + cfg.get("categorical_cols", [])
    return {norm(k) for k in keys if k}


def _members(path: Path):
    """Yield (name, opener) for every CSV/Parquet in a file, folder or ZIP archive."""
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in {".zip", ".csv", ".parquet"}:
                yield from _members(p)
    elif path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.filename.lower().endswith((".csv", ".parquet")) and not info.is_dir():
                yield Path(info.filename).name, (lambda i=info: zf.open(i))
    else:
        yield path.name, (lambda: open(path, "rb"))


def _read_any(path: str | Path, cfg: dict | None = None) -> pd.DataFrame:
    """Read a CSV/Parquet file, a folder of them, or a ZIP archive of them.

    Each file is reduced before concatenation: column names normalised,
    ``drop_cols`` removed, and every non-protected column stored as float32 with
    +/-inf turned into NaN. This keeps CIC-IDS2017 (2-3 million flows) within a
    laptop's memory.
    """
    cfg = cfg or {}
    keep_text = _protected(cfg)
    drop = {norm(c) for c in cfg.get("drop_cols", [])} - keep_text
    encoding = cfg.get("encoding", "utf-8")
    frames = []
    for name, opener in _members(Path(path)):
        with opener() as fh:
            if name.lower().endswith(".parquet"):
                df = pd.read_parquet(fh)
            else:
                df = pd.read_csv(fh, low_memory=False, encoding=encoding, encoding_errors="replace")
        df.columns = [norm(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.drop(columns=[c for c in df.columns if c in drop])
        for c in df.columns:
            if c not in keep_text:
                col = pd.to_numeric(df[c], errors="coerce").astype("float32")
                col[np.isinf(col)] = np.nan
                df[c] = col
        df["_source_file"] = name
        frames.append(df)
        print(f"    read {name}: {len(df):,} rows")
    if not frames:
        raise FileNotFoundError(f"No CSV/Parquet files in {path}")
    out = pd.concat(frames, ignore_index=True)
    del frames
    return out


def _binary_target(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (y, class name, keep-mask)."""
    label_col = norm(cfg.get("label_col", "label"))
    class_col = norm(cfg["class_col"]) if cfg.get("class_col") else label_col
    # Some CIC-IDS2017 CSVs carry a broken en dash ("Web Attack \ufffd Brute Force").
    cls = (df[class_col].astype(str).str.replace("\ufffd", "-", regex=False)
           .str.replace("\u00ef\u00bf\u00bd", "-", regex=False).str.replace("\u2013", "-", regex=False)
           .str.strip())
    keep = pd.Series(True, index=df.index)

    if cfg.get("label_is_binary", False):
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        cls = cls.where(y == 1, "benign")
    else:
        benign = {s.lower() for s in cfg.get("benign_labels", ["benign", "normal"])}
        y = (~cls.str.lower().isin(benign)).astype(int)

    # Engelen et al. (2021) mark failed attack attempts as "... - Attempted".
    pattern = cfg.get("attempted_pattern")
    if pattern:
        attempted = cls.str.contains(pattern, case=False, regex=True)
        policy = cfg.get("attempted_policy", "drop")
        if policy == "drop":
            keep &= ~attempted
        elif policy == "benign":
            y = y.where(~attempted, 0)
        elif policy != "attack":
            raise ValueError("attempted_policy must be drop, benign or attack")
    return y, cls, keep


def load_dataset(name: str, cfg: dict, path_key: str = "path") -> FlowData:
    """Load one dataset described by its config block."""
    df = _read_any(cfg[path_key], cfg)

    y, cls, keep = _binary_target(df, cfg)

    # ---- timestamp / day (metadata only) -----------------------------------
    meta = pd.DataFrame(index=df.index)
    ts_col = norm(cfg["timestamp_col"]) if cfg.get("timestamp_col") else None
    if ts_col and ts_col in df:
        fmt = cfg.get("timestamp_format")
        if fmt:
            ts = pd.to_datetime(df[ts_col], format=fmt, errors="coerce")
        else:
            ts = pd.to_datetime(df[ts_col], dayfirst=cfg.get("dayfirst", False),
                                errors="coerce", format="mixed")
        meta[TIME_COL] = ts
        meta[DAY_COL] = ts.dt.date.astype(str)
        keep &= ts.notna()
    else:
        meta[TIME_COL] = pd.NaT
        meta[DAY_COL] = df.get("_source_file", "all")
    for col in cfg.get("meta_cols", []):
        if norm(col) in df:
            meta[norm(col)] = df[norm(col)]
    meta["_source_file"] = df["_source_file"]

    # ---- features -----------------------------------------------------------
    drop = {norm(c) for c in cfg.get("drop_cols", [])}
    drop |= {norm(cfg.get("label_col", "label")), "_source_file"}
    if cfg.get("class_col"):
        drop.add(norm(cfg["class_col"]))
    if ts_col:
        drop.add(ts_col)
    drop |= {norm(c) for c in cfg.get("meta_cols", [])}
    categorical = [norm(c) for c in cfg.get("categorical_cols", []) if norm(c) in df]

    feats = [c for c in df.columns if c not in drop]
    for c in categorical:
        df[c] = df[c].astype(str)
    X = df[feats]
    # Constant columns carry no information and break rank statistics.
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    categorical = [c for c in categorical if c in X]

    data = FlowData(name, X, y, cls, meta, categorical).subset(df.index[keep])

    frac = cfg.get("sample_frac")
    if frac and frac < 1:
        # Uniform sample that keeps every class present and keeps time order.
        rng = np.random.default_rng(cfg.get("sample_seed", 0))
        mask = rng.random(len(data)) < frac
        mask |= data.y.to_numpy() == 1 if cfg.get("keep_all_attacks", False) else False
        data = data.subset(data.y.index[mask])
    return data


def shared_feature_list(datasets: list[FlowData], explicit: list[str] | None = None) -> list[str]:
    """Features present in every dataset (the shared space used for transfer)."""
    if explicit:
        feats = [norm(f) for f in explicit]
        for d in datasets:
            missing = [f for f in feats if f not in d.X]
            if missing:
                raise KeyError(f"{d.name} lacks shared features: {missing[:10]}")
        return feats
    common = set(datasets[0].X.columns)
    for d in datasets[1:]:
        common &= set(d.X.columns)
    return [c for c in datasets[0].X.columns if c in common]

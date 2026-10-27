"""Sanity checks for a folder of (labelled) CICFlowMeter CSVs.

Reports rows, label counts, timestamp range and parse rate, and whether the
header matches the label-corrected CIC-IDS2017 release (needed for transfer).

    python scripts/check_flows.py data/ptma_flows_labelled
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_ZIP = ROOT / "raw" / "dataset.zip"
FMT = "%d/%m/%Y %I:%M:%S %p"


def reference_header() -> list[str] | None:
    if not REF_ZIP.exists():
        return None
    with zipfile.ZipFile(REF_ZIP) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            return [c.strip() for c in fh.readline().decode("utf-8", "replace").split(",")]


def main(folder: str) -> None:
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        sys.exit(f"no CSV files in {folder}")
    ref = reference_header()
    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, usecols=lambda c: c.strip().lower() in {"timestamp", "label"})
        df.columns = [c.strip().lower() for c in df.columns]
        df["file"] = f.name
        frames.append(df)
        header = [c.strip() for c in pd.read_csv(f, nrows=0).columns]
        if ref is not None:
            extra = [c for c in header if c not in ref and not c.endswith("_internal")]
            # Extra columns from the newer CICFlowMeter are fine: transfer uses the shared columns.
            missing = [c for c in ref if c not in header]
            if missing:
                status = f"MISSING CIC-IDS2017 columns {missing[:5]} - transfer would lose features"
            elif extra:
                status = f"all CIC-IDS2017 columns present (+{len(extra)} extra, ignored in transfer)"
            else:
                status = "same as CIC-IDS2017"
            print(f"{f.name}: header {status}")
    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], format=FMT, errors="coerce")
    if ts.notna().mean() < 0.5:   # current CICFlowMeter writes ISO 8601 in UTC
        ts = pd.to_datetime(df["timestamp"], format="ISO8601", errors="coerce")
    print(f"\nrows {len(df):,}; timestamp parsed {ts.notna().mean():.1%}; range {ts.min()} -> {ts.max()} (UTC)")
    print("labels:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")

"""Attach labels to CICFlowMeter flows from an attack schedule or a ground-truth file.

Used for
* UNSW-NB15: the official ground-truth file NUSW-NB15_GT.csv (one row per attack);
* PTMA: the schedule produced from Suricata alerts (scripts/suricata_to_schedule.py),
  or a hand-written schedule of controlled attacks.

A flow receives a schedule row's label when every key column the row specifies
matches (empty cell = wildcard) and the flow's start time lies inside
[start - tolerance, end + tolerance]. With --bidirectional the row also matches
the reversed direction (src <-> dst), because CICFlowMeter names the initiator
"src" while an alert or ground-truth row may describe either direction.
Unmatched flows get --default-label. Labels therefore come from the schedule,
never from flow features.

Examples
--------
UNSW-NB15 (ground truth, Unix seconds, categories cleaned):
    python scripts/label_flows.py --flows data/unsw_flows --schedule raw/NUSW-NB15_GT.csv \
        --gt-unsw --out data/unsw_flows_labelled

PTMA (schedule from Suricata):
    python scripts/label_flows.py --flows data/ptma_flows --schedule data/ptma_suricata_schedule.csv \
        --bidirectional --out data/ptma_flows_labelled
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xai_ids.data import norm  # noqa: E402

KEYS = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
PROTO = {"tcp": 6, "udp": 17, "icmp": 1, "ipv6-icmp": 58, "icmpv6": 58, "sctp": 132, "gre": 47}
FLOW_TIME_FORMAT = "%d/%m/%Y %I:%M:%S %p"   # patched CICFlowMeter and Engelen's CIC-IDS2017

UNSW_RENAME = ("Source IP=src_ip,Source Port=src_port,Destination IP=dst_ip,Destination Port=dst_port,"
               "Protocol=protocol,Start time=start,Last time=end,Attack category=label")
UNSW_LABEL_MAP = "Backdoors=Backdoor"


def _clean_key(series: pd.Series) -> pd.Series:
    """Text form used for joining: '80.0' -> '80', NaN/'-'/'' -> NA (wildcard)."""
    s = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.mask(s.isin(["", "-", "nan", "<NA>"]))


def load_schedule(path, rename=None, epoch=False, tz_offset_h=0.0, label_map=None,
                  time_format=None) -> pd.DataFrame:
    sch = pd.read_csv(path, low_memory=False, encoding_errors="replace")
    sch.columns = [c.strip() for c in sch.columns]
    if rename:
        sch = sch.rename(columns=dict(item.split("=", 1) for item in rename.split(",")))
    sch.columns = [norm(c) for c in sch.columns]
    for col in ("start", "end"):
        if epoch:
            sch[col] = pd.to_datetime(pd.to_numeric(sch[col], errors="coerce"), unit="s")
        else:
            sch[col] = pd.to_datetime(sch[col], format=time_format or "mixed", errors="coerce", utc=True)
            sch[col] = sch[col].dt.tz_convert(None)
        sch[col] = sch[col] + pd.Timedelta(hours=tz_offset_h)
    sch = sch.dropna(subset=["start", "end"])
    sch["label"] = sch["label"].fillna("attack").astype(str).str.strip()
    if label_map:
        sch["label"] = sch["label"].replace(dict(item.split("=", 1) for item in label_map.split(",")))
    if "protocol" in sch:
        p = sch["protocol"].astype(str).str.strip().str.lower()
        sch["protocol"] = p.map(PROTO).where(p.map(PROTO).notna(), pd.to_numeric(p, errors="coerce"))
    for k in KEYS:
        if k in sch:
            sch[k] = _clean_key(sch[k])
    return sch


def add_reversed(sch: pd.DataFrame) -> pd.DataFrame:
    rev = sch.copy()
    for a, b in (("src_ip", "dst_ip"), ("src_port", "dst_port")):
        if a in sch and b in sch:
            rev[a], rev[b] = sch[b].values, sch[a].values
    return pd.concat([sch, rev], ignore_index=True).drop_duplicates()


def label_frame(flows: pd.DataFrame, sch: pd.DataFrame, ts_col: str, tol: float, default: str,
                flow_time_format: str) -> pd.Series:
    ts = pd.to_datetime(flows[ts_col], format=flow_time_format, errors="coerce")
    if ts.isna().mean() > 0.5:   # current CICFlowMeter writes ISO 8601 (UTC)
        ts = pd.to_datetime(flows[ts_col], format="ISO8601", errors="coerce")
    if ts.isna().mean() > 0.5:   # last resort for other styles
        ts = pd.to_datetime(flows[ts_col], format="mixed", dayfirst=True, errors="coerce")
    labels = pd.Series(default, index=flows.index, dtype=object)
    keys = [k for k in KEYS if k in sch.columns and k in flows.columns]
    left = pd.DataFrame({k: _clean_key(flows[k]) for k in keys})
    left["_ts"], left["_row"] = ts.values, flows.index
    tol_td = pd.Timedelta(seconds=tol)
    spec = sch[keys].notna().apply(lambda r: tuple(k for k in keys if r[k]), axis=1)
    for used, rows in sch.groupby(spec):
        used = list(used)
        right = rows[used + ["start", "end", "label"]]
        m = left.merge(right, on=used, how="inner") if used else left.merge(right, how="cross")
        m = m[(m["_ts"] >= m["start"] - tol_td) & (m["_ts"] <= m["end"] + tol_td)]
        if len(m):
            first = m.drop_duplicates("_row").set_index("_row")["label"]
            unset = labels.loc[first.index] == default
            labels.loc[first.index[unset.values]] = first[unset.values]
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flows", required=True, help="CICFlowMeter CSV file or folder of CSVs")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--flow-time-format", default=FLOW_TIME_FORMAT)
    ap.add_argument("--tolerance", type=float, default=2.0, help="seconds")
    ap.add_argument("--default-label", default="BENIGN")
    ap.add_argument("--rename", help="'Old=new,...' renames for schedule columns")
    ap.add_argument("--label-map", help="'old=new,...' label replacements")
    ap.add_argument("--epoch", action="store_true", help="schedule start/end are Unix seconds")
    ap.add_argument("--tz-offset", type=float, default=0.0, help="hours added to schedule times")
    ap.add_argument("--bidirectional", action="store_true", help="also match src<->dst reversed")
    ap.add_argument("--gt-unsw", action="store_true",
                    help="shortcut for NUSW-NB15_GT.csv: column renames, Unix seconds, category clean-up")
    a = ap.parse_args()
    if a.gt_unsw:
        a.rename, a.epoch = a.rename or UNSW_RENAME, True
        a.label_map = a.label_map or UNSW_LABEL_MAP

    sch = load_schedule(a.schedule, a.rename, a.epoch, a.tz_offset, a.label_map)
    if a.bidirectional:
        sch = add_reversed(sch)
    print(f"schedule: {len(sch):,} rows, {sch['start'].min()} -> {sch['end'].max()} (UTC)")
    src = Path(a.flows)
    files = sorted(src.glob("*.csv")) if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = attacks = 0
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        normed = {c: norm(c) for c in df.columns}
        view = df.rename(columns=normed)
        lab = label_frame(view, sch, norm(a.timestamp_col), a.tolerance, a.default_label,
                          a.flow_time_format)
        label_col = next((c for c in df.columns if normed[c] == "label"), "Label")
        df[label_col] = lab.values
        df.to_csv(out / f.name, index=False)
        n_att = int((lab != a.default_label).sum())
        total, attacks = total + len(df), attacks + n_att
        print(f"{f.name}: {len(df):,} flows, {n_att:,} labelled as attack "
              f"({lab[lab != a.default_label].value_counts().head(5).to_dict()})")
    print(f"TOTAL: {total:,} flows, {attacks:,} attack flows ({attacks / max(total, 1):.2%})")


if __name__ == "__main__":
    main()

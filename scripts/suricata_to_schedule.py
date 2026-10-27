"""Turn Suricata alerts (eve.json) into a labelling schedule for scripts/label_flows.py.

One schedule row per (src_ip, src_port, dst_ip, dst_port, protocol) with
start = first alert (or the flow start Suricata reports) and end = last alert,
all converted to UTC. Engine events ("SURICATA ..." decoder/stream signatures),
low-severity alerts and benign categories are dropped, so only signature hits
that indicate an attack become labels.

    python scripts/suricata_to_schedule.py --eve data/suricata_ptma/eve.json \
        --out data/ptma_suricata_schedule.csv --max-severity 2

Severity in Suricata: 1 = high, 2 = medium, 3 = low, 4 = informational.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

BENIGN_CATEGORIES = ["Not Suspicious Traffic", "Generic Protocol Command Decode", "Unknown Traffic"]
PROTO_NUM = {"TCP": 6, "UDP": 17, "ICMP": 1, "IPV6-ICMP": 58, "SCTP": 132}


def read_alerts(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"event_type":"alert"' not in line.replace(" ", ""):
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            al = ev.get("alert", {})
            rows.append({
                "time": ev.get("timestamp"),
                "flow_start": (ev.get("flow") or {}).get("start"),
                "src_ip": ev.get("src_ip"), "src_port": ev.get("src_port"),
                "dst_ip": ev.get("dest_ip"), "dst_port": ev.get("dest_port"),
                "protocol": PROTO_NUM.get(str(ev.get("proto", "")).upper()),
                "signature": al.get("signature", ""), "signature_id": al.get("signature_id"),
                "category": al.get("category", "") or "Uncategorised",
                "severity": al.get("severity", 3),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eve", required=True, nargs="+", help="one or more eve.json files")
    ap.add_argument("--out", required=True, help="schedule CSV to write")
    ap.add_argument("--max-severity", type=int, default=2)
    ap.add_argument("--exclude-category", nargs="*", default=BENIGN_CATEGORIES)
    ap.add_argument("--label-field", choices=["category", "signature", "attack"], default="category")
    ap.add_argument("--report", help="optional CSV with alert counts per signature")
    ap.add_argument("--propagate-sid", nargs="*", type=int, default=[2001972],
                    help="campaign signatures (kept whatever their severity): every flow from the alerted "
                         "source to the same destination and port inside the alert window is labelled. "
                         "Default 2001972 = ET SCAN Behavioral Unusually fast Terminal Server Traffic (RDP)")
    ap.add_argument("--propagate-window", type=float, default=600, help="seconds added before/after")
    ap.add_argument("--exclude-src", nargs="*", default=[],
                    help="source IPs confirmed legitimate by the institution (never labelled)")
    a = ap.parse_args()

    al = pd.concat([read_alerts(p) for p in a.eve], ignore_index=True)
    print(f"alerts read: {len(al):,}")
    if al.empty:
        pd.DataFrame(columns=["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "start", "end",
                              "label"]).to_csv(a.out, index=False)
        print("no alerts - empty schedule written")
        return
    engine = al["signature"].str.startswith("SURICATA ")
    low = al["severity"] > a.max_severity
    benign = al["category"].isin(a.exclude_category)
    print(f"  dropped: engine events {int(engine.sum()):,}, severity > {a.max_severity} "
          f"{int((low & ~engine).sum()):,}, benign categories {int((benign & ~engine & ~low).sum()):,}")
    campaign = al["signature_id"].isin(a.propagate_sid)
    excluded = al["src_ip"].isin(a.exclude_src)
    if a.exclude_src:
        print(f"  excluded sources: {int(excluded.sum()):,} alerts from {al.loc[excluded, 'src_ip'].nunique()} IPs")
    kept = al[~engine & ~low & ~benign & ~campaign & ~excluded].copy()
    camp = al[campaign & ~excluded].copy()
    if a.report:
        (al.assign(kept=~engine & ~low & ~benign)
           .groupby(["signature_id", "signature", "category", "severity", "kept"]).size()
           .rename("alerts").reset_index().sort_values("alerts", ascending=False)
           .to_csv(a.report, index=False))

    for c in ("time", "flow_start"):
        kept[c] = pd.to_datetime(kept[c], utc=True, errors="coerce", format="ISO8601").dt.tz_convert(None)
    kept["label"] = kept[a.label_field] if a.label_field != "attack" else "attack"
    keys = ["src_ip", "src_port", "dst_ip", "dst_port", "protocol"]
    sch = (kept.groupby(keys, dropna=False)
               .agg(start=("flow_start", "min"), t_first=("time", "min"), end=("time", "max"),
                    label=("label", lambda s: Counter(s).most_common(1)[0][0]))
               .reset_index())
    sch["start"] = sch["start"].fillna(sch["t_first"])
    sch = sch.drop(columns="t_first")

    if len(camp):
        camp["time"] = pd.to_datetime(camp["time"], utc=True, errors="coerce", format="ISO8601").dt.tz_convert(None)
        w = pd.Timedelta(seconds=a.propagate_window)
        c = (camp.groupby(["src_ip", "dst_ip", "dst_port", "protocol"], dropna=False)
                 .agg(start=("time", "min"), end=("time", "max"), alerts=("time", "size"),
                      label=("category", "first"))
                 .reset_index())
        c["start"], c["end"] = c["start"] - w, c["end"] + w
        c["src_port"] = pd.NA       # wildcard: the whole campaign, not just the alerting connection
        print(f"  campaign rows: {len(c)} sources from SIDs {a.propagate_sid} "
              f"({int(c['alerts'].sum()):,} alerts, window ±{a.propagate_window:.0f}s)")
        sch = pd.concat([sch, c[keys + ["start", "end", "label"]]], ignore_index=True)
    sch.to_csv(a.out, index=False)
    print(f"schedule rows: {len(sch):,} (signature alerts kept {len(kept):,})")
    print("labels:", sch["label"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()

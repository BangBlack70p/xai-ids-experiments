"""Pseudonymise IP addresses in labelled flow CSVs (run AFTER labelling).

Each IP is replaced by "ip_" + the first 16 hex characters of HMAC-SHA256(key, ip).
The same IP always maps to the same pseudonym, so entity-level analysis
(Fig. 5) still works, while the original address cannot be recovered without
the key. The Flow ID column, which embeds IPs, is rebuilt from the pseudonyms.
An optional flag marks whether each address was internal (RFC 1918 or the
--internal prefixes) before it is hidden.

    python scripts/pseudonymize.py --in data/ptma_flows_labelled_raw --out data/ptma_flows_labelled \
        --key-file secrets/ptma_hmac.key --internal <campus-prefix>

Keep the key file outside the project folder and never publish it.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import secrets
from pathlib import Path

import pandas as pd

IP_COLS = {"src ip": "src", "dst ip": "dst", "source ip": "src", "destination ip": "dst",
           "src_ip": "src", "dst_ip": "dst"}


def load_key(path: Path) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
        print(f"new key written to {path} - keep it private")
    return bytes.fromhex(path.read_text().strip())


def is_internal(ip: str, prefixes: list[str]) -> bool:
    try:
        if ipaddress.ip_address(ip).is_private:
            return True
    except ValueError:
        return False
    return any(ip.startswith(p) for p in prefixes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="CSV file or folder")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--internal", nargs="*", default=[], help="extra internal prefixes, e.g. 172.31.")
    a = ap.parse_args()

    key = load_key(Path(a.key_file))
    cache: dict[str, str] = {}

    def pseudo(ip) -> str:
        ip = str(ip)
        if ip not in cache:
            cache[ip] = "ip_" + hmac.new(key, ip.encode(), hashlib.sha256).hexdigest()[:16]
        return cache[ip]

    src = Path(a.inp)
    files = sorted(src.glob("*.csv")) if src.is_dir() else [src]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        cols = {c: IP_COLS[c.strip().lower()] for c in df.columns if c.strip().lower() in IP_COLS}
        for col, side in cols.items():
            df[f"{side}_internal"] = [int(is_internal(str(v), a.internal)) for v in df[col]]
            df[col] = df[col].map(pseudo)
        fid = next((c for c in df.columns if c.strip().lower() in {"flow id", "flow_id"}), None)
        if fid and len(cols) == 2:
            s, d = list(cols)
            sp = next((c for c in df.columns if c.strip().lower() in {"src port", "src_port", "source port"}), None)
            dp = next((c for c in df.columns if c.strip().lower() in {"dst port", "dst_port", "destination port"}), None)
            pr = next((c for c in df.columns if c.strip().lower() == "protocol"), None)
            parts = [df[s], df[d]] + [df[c].astype(str) for c in (sp, dp, pr) if c]
            df[fid] = pd.concat(parts, axis=1).astype(str).agg("-".join, axis=1)
        df.to_csv(out / f.name, index=False)
        print(f"{f.name}: {len(df):,} rows, {len(cols)} IP columns pseudonymised")
    print(f"{len(cache):,} distinct addresses replaced")


if __name__ == "__main__":
    main()

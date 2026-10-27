"""Write the pseudonyms of the NOC monitoring addresses confirmed by IT staff (Section 5.5).

The four addresses were excluded from the attack labels with
`prepare_data.py ptma --exclude-src ...`. This script takes them from the last such
run in logs/prepare_data.log (or from --ips), applies the same keyed HMAC as
scripts/pseudonymize.py, and writes ONLY the pseudonyms to
data/ptma_noc_pseudonyms.txt. No original address is written anywhere.

    python scripts/noc_pseudonyms.py
    python scripts/noc_pseudonyms.py --ips a.b.c.d e.f.g.h
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def from_log(log: Path) -> list[str]:
    ips: list[str] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "--exclude-src" not in line:
            continue
        toks = line.split("--exclude-src", 1)[1].split()
        found = []
        for t in toks:
            if t.startswith("--") or t.startswith("="):
                break
            if IP.match(t.strip('"')):
                found.append(t.strip('"'))
        if found:
            ips = found          # keep the last run
    return ips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ips", nargs="*", help="NOC addresses (default: read from the log)")
    ap.add_argument("--log", default=str(ROOT / "logs" / "prepare_data.log"))
    ap.add_argument("--key-file", default=str(Path.home() / ".xai_ids" / "ptma_hmac.key"))
    ap.add_argument("--out", default=str(ROOT / "data" / "ptma_noc_pseudonyms.txt"))
    a = ap.parse_args()
    ips = a.ips or from_log(Path(a.log))
    if not ips:
        raise SystemExit("no addresses found: pass them with --ips")
    key = bytes.fromhex(Path(a.key_file).read_text().strip())
    pseudo = sorted({"ip_" + hmac.new(key, ip.encode(), hashlib.sha256).hexdigest()[:16] for ip in ips})
    Path(a.out).write_text("\n".join(pseudo) + "\n")
    print(f"{len(pseudo)} NOC pseudonyms written to {a.out}")


if __name__ == "__main__":
    main()

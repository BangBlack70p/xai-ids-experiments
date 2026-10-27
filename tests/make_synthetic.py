"""Synthetic stand-ins for CIC-IDS2017, UNSW-NB15 and PTMA, for smoke tests only.

The files imitate the *format* of real flow exports (CICFlowMeter-style column
names, identifiers, timestamps, 'Attempted' labels, duplicates, inf values).
The numbers mean nothing; never report results computed on them.

    python tests/make_synthetic.py tests/synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["Flow Duration", "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
            "Total Length of Bwd Packet", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
            "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "SYN Flag Count", "RST Flag Count",
            "ACK Flag Count", "Down/Up Ratio", "Init Fwd Win Bytes", "Idle Mean"]

# class -> multiplicative profile on the benign baseline
PROFILES = {
    "BENIGN": {},
    "FTP-Patator": {"Flow Duration": 0.3, "Total Fwd Packet": 3, "RST Flag Count": 5, "Init Fwd Win Bytes": 2},
    "SSH-Patator": {"Flow Duration": 0.5, "Total Fwd Packet": 4, "Fwd Packet Length Mean": 0.4, "ACK Flag Count": 3},
    "DoS Hulk": {"Flow Packets/s": 8, "Flow Duration": 0.2, "SYN Flag Count": 6, "Total Bwd packets": 0.3},
    "DoS Slowloris": {"Flow Duration": 8, "Idle Mean": 6, "Flow Packets/s": 0.2},
    "Web Attack - Brute Force": {"Total Length of Fwd Packet": 4, "Fwd Packet Length Mean": 3},
    "Infiltration": {"Total Length of Fwd Packet": 12, "Down/Up Ratio": 0.1, "Flow Duration": 5},
    "Bot": {"Flow IAT Mean": 0.2, "Total Fwd Packet": 0.5, "Idle Mean": 3},
    "PortScan": {"Total Fwd Packet": 0.1, "Flow Duration": 0.05, "RST Flag Count": 8, "Total Bwd packets": 0.1},
    "DDoS": {"Flow Packets/s": 10, "SYN Flag Count": 9, "Total Length of Bwd Packet": 0.1},
}
SCHEDULE = {  # CIC-IDS2017 week, attacks per day
    "2017-07-03": [],
    "2017-07-04": ["FTP-Patator", "SSH-Patator"],
    "2017-07-05": ["DoS Hulk", "DoS Slowloris"],
    "2017-07-06": ["Web Attack - Brute Force", "Infiltration"],
    "2017-07-07": ["Bot", "PortScan", "DDoS"],
}


def _flows(rng, labels, shift=1.0, noise=0.6):
    base = np.exp(rng.normal(0, noise, (len(labels), len(FEATURES)))) * 100 * shift
    df = pd.DataFrame(base, columns=FEATURES)
    for i, lab in enumerate(labels):
        for f, m in PROFILES.get(lab, {}).items():
            df.iat[i, FEATURES.index(f)] *= m * rng.uniform(0.7, 1.3)
    return df


def _ids(rng, n, day):
    t0 = pd.Timestamp(day) + pd.Timedelta(hours=9)
    ts = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 8 * 3600, n)), unit="s")
    src = [f"192.168.10.{x}" for x in rng.integers(2, 60, n)]
    dst = [f"172.16.0.{x}" for x in rng.integers(1, 20, n)]
    sp, dp = rng.integers(1024, 65535, n), rng.choice([21, 22, 80, 443, 8080, 3306], n)
    proto = rng.choice([6, 17], n, p=[0.85, 0.15])
    return pd.DataFrame({"Flow ID": [f"{a}-{b}-{c}-{d}" for a, b, c, d in zip(src, dst, sp, dp)],
                         "Src IP": src, "Src Port": sp, "Dst IP": dst, "Dst Port": dp,
                         "Protocol": proto, "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")})


def cic(out: Path, rng, n_day=6000, noisy_labels=False):
    out.mkdir(parents=True, exist_ok=True)
    for day, attacks in SCHEDULE.items():
        n_att = int(n_day * 0.15) if attacks else 0
        labels = ["BENIGN"] * (n_day - n_att) + list(rng.choice(attacks, n_att)) if attacks \
            else ["BENIGN"] * n_day
        labels = list(rng.permutation(labels))
        ids = _ids(rng, n_day, day)
        # attacks happen in bursts: sort attack rows into the second half of the day
        ids = ids.sort_values("Timestamp").reset_index(drop=True)
        order = np.argsort([0 if lab == "BENIGN" else rng.uniform(0.3, 1) for lab in labels], kind="stable")
        labels = [labels[i] for i in order]
        df = pd.concat([ids, _flows(rng, labels)], axis=1)
        df["Label"] = labels
        att = df["Label"] != "BENIGN"
        attempted = att & (rng.random(n_day) < 0.05)
        df.loc[attempted, "Label"] = df.loc[attempted, "Label"] + " - Attempted"
        if noisy_labels:  # 'original' release: some attacks mislabelled as benign and vice versa
            flip = rng.random(n_day) < 0.03
            df.loc[flip & att, "Label"] = "BENIGN"
        df.loc[rng.random(n_day) < 0.002, "Flow Bytes/s"] = np.inf
        dups = df.sample(frac=0.05, random_state=1)
        df = pd.concat([df, dups]).sort_values("Timestamp")
        df.to_csv(out / f"{day}.csv", index=False)


def flows_like(out: Path, rng, days, attacks, n_day, shift):
    out.mkdir(parents=True, exist_ok=True)
    for day in days:
        n_att = int(n_day * 0.1)
        labels = ["BENIGN"] * (n_day - n_att) + list(rng.choice(attacks, n_att))
        labels = list(rng.permutation(labels))
        df = pd.concat([_ids(rng, n_day, day), _flows(rng, labels, shift=shift, noise=0.8)], axis=1)
        df["Label"] = labels
        df.to_csv(out / f"{day}.csv", index=False)


def unsw_official(out: Path, rng):
    out.mkdir(parents=True, exist_ok=True)
    cats = ["Normal", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Generic", "Backdoor"]
    for name, n in [("UNSW_NB15_training-set.csv", 8000), ("UNSW_NB15_testing-set.csv", 4000)]:
        cat = rng.choice(cats, n, p=[0.55, 0.12, 0.1, 0.08, 0.07, 0.05, 0.03])
        lab = (cat != "Normal").astype(int)
        df = pd.DataFrame({"id": np.arange(n), "dur": rng.exponential(1, n) * (1 + lab),
                           "proto": rng.choice(["tcp", "udp", "arp"], n),
                           "service": rng.choice(["-", "http", "dns", "ftp"], n),
                           "state": rng.choice(["FIN", "INT", "CON"], n),
                           "spkts": rng.poisson(10 + 20 * lab), "dpkts": rng.poisson(10),
                           "sbytes": rng.exponential(500, n) * (1 + 3 * lab),
                           "dbytes": rng.exponential(800, n), "rate": rng.exponential(50, n) * (1 + lab),
                           "sttl": rng.choice([31, 62, 254], n), "attack_cat": cat, "label": lab})
        df.to_csv(out / name, index=False)


def main(target: str) -> None:
    root = Path(target)
    rng = np.random.default_rng(42)
    cic(root / "cic_corrected", rng)
    cic(root / "cic_original", np.random.default_rng(42), noisy_labels=True)
    flows_like(root / "unsw_flows", rng, ["2015-01-22", "2015-01-23", "2015-02-17", "2015-02-18"],
               ["DoS Hulk", "PortScan", "Infiltration", "Web Attack - Brute Force"], 4000, shift=1.4)
    unsw_official(root / "unsw_official", rng)
    flows_like(root / "ptma_flows", rng, ["2026-11-02", "2026-11-09", "2026-11-16", "2026-11-23"],
               ["SSH-Patator", "PortScan", "Infiltration"], 4000, shift=0.8)
    print(f"Synthetic data written to {root.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/synthetic")

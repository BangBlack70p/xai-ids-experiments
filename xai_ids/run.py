"""Command line entry point.

    python -m xai_ids.run --config configs/cic_unsw_ptma.yaml --exp E1 E2 E3 E4 E5
    python -m xai_ids.run --config configs/cic_unsw_ptma.yaml --exp all
    python -m xai_ids.run --config configs/cic_unsw_ptma.yaml --exp E4 E5 CASE --e5-transfer-only
"""
from __future__ import annotations

import argparse
import time

from .config import load_config
from .experiments import Context, run_case, run_e1, run_e2, run_e3, run_e4, run_e5


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Paper A experiments E1-E5")
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", nargs="+", default=["all"],
                    help="E1 E2 E3 E4 E5 CASE or 'all'")
    ap.add_argument("--e5-transfer-only", action="store_true",
                    help="E5: only the cross-dataset attribution transfer (skip per-dataset XAI already done)")
    ap.add_argument("--inspect", action="store_true",
                    help="only load each dataset and print features, classes and days")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.e5_transfer_only:
        cfg.setdefault("e5", {})["datasets"] = []
    ctx = Context(cfg)
    if args.inspect:
        for key, dcfg in cfg["datasets"].items():
            for pk in ("path", "train_path", "test_path"):
                if dcfg.get(pk):
                    d = ctx.raw(key, pk)
                    print(f"\n== {key} ({pk}) ==")
                    print("features:", ", ".join(d.X.columns))
                    print("categorical:", d.categorical)
                    print("classes:\n", d.cls.value_counts().to_string())
                    print("flows per day:\n", d.meta["_day"].value_counts().sort_index().to_string())
        return
    wanted = {"E1", "E2", "E3", "E4", "E5", "CASE"} if "all" in args.exp else {e.upper() for e in args.exp}
    start = time.time()
    if "E1" in wanted:
        run_e1(ctx)
    if "E2" in wanted:
        run_e2(ctx)
    if "E3" in wanted:
        run_e3(ctx)
    transfer = None
    if "E4" in wanted:
        transfer = run_e4(ctx)
    elif "E5" in wanted:   # E5 only needs the transfer models, not the replication runs
        transfer = run_e4(ctx, replication=False)
    if "E5" in wanted:
        run_e5(ctx, transfer)
    if "CASE" in wanted:
        run_case(ctx)
    print(f"Done in {time.time() - start:.0f}s. Results in {cfg['output_dir']}")


if __name__ == "__main__":
    main()

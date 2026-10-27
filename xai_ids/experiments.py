"""Experiments E1-E5 of Paper A.

E1  conventional vs leakage-safe protocol (CIC-IDS2017)                 -> RQ1
E2  model comparison under the leakage-safe protocol (CIC-IDS2017)       -> RQ1
E3  leave-one-day-out, unseen attack types (CIC-IDS2017)                 -> RQ1, RQ2
E4  replication on UNSW-NB15 + transfer CIC -> UNSW-NB15 / PTMA          -> RQ2
E5  explanation agreement, faithfulness, stability, transfer             -> RQ2, RQ3

Each experiment writes CSV tables, PNG figures and a short Markdown summary to
<output_dir>/<E#>/. Numbers that go into the paper are also collected in
<output_dir>/paper_numbers.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from . import metrics as M  # noqa: E402
from . import xai as XA  # noqa: E402
from .data import DAY_COL, FlowData, load_dataset, shared_feature_list  # noqa: E402
from .models import NEEDS_SCALING, Detector  # noqa: E402
from .split import (Preprocessor, chronological_split, class_counts, deduplicate,  # noqa: E402
                    leakage_report, leave_day_out, random_split, row_hash)


# ====================================================================== helpers
class Context:
    """Shared state: config, output folder, dataset cache, paper numbers."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out = Path(cfg["output_dir"])
        self.out.mkdir(parents=True, exist_ok=True)
        self._raw: dict[str, FlowData] = {}
        self.numbers_path = self.out / "paper_numbers.json"
        self.numbers = json.loads(self.numbers_path.read_text()) if self.numbers_path.exists() else {}

    def ds_cfg(self, key: str) -> dict:
        if key not in self.cfg["datasets"]:
            raise KeyError(f"Dataset '{key}' is not defined in the config")
        return self.cfg["datasets"][key]

    def has(self, key: str) -> bool:
        return key in self.cfg["datasets"]

    def raw(self, key: str, path_key: str = "path") -> FlowData:
        ck = f"{key}:{path_key}"
        if ck not in self._raw:
            t = time.time()
            self._raw[ck] = load_dataset(key, self.ds_cfg(key), path_key)
            d = self._raw[ck]
            print(f"  loaded {key} ({path_key}): {len(d):,} flows, {d.X.shape[1]} features, "
                  f"{int(d.y.sum()):,} attacks [{time.time() - t:.1f}s]")
        return self._raw[ck]

    def folder(self, name: str) -> Path:
        p = self.out / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def record(self, key: str, value) -> None:
        self.numbers[key] = value
        self.numbers_path.write_text(json.dumps(self.numbers, indent=2, default=float))


def leakage_safe_parts(ctx: Context, key: str, features: list[str] | None = None):
    """Dedup + partition according to the dataset's `partition` setting."""
    cfg = ctx.ds_cfg(key)
    mode = cfg.get("partition", "chronological")
    if mode == "official":
        tr_raw, te_raw = ctx.raw(key, "train_path"), ctx.raw(key, "test_path")
        if features:
            tr_raw, te_raw = tr_raw.restrict_features(features), te_raw.restrict_features(features)
        common = [c for c in tr_raw.X.columns if c in te_raw.X.columns]
        tr_raw, te_raw = tr_raw.restrict_features(common), te_raw.restrict_features(common)
        tr, s1 = deduplicate(tr_raw)
        te, s2 = deduplicate(te_raw)
        in_train = row_hash(te.X).isin(set(row_hash(tr.X))).to_numpy()
        te = te.subset(te.y.index[~in_train])
        idx_tr, idx_va = train_test_split(tr.y.index, test_size=ctx.cfg["split"]["val"],
                                          stratify=tr.y, random_state=0)
        parts = {"train": tr.subset(idx_tr), "val": tr.subset(idx_va), "test": te}
        stats = {"train_file": s1, "test_file": s2}
    else:
        data = ctx.raw(key)
        if features:
            data = data.restrict_features(features)
        data, stats = deduplicate(data)
        parts = chronological_split(data, ctx.cfg["split"], ctx.cfg["split"].get("by_class", True))
    stats.update(leakage_report(parts))
    return parts, stats


def fit_predict(model: str, parts: dict[str, FlowData], seed: int, ctx: Context,
                leaky_preprocessing: bool = False, fixed_theta: float | None = None):
    """Fit preprocessing + detector, select theta* on validation, score the test set."""
    pre = Preprocessor(parts["train"].categorical, scale=model in NEEDS_SCALING)
    if leaky_preprocessing:  # deliberately wrong: used only for the E1 contrast
        pre.fit(pd.concat([p.X for p in parts.values()]))
    else:
        pre.fit(parts["train"].X)
    Xtr, Xva, Xte = (pre.transform(parts[k].X) for k in ("train", "val", "test"))
    ytr, yva, yte = (parts[k].y.to_numpy() for k in ("train", "val", "test"))
    params = ctx.cfg["model"].get(model, {})
    det = Detector(model, seed, params, ctx.cfg["model"].get("resample")).fit(Xtr, ytr)
    s_va, s_te = det.score(Xva), det.score(Xte)
    theta = fixed_theta if fixed_theta is not None else M.select_threshold(yva, s_va)
    return {"det": det, "pre": pre, "theta": theta, "s_val": s_va, "s_test": s_te,
            "X_train": Xtr, "X_val": Xva, "X_test": Xte, "y_val": yva, "y_test": yte,
            "metrics": M.detection_metrics(yte, s_te, theta)}


def mean_sd(rows: list[dict], group: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    num = [c for c in ["precision", "recall", "f1", "far", "roc_auc", "pr_auc", "mcc", "accuracy",
                       "theta"] if c in df]
    agg = df.groupby(group, sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg


def fmt(agg: pd.DataFrame, cols=("precision", "recall", "f1", "far", "pr_auc")) -> pd.DataFrame:
    out = pd.DataFrame(index=agg.index)
    for c in cols:
        if f"{c}_mean" in agg:
            out[c] = [f"{m:.4f} ± {s:.4f}" if s == s else f"{m:.4f}"
                      for m, s in zip(agg[f"{c}_mean"], agg[f"{c}_std"])]
    return out


def write_md(path: Path, title: str, blocks: list[tuple[str, pd.DataFrame | str]]) -> None:
    lines = [f"# {title}", ""]
    for head, body in blocks:
        lines += [f"## {head}", ""]
        lines += [body.to_markdown() if isinstance(body, pd.DataFrame) else str(body), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ====================================================================== E1
def run_e1(ctx: Context) -> None:
    """Conventional vs leakage-safe protocol, Random Forest, all seeds."""
    print("E1: protocol ablation")
    out, seeds, rows, counts = ctx.folder("E1"), ctx.cfg["seeds"], [], {}
    variants = []
    if ctx.has("cic_original"):
        variants.append(("conventional (original labels, random split, leaky preprocessing, θ=0.5)",
                         "cic_original", False, "random", True, 0.5))
    variants += [
        ("corrected labels, random split", "cic", False, "random", False, None),
        ("+ deduplication, random split", "cic", True, "random", False, None),
        ("leakage-safe (corrected, dedup, chronological)", "cic", True, "chrono", False, None),
    ]
    for label, key, dedup, split, leaky, theta in variants:
        data = ctx.raw(key)
        if dedup:
            data, _ = deduplicate(data)
        for seed in seeds:
            parts = random_split(data, ctx.cfg["split"], seed) if split == "random" \
                else chronological_split(data, ctx.cfg["split"], ctx.cfg["split"].get("by_class", True))
            r = fit_predict("rf", parts, seed, ctx, leaky, theta)
            rows.append({"protocol": label, "seed": seed, **r["metrics"],
                         **leakage_report(parts)})
            print(f"  {label} seed={seed}: F1={r['metrics']['f1']:.4f}")
        counts[label] = class_counts(parts)
    df = pd.DataFrame(rows)
    df.to_csv(out / "e1_runs.csv", index=False)
    agg = mean_sd(rows, "protocol")
    agg.to_csv(out / "e1_summary.csv")
    leak = df.groupby("protocol", sort=False)[[c for c in df if c.startswith("overlap_")]].max()
    write_md(out / "E1.md", "E1 - protocol ablation (Random Forest)",
             [("Mean ± SD over seeds", fmt(agg)), ("Shared feature vectors between partitions", leak)])
    safe, first = agg.iloc[-1], agg.iloc[0]
    ctx.record("E1", {"f1_first_variant": first["f1_mean"], "f1_leakage_safe": safe["f1_mean"],
                      "far_first_variant": first["far_mean"], "far_leakage_safe": safe["far_mean"],
                      "first_variant": agg.index[0]})


# ====================================================================== E2
def run_e2(ctx: Context) -> None:
    print("E2: model comparison (leakage-safe CIC-IDS2017)")
    out = ctx.folder("E2")
    parts, stats = leakage_safe_parts(ctx, "cic")
    class_counts(parts).to_csv(out / "class_counts_per_partition.csv")
    pd.Series(stats).to_csv(out / "dedup_and_leakage.csv")
    models = ctx.cfg.get("e2", {}).get("models", ["lr", "dt", "rf", "xgb", "iforest", "autoencoder"])
    rows, first_seed = [], {}
    for model in models:
        for seed in ctx.cfg["seeds"]:
            t = time.time()
            r = fit_predict(model, parts, seed, ctx)
            rows.append({"model": model, "seed": seed, "fit_seconds": time.time() - t, **r["metrics"]})
            print(f"  {model} seed={seed}: F1={r['metrics']['f1']:.4f} PR-AUC={r['metrics']['pr_auc']:.4f}")
            if seed == ctx.cfg["seeds"][0]:
                first_seed[model] = r
    pd.DataFrame(rows).to_csv(out / "e2_runs.csv", index=False)
    agg = mean_sd(rows, "model")
    agg.to_csv(out / "e2_summary.csv")

    rf = first_seed["rf"]
    y = rf["y_test"]
    lo, hi = M.bootstrap_ci(y, rf["s_test"], rf["theta"], "f1", ctx.cfg["bootstrap"]["n"])
    tests = []
    for m, r in first_seed.items():
        if m == "rf":
            continue
        mc = M.mcnemar(y, rf["s_test"] >= rf["theta"], r["s_test"] >= r["theta"])
        dl = M.delong(y, rf["s_test"], r["s_test"])
        tests.append({"rf_vs": m, "mcnemar_p": mc["p_value"], "auc_rf": dl["auc_a"],
                      "auc_other": dl["auc_b"], "delong_p": dl["p_value"]})
    tests = pd.DataFrame(tests)
    tests.to_csv(out / "e2_significance.csv", index=False)
    pcr = M.per_class_recall(parts["test"].cls, y, rf["s_test"], rf["theta"])
    pcr.to_csv(out / "e2_rf_per_class_recall.csv")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for m, r in first_seed.items():
        p, rcl, _ = precision_recall_curve(y, r["s_test"])
        ax.plot(rcl, p, label=m, lw=1.2)
    pred = rf["s_test"] >= rf["theta"]
    ax.scatter([pred[y == 1].mean()], [y[pred].mean() if pred.any() else 0], color="k", zorder=5,
               label="RF at θ*")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall, CIC-IDS2017 test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_pr_curves.png", dpi=200)
    plt.close(fig)

    write_md(out / "E2.md", "E2 - model comparison", [
        ("Mean ± SD over seeds", fmt(agg, ("precision", "recall", "f1", "far", "roc_auc", "pr_auc"))),
        ("Random Forest, first seed", f"θ* = {rf['theta']:.4f}; F1 = {rf['metrics']['f1']:.4f} "
                                      f"(95% bootstrap CI {lo:.4f}–{hi:.4f})"),
        ("Significance vs RF (first seed)", tests.set_index("rf_vs")),
        ("RF per-class recall", pcr)])
    best = agg["f1_mean"].idxmax()
    ctx.record("E2", {"rf_f1_mean": agg.loc["rf", "f1_mean"], "rf_far_mean": agg.loc["rf", "far_mean"],
                      "rf_theta_seed0": rf["theta"], "rf_f1_ci": [lo, hi], "best_model": best})


# ====================================================================== E3
def run_e3(ctx: Context) -> None:
    print("E3: leave-one-day-out")
    out = ctx.folder("E3")
    data, _ = deduplicate(ctx.raw("cic"))
    days = sorted(data.meta[DAY_COL].unique())
    rows, pcr_all = [], []
    seed = ctx.cfg["seeds"][0]
    for day in days:
        parts = leave_day_out(data, day, ctx.cfg["split"]["val"])
        if parts["test"].y.sum() == 0:
            print(f"  {day}: no attacks, skipped")
            continue
        r = fit_predict("rf", parts, seed, ctx)
        seen = set(parts["train"].cls.unique())
        pcr = M.per_class_recall(parts["test"].cls, r["y_test"], r["s_test"], r["theta"])
        pcr["held_out_day"] = day
        pcr["seen_in_training"] = [c in seen for c in pcr.index]
        pcr_all.append(pcr)
        rows.append({"held_out_day": day, **r["metrics"]})
        print(f"  {day}: F1={r['metrics']['f1']:.4f}")
    df = pd.DataFrame(rows).set_index("held_out_day")
    df.to_csv(out / "e3_per_day.csv")
    pcr_all = pd.concat(pcr_all)
    pcr_all.to_csv(out / "e3_per_class_recall.csv")
    unseen = pcr_all[(~pcr_all["seen_in_training"]) & (pcr_all.index != "(benign specificity)")]
    write_md(out / "E3.md", "E3 - leave-one-day-out (Random Forest)", [
        ("Per held-out day", df[["precision", "recall", "f1", "far", "pr_auc"]].round(4)),
        ("Recall on attack types never seen in training", unseen.round(4))])
    ctx.record("E3", {"mean_recall_unseen": float(unseen["recall"].mean()) if len(unseen) else None})


# ====================================================================== E4
def _available(ctx: Context, key: str) -> bool:
    """True when the dataset can be loaded (path exists and holds data files)."""
    try:
        ctx.raw(key)
        return True
    except (FileNotFoundError, KeyError) as exc:
        print(f"  skipping {key}: {exc}")
        return False


def run_e4(ctx: Context, replication: bool = True) -> dict:
    print("E4: replication and transfer")
    out = ctx.folder("E4")
    seed, numbers = ctx.cfg["seeds"][0], {}

    # ---- replication on UNSW-NB15 official split (its own feature set)
    if replication and ctx.has("unsw_official"):
        parts, stats = leakage_safe_parts(ctx, "unsw_official")
        rows = []
        for s in ctx.cfg["seeds"]:
            r = fit_predict("rf", parts, s, ctx)
            rows.append({"dataset": "UNSW-NB15 (official split)", "seed": s, **r["metrics"]})
        agg = mean_sd(rows, "dataset")
        agg.to_csv(out / "e4_replication.csv")
        numbers["replication_f1"] = agg["f1_mean"].iloc[0]
        print(f"  replication UNSW-NB15: F1={numbers['replication_f1']:.4f}")

    # ---- transfer in the shared feature space
    targets = [t for t in ctx.cfg.get("e4", {}).get("targets", ["unsw", "ptma"])
               if ctx.has(t) and _available(ctx, t)]
    if not targets:
        print("  no transfer targets available (prepare them with scripts/prepare_data.py)")
        if numbers:
            ctx.record("E4", {**ctx.numbers.get("E4", {}), **numbers})
        return {}
    shared = shared_feature_list([ctx.raw("cic")] + [ctx.raw(t) for t in targets],
                                 ctx.cfg.get("shared_features"))
    pd.Series(shared, name="feature").to_csv(out / "shared_features.csv", index=False)
    print(f"  shared feature space: {len(shared)} features")
    src_parts, _ = leakage_safe_parts(ctx, "cic", shared)
    tgt_parts = {t: leakage_safe_parts(ctx, t, shared)[0] for t in targets}
    rows, pcr_rows, transfer = [], [], None
    for s in ctx.cfg["seeds"]:
        src = fit_predict("rf", src_parts, s, ctx)
        rows.append({"setting": "CIC-IDS2017 within-dataset", "seed": s, **src["metrics"]})
        if transfer is None:          # E5 explains the first-seed model
            transfer = {"source": src, "source_parts": src_parts, "targets": {}}
        for t in targets:
            tparts = tgt_parts[t]
            Xva = src["pre"].transform(tparts["val"].X)
            Xte = src["pre"].transform(tparts["test"].X)
            yva, yte = tparts["val"].y.to_numpy(), tparts["test"].y.to_numpy()
            s_va, s_te = src["det"].score(Xva), src["det"].score(Xte)
            fixed = M.detection_metrics(yte, s_te, src["theta"])
            theta_t = M.select_threshold(yva, s_va) if yva.sum() else src["theta"]
            resel = M.detection_metrics(yte, s_te, theta_t)
            rows.append({"setting": f"CIC → {t}, fixed θ*", "seed": s, **fixed})
            rows.append({"setting": f"CIC → {t}, θ* re-selected on target validation", "seed": s, **resel})
            if tparts["train"].y.sum():
                within = fit_predict("rf", tparts, s, ctx)
                rows.append({"setting": f"{t} within-dataset (reference)", "seed": s, **within["metrics"]})
            pcr = M.per_class_recall(tparts["test"].cls, yte, s_te, src["theta"]).reset_index()
            pcr["target"], pcr["seed"] = t, s
            pcr_rows.append(pcr)
            if s == seed:
                transfer["targets"][t] = {"parts": tparts, "X_test": Xte, "y_test": yte, "s_test": s_te}
            print(f"  seed {s} CIC → {t}: F1 fixed={fixed['f1']:.4f}, re-selected={resel['f1']:.4f}, "
                  f"ROC-AUC={fixed['roc_auc']:.4f}")
        del src
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "e4_transfer_runs.csv", index=False)
    agg = mean_sd(rows, "setting")
    agg.to_csv(out / "e4_transfer.csv")
    pcr_all = pd.concat(pcr_rows)
    pcr_all.to_csv(out / "e4_transfer_per_class_recall_runs.csv", index=False)
    pcr_all.groupby(["target", "cls"], sort=False).agg(
        recall_mean=("recall", "mean"), recall_std=("recall", "std"), n=("n", "first")
    ).to_csv(out / "e4_transfer_per_class_recall.csv")
    for t in targets:
        for tag, setting in (("fixed", f"CIC → {t}, fixed θ*"),
                             ("reselected", f"CIC → {t}, θ* re-selected on target validation")):
            numbers[f"transfer_{t}_f1_{tag}"] = agg.loc[setting, "f1_mean"]
            numbers[f"transfer_{t}_f1_{tag}_sd"] = agg.loc[setting, "f1_std"]
            numbers[f"transfer_{t}_far_{tag}"] = agg.loc[setting, "far_mean"]
        numbers[f"transfer_{t}_roc_auc"] = agg.loc[f"CIC → {t}, fixed θ*", "roc_auc_mean"]
        numbers[f"transfer_{t}_roc_auc_sd"] = agg.loc[f"CIC → {t}, fixed θ*", "roc_auc_std"]
    numbers["transfer_seeds"] = len(ctx.cfg["seeds"])

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(range(len(agg)), agg["f1_mean"], xerr=agg["f1_std"].fillna(0), color="#4a6fa5", capsize=3)
    ax.set_yticks(range(len(agg)), agg.index, fontsize=8)
    ax.set_xlabel(f"F1 on target test partition (mean ± SD, {len(ctx.cfg['seeds'])} seeds)")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "fig_transfer_f1.png", dpi=200)
    plt.close(fig)
    table = fmt(agg, ("precision", "recall", "f1", "far", "roc_auc", "pr_auc"))
    table["theta"] = [f"{m:.3f} ± {sd:.3f}" for m, sd in zip(agg["theta_mean"], agg["theta_std"].fillna(0))]
    write_md(out / "E4.md", "E4 - replication and transfer",
             [(f"Transfer (shared feature space, Random Forest, mean ± SD over {len(ctx.cfg['seeds'])} seeds)",
               table)])
    ctx.record("E4", {**ctx.numbers.get("E4", {}), **numbers})
    return transfer


# ====================================================================== E5
def run_e5(ctx: Context, transfer: dict | None = None) -> None:
    print("E5: explanation evaluation")
    out, xc = ctx.folder("E5"), ctx.cfg["xai"]
    seed, k = ctx.cfg["seeds"][0], xc["top_k"]
    rng = np.random.default_rng(seed)
    summary_blocks, numbers = [], {}

    for key in ctx.cfg.get("e5", {}).get("datasets", ["cic"]):
        if not ctx.has(key):
            continue
        print(f"  dataset {key}")
        parts, _ = leakage_safe_parts(ctx, key)
        r = fit_predict("rf", parts, seed, ctx)
        rf, names = r["det"].model, r["pre"].feature_names
        proba = rf.predict_proba

        # -- alerts = flagged test flows, sampled across classes
        flagged = np.where(r["s_test"] >= r["theta"])[0]
        cls = parts["test"].cls.to_numpy()[flagged]
        n = min(xc["n_alerts"], len(flagged))
        per_class = pd.Series(cls).value_counts()
        quota = np.maximum(1, (per_class / per_class.sum() * n).round().astype(int))
        pick = []
        for c, q in quota.items():
            cand = flagged[cls == c]
            pick.extend(rng.choice(cand, min(q, len(cand)), replace=False))
        alerts = np.array(sorted(pick))
        Xa = r["X_test"][alerts]
        alert_cls = parts["test"].cls.to_numpy()[alerts]
        print(f"    {len(alerts)} alerts explained ({len(flagged):,} flagged)")

        t = time.time()
        shap_a = XA.tree_shap(rf, Xa)
        t_shap = (time.time() - t) / len(Xa)
        ref = r["X_train"][rng.choice(len(r["X_train"]), min(5000, len(r["X_train"])), replace=False)]
        t = time.time()
        lime_a = XA.lime_weights(proba, ref, Xa, names, xc["lime_num_samples"], seed)
        t_lime = (time.time() - t) / len(Xa)

        # -- local agreement
        loc = XA.local_agreement(shap_a, lime_a, k)
        loc["class"] = alert_cls
        loc.to_csv(out / f"{key}_local_agreement.csv", index=False)
        loc_summary = loc.groupby("class")[["kendall_tau", "topk_overlap", "sign_agreement"]].mean()
        loc_summary.loc["ALL"] = loc[["kendall_tau", "topk_overlap", "sign_agreement"]].mean()

        # -- global agreement
        gs = rng.choice(len(r["X_test"]), min(2000, len(r["X_test"])), replace=False)
        shap_g = XA.tree_shap(rf, r["X_test"][gs])
        imp = XA.global_importances(rf, r["X_val"], r["y_val"], shap_g, names,
                                    xc["permutation_sample"], seed)
        imp.sort_values("mean_abs_shap", ascending=False).to_csv(out / f"{key}_global_importance.csv")
        glob = XA.global_agreement(imp, k)
        glob.to_csv(out / f"{key}_global_agreement.csv", index=False)

        # -- faithfulness (deletion)
        bg = np.median(r["X_train"], axis=0)
        K = min(xc["deletion_max_k"], Xa.shape[1])
        curves = {"TreeSHAP": XA.deletion_curve(proba, Xa, shap_a, bg, K),
                  "LIME": XA.deletion_curve(proba, Xa, lime_a, bg, K),
                  "random": XA.deletion_curve(proba, Xa, None, bg, K, seed)}
        cdf = pd.DataFrame(curves)
        cdf.index.name = "k_removed"
        cdf.to_csv(out / f"{key}_deletion_curves.csv")
        areas = {m: XA.curve_area(c) for m, c in curves.items()}

        # -- LIME stability
        m_st = min(xc["lime_stability_alerts"], len(Xa))
        stab = XA.lime_stability(proba, ref, Xa[:m_st], names, xc["lime_stability_runs"],
                                 xc["lime_num_samples"], k)
        stab.to_csv(out / f"{key}_lime_stability.csv", index=False)

        # -- figures
        fig, ax = plt.subplots(figsize=(5, 3.8))
        for m, c in curves.items():
            ax.plot(range(len(c)), c, marker="o", ms=3, label=f"{m} (area {areas[m]:.3f})",
                    ls="--" if m == "random" else "-")
        ax.set_xlabel("Top-k features replaced by training median")
        ax.set_ylabel("Mean P(threat) on alerts")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"fig_{key}_deletion.png", dpi=200)
        plt.close(fig)

        methods = list(imp.columns)
        mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
        for _, row in glob.iterrows():
            mat.loc[row.method_a, row.method_b] = mat.loc[row.method_b, row.method_a] = row.kendall_tau
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(mat.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu")
        ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(methods)), methods, fontsize=8)
        for i in range(len(methods)):
            for j in range(len(methods)):
                ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(float(mat.iat[i, j])) > 0.6 else "black")
        fig.colorbar(im, ax=ax, label="Kendall τ")
        ax.set_title("Global rank agreement")
        fig.tight_layout()
        fig.savefig(out / f"fig_{key}_global_agreement.png", dpi=200)
        plt.close(fig)

        try:
            import shap
            shap.summary_plot(shap_g, r["X_test"][gs], feature_names=names, show=False, max_display=15)
            plt.gcf().savefig(out / f"fig_{key}_shap_beeswarm.png", dpi=200, bbox_inches="tight")
            plt.close("all")
        except Exception as exc:  # pragma: no cover - plotting only
            print(f"    beeswarm skipped: {exc}")

        summary_blocks += [
            (f"{key}: local agreement TreeSHAP vs LIME (top-{k})", loc_summary.round(3)),
            (f"{key}: global agreement", glob.round(3).set_index(["method_a", "method_b"])),
            (f"{key}: deletion-curve area (lower = more faithful)", pd.Series(areas).round(4).to_frame("area")),
            (f"{key}: LIME stability over {xc['lime_stability_runs']} runs",
             stab.describe().loc[["mean", "std"]].round(3)),
            (f"{key}: time per alert", f"TreeSHAP {t_shap * 1000:.1f} ms; LIME {t_lime * 1000:.1f} ms"),
        ]
        numbers[key] = {
            "local_topk_overlap": float(loc["topk_overlap"].mean()),
            "local_kendall_tau": float(loc["kendall_tau"].mean()),
            "p_drop_shap_topk": float(curves["TreeSHAP"][0] - curves["TreeSHAP"][min(k, K)]),
            "p_drop_lime_topk": float(curves["LIME"][0] - curves["LIME"][min(k, K)]),
            "p_drop_random_topk": float(curves["random"][0] - curves["random"][min(k, K)]),
            "lime_stability_tau": float(stab["mean_pairwise_tau"].mean()),
            "ms_per_alert_shap": t_shap * 1000, "ms_per_alert_lime": t_lime * 1000,
        }

    # ---- transfer of attributions (shared feature space, from E4)
    if transfer and transfer.get("targets"):
        src = transfer["source"]
        rf, names = src["det"].model, src["pre"].feature_names
        gs = rng.choice(len(src["X_test"]), min(2000, len(src["X_test"])), replace=False)
        ranks = {"CIC-IDS2017": np.abs(XA.tree_shap(rf, src["X_test"][gs])).mean(axis=0)}
        for t, tt in transfer["targets"].items():
            ts = rng.choice(len(tt["X_test"]), min(2000, len(tt["X_test"])), replace=False)
            ranks[t] = np.abs(XA.tree_shap(rf, tt["X_test"][ts])).mean(axis=0)
        rk = pd.DataFrame(ranks, index=names)
        rk.to_csv(out / "transfer_global_shap.csv")
        rows = []
        for t in transfer["targets"]:
            a = XA.rank_agreement(rk["CIC-IDS2017"].to_numpy(), rk[t].to_numpy(), k)
            rows.append({"target": t, "kendall_tau": a["kendall_tau"], "topk_overlap": a["topk_overlap"]})
            numbers[f"attribution_transfer_tau_{t}"] = a["kendall_tau"]
        tdf = pd.DataFrame(rows).set_index("target")
        tdf.to_csv(out / "transfer_attribution_agreement.csv")
        top10 = {c: list(rk[c].sort_values(ascending=False).index[:10]) for c in rk}
        summary_blocks.append(("Transfer of global attributions (same CIC-trained model)", tdf.round(3)))
        summary_blocks.append(("Top-10 features by mean |SHAP|", pd.DataFrame(top10)))

    name = "E5.md" if ctx.cfg.get("e5", {}).get("datasets", ["cic"]) else "E5_transfer.md"
    write_md(out / name, "E5 - explanation evaluation", summary_blocks)
    ctx.record("E5", {**ctx.numbers.get("E5", {}), **numbers})


# ====================================================================== CASE
def _noc_set(cfg: dict) -> set[str]:
    path = cfg.get("noc_pseudonyms")
    if not path or not Path(path).exists():
        return set()
    return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()}


def _explain_line(p: float, theta: float, feats: list[tuple[str, float, float]]) -> str:
    """Natural-language explanation from (feature, value, shap) triples, as Fig. 6 renders it."""
    flagged = p >= theta
    pick = [f for f in feats if (f[2] > 0) == flagged][:3] or feats[:3]
    parts = [f"{n} = {v:,.4g} ({'+' if sh >= 0 else ''}{sh:.3f})" for n, v, sh in pick]
    if flagged:
        return f"Flagged as suspicious (p = {p:.3f} ≥ θ* = {theta:.3f}), mainly because " + "; ".join(parts) + "."
    return f"Not flagged (p = {p:.3f} < θ* = {theta:.3f}); strongest evidence against an attack: " + "; ".join(parts) + "."


def run_case(ctx: Context, key: str = "ptma") -> None:
    """Section 5.5: explained alerts from the PTMA-trained model and the effect of the IT label correction."""
    print("CASE: explained PTMA alerts and the effect of analyst confirmation")
    if not ctx.has(key) or not _available(ctx, key):
        return
    out, seed, k = ctx.folder("CASE"), ctx.cfg["seeds"][0], ctx.cfg["xai"]["top_k"]
    noc = _noc_set(ctx.ds_cfg(key))
    if not noc:
        print("  no NOC pseudonyms found: run scripts/noc_pseudonyms.py first (NOC rows will be empty)")
    parts, _ = leakage_safe_parts(ctx, key)

    def noc_mask(fd: FlowData) -> np.ndarray:
        if not noc or "src_ip" not in fd.meta:
            return np.zeros(len(fd), dtype=bool)
        return fd.meta["src_ip"].astype(str).isin(noc).to_numpy()

    # ---- corrected labels (as used in the paper) vs labels before IT confirmation
    r = fit_predict("rf", parts, seed, ctx)
    uncorrected = {}
    for name, fd in parts.items():
        y = fd.y.copy()
        if name != "test":                   # the test partition keeps the confirmed labels
            y[noc_mask(fd)] = 1
        uncorrected[name] = FlowData(fd.name, fd.X, y, fd.cls, fd.meta, list(fd.categorical))
    r_unc = fit_predict("rf", uncorrected, seed, ctx) if noc else None

    te = parts["test"]
    y, s, th = r["y_test"], r["s_test"], r["theta"]
    is_noc = noc_mask(te)
    other_benign = (y == 0) & ~is_noc
    effect = []
    for label, rr in (("corrected labels (after IT confirmation)", r),
                      ("labels before IT confirmation", r_unc)):
        if rr is None:
            continue
        flag = rr["s_test"] >= rr["theta"]
        effect.append({"training labels": label, "theta": rr["theta"],
                       "NOC test flows": int(is_noc.sum()),
                       "NOC flows flagged": int(flag[is_noc].sum()),
                       "share of NOC flows flagged": flag[is_noc].mean() if is_noc.any() else np.nan,
                       "FAR on other benign flows": flag[other_benign].mean(),
                       "recall on attacks": flag[y == 1].mean()})
    eff = pd.DataFrame(effect).set_index("training labels")
    eff.to_csv(out / "case_noc_effect.csv")

    # ---- pick explained flows: a typical detected scan, the hardest NOC flow, a missed attack, a false alarm
    pred = s >= th
    picks = []
    tp = np.where((y == 1) & pred)[0]
    if len(tp):
        picks.append(("detected attack (typical score)", tp[np.argsort(s[tp])[len(tp) // 2]]))
    nn = np.where(is_noc & (y == 0))[0]
    if len(nn):
        picks.append(("NOC monitoring flow (highest score)", nn[np.argmax(s[nn])]))
    fn = np.where((y == 1) & ~pred)[0]
    if len(fn):
        picks.append(("missed attack (highest score)", fn[np.argmax(s[fn])]))
    fp = np.where(other_benign & pred)[0]
    if len(fp):
        picks.append(("false alarm (highest score)", fp[np.argmax(s[fp])]))

    rf, names = r["det"].model, r["pre"].feature_names
    idx = np.array([i for _, i in picks])
    shap_v = XA.tree_shap(rf, r["X_test"][idx])
    rng = np.random.default_rng(seed)
    ref = r["X_train"][rng.choice(len(r["X_train"]), min(5000, len(r["X_train"])), replace=False)]
    lime_v = XA.lime_weights(rf.predict_proba, ref, r["X_test"][idx], names,
                             ctx.cfg["xai"]["lime_num_samples"], seed)
    rows, blocks = [], []
    meta = te.meta.reset_index(drop=True)
    for j, (kind, i) in enumerate(picks):
        order = np.argsort(-np.abs(shap_v[j]))[:k]
        feats = [(names[f], float(r["X_test"][i, f]), float(shap_v[j, f])) for f in order]
        lime_top = [names[f] for f in np.argsort(-np.abs(lime_v[j]))[:k]]
        text = _explain_line(float(s[i]), th, feats)
        src = str(meta.loc[i, "src_ip"]) if "src_ip" in meta else ""
        dport = meta.loc[i, "dst_port"] if "dst_port" in meta else ""
        rows.append({"case": kind, "class": te.cls.iloc[i], "p_threat": s[i], "theta": th,
                     "src_pseudonym": src[:11], "dst_port": dport,
                     "shap_top": "; ".join(f"{n}={v:.4g} ({sh:+.3f})" for n, v, sh in feats),
                     "lime_top": "; ".join(lime_top),
                     "topk_overlap": len(set(lime_top) & {f[0] for f in feats}) / k,
                     "explanation": text})
        tbl = pd.DataFrame(feats, columns=["feature", "value", "TreeSHAP"])
        blocks.append((f"{kind}: {te.cls.iloc[i]}, source {src[:11]}…, port {dport}",
                       f"{text}\n\nLIME top-{k}: {', '.join(lime_top)} "
                       f"(overlap with TreeSHAP {rows[-1]['topk_overlap']:.1f})\n\n" + tbl.to_markdown(index=False)))
    pd.DataFrame(rows).to_csv(out / "case_alerts.csv", index=False)
    write_md(out / "CASE.md", "Case study - PTMA (Section 5.5)",
             [("Effect of the IT label confirmation on a PTMA-trained Random Forest", eff.round(4))] + blocks)
    nums = {"noc_test_flows": int(is_noc.sum())}
    for label, rec in zip(("corrected", "uncorrected"), effect):
        nums[f"noc_flagged_share_{label}"] = rec["share of NOC flows flagged"]
        nums[f"far_other_{label}"] = rec["FAR on other benign flows"]
        nums[f"recall_{label}"] = rec["recall on attacks"]
    ctx.record("CASE", nums)


EXPERIMENTS = {"E1": run_e1, "E2": run_e2, "E3": run_e3}

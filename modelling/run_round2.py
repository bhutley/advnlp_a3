"""Round 2 (pre-registered: docs/plans/2026-10-07-round2-design.md) → results/round2/.

A: R1P (prosody-only audio). B: R3_resid (format-residualised audio). C: the volatility
ladder under naive, ungrouped CV (descriptive). D: ATT (attention over sentences) and its
uniform control UNI. Round 1's saved predictions (results/oof.parquet) supply R1–R4, so
H3/H4 compare against exactly the round-1 results already reported.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402

from attention import load_sentences, make_store, run_attention  # noqa: E402
from cv import outer_splits, ungrouped  # noqa: E402
from data import load_call_data  # noqa: E402
from evaluate import compare, holm, metrics, run_rung  # noqa: E402
from features import TARGETS_DIR, TARGETS_VOL, assemble, task_of  # noqa: E402
from pipelines import ALPHAS, AUDIO_LAYER_GRID, PCA_KS  # noqa: E402
from run_ladder import parse_list  # noqa: E402

CONFIRMATORY = {"H3": ("R1P", "R1"), "H4": ("R3_resid", "R3"), "H5": ("ATT", "UNI")}
EXPLORATORY_VOL = [("R3_resid", "R1"), ("ATT", "R1"), ("UNI", "R4")]
EXPLORATORY_DIR = [("R1P", "R1"), ("R3_resid", "R3")]
DESCRIPTIVE = {"H1_naive": ("R2_naive", "R1_naive"), "H2_naive": ("R4_naive", "R2_naive")}
NAIVE_RUNGS = ["B0", "R1", "R2", "R3", "R4"]
D_WEIGHT_DECAYS = (1e-3, 1e-2, 1e-1, 1.0)
COLORS, REFERENCE, TEXT = ["#2a78d6", "#eb6834"], "#9a9993", "#52514e"


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(".."))
    p.add_argument("--round1", type=Path, default=Path("results"), help="round-1 results dir")
    p.add_argument("--out", type=Path, default=Path("results/round2"))
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--d-repeats", type=int, default=10, help="first N of the repeats, for D")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--targets", default=",".join(TARGETS_VOL + TARGETS_DIR))
    p.add_argument("--quick", action="store_true", help="2 repeats, 1 D repeat")
    p.add_argument("--skip-d", action="store_true")
    p.add_argument("--d-wd", default=",".join(map(str, D_WEIGHT_DECAYS)))
    p.add_argument("--d-seeds", type=int, default=3)
    p.add_argument("--d-max-iter", type=int, default=200)
    p.add_argument("--d-pca", type=int, default=32)
    p.add_argument("--alphas", default=None)  # grid overrides: tests and trials only
    p.add_argument("--ks", default=None)
    p.add_argument("--layers", default=None)
    args = p.parse_args(argv)
    if args.quick:
        args.repeats, args.d_repeats = 2, 1
    args.targets = args.targets.split(",")
    return args


def _comparisons(oof, groups, vol, targets) -> pd.DataFrame:
    rows = []
    for name, (a, b) in CONFIRMATORY.items():
        if not {a, b} <= set(oof["rung"]):
            continue  # e.g. --skip-d
        results = {t: compare(oof, a, b, t, groups) for t in vol}
        adjusted = holm({t: r["p"] for t, r in results.items()})
        rows += [r | {"hypothesis": name, "status": "confirmatory", "p_holm": adjusted[t]}
                 for t, r in results.items()]  # fmt: skip
    for t in targets:
        pairs = EXPLORATORY_VOL if task_of(t) == "regression" else EXPLORATORY_DIR
        rows += [compare(oof, a, b, t, groups) | {"hypothesis": "exploratory", "status": "exploratory",
                                                  "p_holm": np.nan}
                 for a, b in pairs if {a, b} <= set(oof.loc[oof.target == t, "rung"])]  # fmt: skip
    for name, (a, b) in DESCRIPTIVE.items():
        rows += [compare(oof, a, b, t, groups) | {"hypothesis": name, "status": "descriptive", "p_holm": np.nan}
                 for t in vol]  # fmt: skip
    cols = ["status", "hypothesis", "target", "a", "b", "mean_diff", "ci_low", "ci_high", "p",
            "p_holm", "n_calls", "n_firms"]  # fmt: skip
    return pd.DataFrame(rows)[cols]


def _dot_panels(m, targets, series, path, ylabel):
    """One panel per target; series = [(label, [rungs])], drawn side by side per x position."""
    xs = max(len(r) for _, r in series)
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4), dpi=150, squeeze=False)
    for ax, target in zip(axes[0], targets):
        rows = m[(m["target"] == target) & (m["metric"] == "mse")].set_index("rung")
        for s, (label, rungs) in enumerate(series):
            off = (s - (len(series) - 1) / 2) * 0.18
            present = [r for r in rungs if r in rows.index]
            xpos = [rungs.index(r) + off for r in present]
            ax.errorbar(xpos, rows.loc[present, "mean"], yerr=rows.loc[present, "sd"], fmt="o",
                        color=COLORS[s], ms=6, ecolor=TEXT, elinewidth=1, capsize=3, label=label)  # fmt: skip
        if "R1" in rows.index:
            ax.axhline(rows.loc["R1", "mean"], color=REFERENCE, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(range(xs), [r.removesuffix("_naive") for r in series[0][1]], rotation=45, ha="right", fontsize=8)
        ax.set_title(target, loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel(ylabel)
    if len(series) > 1:
        axes[0][0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    grid = {
        "alphas": parse_list(args.alphas, float) if args.alphas else ALPHAS,
        "ks": parse_list(args.ks, int) if args.ks else PCA_KS,
        "layers": parse_list(args.layers, int) if args.layers else AUDIO_LAYER_GRID,
    }
    run = dict(grid, n_jobs=args.n_jobs, inner_seed=args.seed)
    data = load_call_data(args.root)
    groups = data.table.set_index("event_id")["ticker"]
    splits = outer_splits(data.groups, args.splits, args.repeats, args.seed)  # = round 1's splits
    vol = [t for t in args.targets if task_of(t) == "regression"]

    def timed(label, fn):
        t0 = time.perf_counter()
        out = fn()
        print(f"  {label}: {time.perf_counter() - t0:.0f}s", flush=True)
        return out

    frames, weight_frames = [], []
    for target in args.targets:  # A, B
        for rung in ("R1P", "R3_resid"):
            frames.append(timed(f"{target} {rung}",
                                lambda: run_rung(data, rung, target, splits, **run)).assign(cv="grouped"))  # fmt: skip

    naive = replace(data, groups=ungrouped(data.groups))  # C
    naive_splits = outer_splits(naive.groups, args.splits, args.repeats, args.seed)
    for target in vol:
        for rung in NAIVE_RUNGS:
            oof = timed(f"{target} {rung} naive", lambda: run_rung(naive, rung, target, naive_splits, **run))
            frames.append(oof.assign(rung=f"{rung}_naive", cv="naive"))

    d_meta = {"weight_decays": list(parse_list(args.d_wd, float)), "seeds": args.d_seeds,
              "max_iter": args.d_max_iter, "pca_dim": args.d_pca}  # fmt: skip
    if not args.skip_d:  # D
        texts, audios, indices = load_sentences(args.root, data.table["event_id"], return_idx=True)
        d_splits = [s for s in splits if s.repeat < args.d_repeats]
        for target in vol:
            store = make_store(texts, audios, indices, assemble(data, ["past", "format"], target)[0])
            for rung in ("ATT", "UNI"):
                oof, w = timed(f"{target} {rung}", lambda: run_attention(
                    data, store, rung, target, d_splits, d_meta["weight_decays"], args.seed,
                    args.d_seeds, args.d_max_iter, args.d_pca))  # fmt: skip
                frames.append(oof.assign(cv="grouped"))
                if w is not None:
                    weight_frames.append(w)
    oof2 = pd.concat(frames, ignore_index=True)

    round1 = pd.read_parquet(args.round1 / "oof.parquet")
    round1 = round1[round1["rung"].isin(["R1", "R2", "R3", "R4"]) & round1["target"].isin(args.targets)]
    combined = pd.concat([round1.assign(cv="grouped"), oof2], ignore_index=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    oof2.to_parquet(args.out / "oof.parquet", index=False)
    m = metrics(combined)
    m.to_csv(args.out / "metrics.csv", index=False)
    _comparisons(combined, groups, vol, args.targets).to_csv(args.out / "comparisons.csv", index=False)
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame(
        columns=["event_id", "modality", "rank", "sent_idx", "weight", "target", "repeat", "fold"])  # fmt: skip
    weights.to_parquet(args.out / "attention_weights.parquet", index=False)
    if vol:
        _dot_panels(m, vol, [("", ["R1", "R2", "R3", "R4", "R1P", "R3_resid", "UNI", "ATT"])],
                    args.out / "figures" / "mse_round2.png", "Out-of-fold MSE (dashed: R1)")  # fmt: skip
        _dot_panels(m, vol, [("grouped (firm)", ["R1", "R2", "R3", "R4"]),
                             ("naive (ungrouped)", ["R1_naive", "R2_naive", "R3_naive", "R4_naive"])],
                    args.out / "figures" / "naive_vs_grouped.png", "Out-of-fold MSE")  # fmt: skip

    meta = {
        "design": "docs/plans/2026-10-07-round2-design.md", "targets": args.targets,
        "repeats": args.repeats, "d_repeats": 0 if args.skip_d else args.d_repeats,
        "splits": args.splits, "seed": args.seed, "grid": {k: list(v) for k, v in grid.items()},
        "d": d_meta, "round1": str(args.round1), "n_calls": len(data.table),
        "sklearn": sklearn.__version__, "torch": torch.__version__,
        "runtime_s": round(time.perf_counter() - start, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }  # fmt: skip
    (args.out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwritten to {args.out} ({meta['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

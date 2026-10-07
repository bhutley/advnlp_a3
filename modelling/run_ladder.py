"""Run the ablation ladder and write results/.

For every target: B0, R1, R2, R3, R4, R4_nopast (finbert_tone + hubert), plus R4 with the
other text and audio models (robustness). Then the pre-registered comparisons (H1: R2 vs
R1, H2: R4 vs R2, Holm-corrected across the volatility horizons), exploratory
comparisons, and two leakage sanity checks: a permutation null and the effect of
ungrouped CV.
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

from cv import outer_splits  # noqa: E402
from data import load_call_data  # noqa: E402
from evaluate import compare, holm, metrics, run_rung  # noqa: E402
from features import RUNGS, TARGETS_DIR, TARGETS_VOL, task_of  # noqa: E402
from pipelines import ALPHAS, AUDIO_LAYER_GRID, PCA_KS  # noqa: E402

DEFAULT_MODELS = ("finbert_tone", "hubert")
ROBUSTNESS = {"R4_prosus_finbert": ("prosus_finbert", "hubert"), "R4_wav2vec2": ("finbert_tone", "wav2vec2")}
HYPOTHESES = {"H1": ("R2", "R1"), "H2": ("R4", "R2")}  # (a, b): a should beat b
EXPLORATORY = [("R1", "B0"), ("R3", "R1"), ("R4", "R1"), ("R4", "R3"), ("R4_nopast", "B0")]
EXPLORATORY_DIR = [("R2", "R1"), ("R3", "R1"), ("R4", "R2")]
COLOR, REFERENCE, TEXT = "#2a78d6", "#9a9993", "#52514e"


def parse_list(text: str, cast) -> tuple:
    return tuple(None if v.strip().lower() == "none" else cast(v) for v in text.split(","))


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".."), help="repository root")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--perm-repeats", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--targets", default=",".join(TARGETS_VOL + TARGETS_DIR))
    parser.add_argument("--quick", action="store_true", help="2 repeats, 2 permutations of 1 repeat")
    # Grid overrides, for tests and trials only; the defaults are the design's grid.
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--ks", default=None)
    parser.add_argument("--layers", default=None)
    args = parser.parse_args(argv)
    if args.quick:
        args.repeats, args.permutations, args.perm_repeats = 2, 2, 1
    args.targets = args.targets.split(",")
    return args


def _mse(oof: pd.DataFrame) -> float:
    return float(((oof["y"] - oof["y_hat"]) ** 2).mean())


def comparisons(oof: pd.DataFrame, targets: list[str], groups: pd.Series) -> pd.DataFrame:
    rows = []
    vol = [t for t in targets if task_of(t) == "regression"]
    for name, (a, b) in HYPOTHESES.items():
        results = {t: compare(oof, a, b, t, groups) for t in vol}
        adjusted = holm({t: r["p"] for t, r in results.items()}) if results else {}
        rows += [r | {"hypothesis": name, "p_holm": adjusted[t]} for t, r in results.items()]
    for t in targets:
        pairs = EXPLORATORY if task_of(t) == "regression" else EXPLORATORY_DIR
        rows += [compare(oof, a, b, t, groups) | {"hypothesis": "exploratory", "p_holm": np.nan}
                 for a, b in pairs]  # fmt: skip
    cols = ["hypothesis", "target", "a", "b", "mean_diff", "ci_low", "ci_high", "p", "p_holm",
            "n_calls", "n_firms"]  # fmt: skip
    return pd.DataFrame(rows)[cols]


def plot_mse(m: pd.DataFrame, targets: list[str], path: Path) -> None:
    """MSE (mean ± SD over repeats) of the fitted rungs, one panel per volatility horizon.

    A dot plot, not bars: the differences are small relative to the values, so the axis
    can't start at zero, and bars on a truncated axis would misstate the comparison.
    """
    rungs = [r for r in RUNGS if r != "B0"] + list(ROBUSTNESS)
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 4), dpi=150, squeeze=False)
    for ax, target in zip(axes[0], targets):
        rows = m[(m["target"] == target) & (m["metric"] == "mse")].set_index("rung").loc[rungs]
        ax.errorbar(range(len(rungs)), rows["mean"], yerr=rows["sd"], fmt="o", color=COLOR,
                    ms=6, ecolor=TEXT, elinewidth=1, capsize=3)  # fmt: skip
        ax.axhline(rows.loc["R1", "mean"], color=REFERENCE, lw=1, ls=(0, (3, 3)))
        ax.set_xticks(range(len(rungs)), rungs, rotation=45, ha="right", fontsize=8)
        lo = rows["mean"].min() - 2 * rows["sd"].max()
        ax.set_ylim(max(0, lo), rows["mean"].max() + 2 * rows["sd"].max())
        ax.set_title(target, loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Out-of-fold MSE (dashed: R1)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_permutation(sanity: pd.DataFrame, observed: dict, path: Path) -> None:
    perm = sanity[sanity["check"] == "permutation"]
    targets = list(dict.fromkeys(perm["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.4), dpi=150, squeeze=False)
    for ax, target in zip(axes[0], targets):
        ax.hist(perm.loc[perm["target"] == target, "gain"], bins=10, color=COLOR, alpha=0.8)
        ax.axvline(0, color=REFERENCE, lw=1)
        ax.axvline(observed[target], color=TEXT, lw=2, ls="--", label="observed")
        ax.set_title(f"{target}: R1 − R4 MSE", loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
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
    data = load_call_data(args.root, *DEFAULT_MODELS)
    splits = outer_splits(data.groups, args.splits, args.repeats, args.seed)
    groups = data.table.set_index("event_id")["ticker"]
    vol_targets = [t for t in args.targets if task_of(t) == "regression"]
    print(f"{len(data.table)} calls, {len(set(data.groups))} firms; {len(splits)} outer folds")

    def timed(label, fn):
        t0 = time.perf_counter()
        result = fn()
        print(f"  {label}: {time.perf_counter() - t0:.0f}s", flush=True)
        return result

    alternatives = {label: load_call_data(args.root, *models) for label, models in ROBUSTNESS.items()}
    frames = []
    for target in args.targets:
        for rung in RUNGS:
            frames.append(timed(f"{target} {rung}", lambda: run_rung(data, rung, target, splits, **run)))
        for label, alt in alternatives.items():
            oof = timed(f"{target} {label}", lambda: run_rung(alt, "R4", target, splits, **run))
            frames.append(oof.assign(rung=label))
    oof = pd.concat(frames, ignore_index=True)

    sanity_rows = []
    perm_splits = outer_splits(data.groups, args.splits, args.perm_repeats, args.seed + 1)
    for target in vol_targets:
        for p in range(args.permutations):
            r1 = _mse(run_rung(data, "R1", target, perm_splits, permute=p, **run))
            r4 = _mse(run_rung(data, "R4", target, perm_splits, permute=p, **run))
            sanity_rows.append({"check": "permutation", "target": target, "perm": p,
                                "mse_R1": r1, "mse_R4": r4, "gain": r1 - r4})  # fmt: skip
        ungrouped = replace(data, groups=data.table["event_id"].to_numpy())
        r3 = run_rung(ungrouped, "R3", target, outer_splits(ungrouped.groups, args.splits, 1, args.seed), **run)
        grouped = oof[(oof["target"] == target) & (oof["rung"] == "R3") & (oof["repeat"] == 0)]
        sanity_rows.append({"check": "grouping", "target": target,
                            "mse_grouped": _mse(grouped), "mse_ungrouped": _mse(r3)})  # fmt: skip
        print(f"  {target} sanity checks done ({time.perf_counter() - start:.0f}s)", flush=True)
    sanity = pd.DataFrame(sanity_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    oof.to_parquet(args.out / "oof.parquet", index=False)
    m = metrics(oof)
    m.to_csv(args.out / "metrics.csv", index=False)
    comparisons(oof, args.targets, groups).to_csv(args.out / "comparisons.csv", index=False)
    sanity.to_csv(args.out / "sanity.csv", index=False)
    if vol_targets:
        plot_mse(m, vol_targets, args.out / "figures" / "mse_by_rung.png")
        observed = {t: _mse(oof[(oof.target == t) & (oof.rung == "R1")])
                    - _mse(oof[(oof.target == t) & (oof.rung == "R4")]) for t in vol_targets}  # fmt: skip
        plot_permutation(sanity, observed, args.out / "figures" / "permutation_null.png")

    meta = {
        "targets": args.targets, "repeats": args.repeats, "splits": args.splits,
        "permutations": args.permutations, "perm_repeats": args.perm_repeats, "seed": args.seed,
        "grid": {k: list(v) for k, v in grid.items()}, "models": list(DEFAULT_MODELS),
        "robustness": {k: list(v) for k, v in ROBUSTNESS.items()},
        "n_calls": len(data.table), "n_firms": int(len(set(data.groups))),
        "sklearn": sklearn.__version__, "runtime_s": round(time.perf_counter() - start, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }  # fmt: skip
    (args.out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwritten to {args.out} ({meta['runtime_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

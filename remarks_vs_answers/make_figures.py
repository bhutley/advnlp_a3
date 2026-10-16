"""Round-3 figures from results/round3/metrics.csv → results/round3/figures/.

- mse_round3.png: out-of-fold MSE (mean ± SD over repeats) by rung, one panel per
  volatility horizon; dashed line = S1, the baseline every round-3 rung extends.
- auc_round3.png: direction AUC by rung, one panel per CAR window; dashed line = S1.

Dot plots, as in the paper's Figures 2–3: the differences are small relative to the
values, so a bar's zero baseline would hide them. Reads only saved metrics.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLOR, REFERENCE, TEXT = "#2a78d6", "#9a9993", "#52514e"  # the paper's figure colours
RUNGS = ["S1", "R1", "R2", "TR", "TA", "TB", "PR", "PA", "PD", "AR", "AA"]
VOL = {"vol_3": "3 days", "vol_7": "7 days", "vol_15": "15 days", "vol_30": "30 days"}
DIR = {"dir_0_1": "CAR [0, +1]", "dir_0_5": "CAR [0, +5]", "dir_2_5": "CAR [+2, +5]"}


def dot_panels(m: pd.DataFrame, metric: str, panels: dict, ylabel: str, path: Path) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), dpi=200, squeeze=False)
    for ax, (target, title) in zip(axes[0], panels.items()):
        rows = m[(m["target"] == target) & (m["metric"] == metric)].set_index("rung").reindex(RUNGS)
        x = range(len(RUNGS))
        ax.errorbar(x, rows["mean"], yerr=rows["sd"], fmt="o", color=COLOR, ms=5, ecolor=TEXT,
                    elinewidth=1, capsize=2.5)  # fmt: skip
        ax.axhline(rows.loc["S1", "mean"], color=REFERENCE, lw=1, ls=(0, (3, 3)))
        for boundary in (2.5, 5.5, 8.5):  # baselines | text | prosody | audio
            ax.axvline(boundary, color="#e6e5e0", lw=0.8, zorder=0)
        ax.set_xticks(list(x), RUNGS, rotation=90, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5, colors=TEXT)
        ax.set_title(title, loc="left", fontsize=9, color=TEXT)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c9c8c2")
    axes[0][0].set_ylabel(ylabel, fontsize=8.5, color=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results/round3"))
    args = p.parse_args(argv)
    m = pd.read_csv(args.results / "metrics.csv")
    out = args.results / "figures"
    out.mkdir(exist_ok=True)
    dot_panels(m, "mse", VOL, "Out-of-fold MSE (dashed: S1)", out / "mse_round3.png")
    dot_panels(m, "auc", DIR, "AUC (dashed: S1)", out / "auc_round3.png")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

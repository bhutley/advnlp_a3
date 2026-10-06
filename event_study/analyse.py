"""Summarise the event study: day-0 diagnostic, CAAR and volatility tables, CAAR plots.

Reads data/events.csv and data/ar_daily.csv (from build_events.py), prints markdown
tables, writes them to <out-dir>/summary.md and saves PNG figures there. Only events with
status 'ok' are used.
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy import stats  # noqa: E402

from returns import CAR_WINDOWS, VOL_HORIZONS  # noqa: E402

WINSOR = 0.01  # CARs are winsorized at the 1st and 99th percentiles

# Validated categorical palette (dataviz reference, light mode), assigned in fixed order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"
GRID, REFERENCE = "#e6e5e1", "#9a9993"


def ok_events(events: pd.DataFrame) -> pd.DataFrame:
    return events[events["status"] == "ok"]


def winsorize(s: pd.Series, q: float) -> pd.Series:
    """Clip s to its q and 1−q quantiles. NaNs stay NaN; q=0 returns s unchanged."""
    if q == 0:
        return s
    return s.clip(s.quantile(q), s.quantile(1 - q))


def _t_test(x: pd.Series) -> tuple[float, float]:
    if len(x) < 2:
        return math.nan, math.nan
    result = stats.ttest_1samp(x, 0.0)
    return float(result.statistic), float(result.pvalue)


def caar_table(events: pd.DataFrame, model: str = "mm", winsor: float = WINSOR) -> pd.DataFrame:
    """Per CAR window: n, mean, t, p, pct_pos, and mean_w, t_w, p_w after winsorizing.

    The t-test is a cross-sectional one-sample test against 0. NaNs are dropped per window.
    """
    rows = {}
    for window in CAR_WINDOWS:
        x = ok_events(events)[f"car_{model}_{window}"].dropna()
        xw = winsorize(x, winsor)
        t, p = _t_test(x)
        t_w, p_w = _t_test(xw)
        rows[window] = {
            "n": len(x),
            "mean": x.mean(),
            "t": t,
            "p": p,
            "pct_pos": 100 * (x > 0).mean(),
            "mean_w": xw.mean(),
            "t_w": t_w,
            "p_w": p_w,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("window")


def vol_table(events: pd.DataFrame) -> pd.DataFrame:
    """Per horizon n: mean post-call vol_n, pre-call vol_pre_n, and a paired t-test."""
    rows = {}
    for n in VOL_HORIZONS:
        pair = ok_events(events)[[f"vol_{n}", f"vol_pre_{n}"]].dropna()
        post, pre = pair[f"vol_{n}"], pair[f"vol_pre_{n}"]
        result = stats.ttest_rel(post, pre)
        rows[n] = {
            "n": len(pair),
            "post": post.mean(),
            "pre": pre.mean(),
            "diff": (post - pre).mean(),
            "t": float(result.statistic),
            "p": float(result.pvalue),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("horizon")


def abs_ar_by_day(
    ar_daily: pd.DataFrame, events: pd.DataFrame, days: tuple[int, int] = (-5, 5)
) -> pd.DataFrame:
    """Mean |ar_mm| by timing (rows) and rel_day (columns): the day-0 diagnostic."""
    ar = ar_daily[ar_daily["rel_day"].between(*days)].merge(
        ok_events(events)[["event_id", "timing"]], on="event_id"
    )
    ar = ar.assign(abs_ar=ar["ar_mm"].abs())
    return ar.pivot_table(index="timing", columns="rel_day", values="abs_ar", aggfunc="mean")


def caar_paths(
    ar_daily: pd.DataFrame, events: pd.DataFrame, group_col: str | None = None, model: str = "mm"
) -> pd.DataFrame:
    """Mean cumulative AR from the first event day, per group.

    Only events with every day present are used, so a curve's composition is the same on
    every day. Returns columns group, rel_day, caar, se (standard error of the mean), n.
    """
    wide = ar_daily.pivot(index="rel_day", columns="event_id", values=f"ar_{model}")
    wide = wide.dropna(axis="columns")
    car = wide.cumsum()
    meta = ok_events(events).set_index("event_id")
    groups = meta[group_col] if group_col else pd.Series("All", index=meta.index)

    frames = []
    for group in sorted(groups.unique()):
        ids = [e for e in car.columns if groups.get(e) == group]
        g = car[ids]
        n = g.shape[1]
        frames.append(
            pd.DataFrame(
                {
                    "group": group,
                    "rel_day": g.index,
                    "caar": g.mean(axis=1).to_numpy(),
                    "se": (g.std(axis=1, ddof=1) / math.sqrt(n)).to_numpy(),
                    "n": n,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12)
    ax.set_xlabel("Trading days relative to day 0", color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))


def _spread(ys: list[float], min_gap: float) -> list[float]:
    """Nudge sorted label positions apart so neighbours are at least min_gap apart."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for prev, cur in zip(order, order[1:]):
        out[cur] = max(out[cur], out[prev] + min_gap)
    return out


def plot_caar(paths: pd.DataFrame, title: str, out: Path) -> None:
    """One CAAR line per group with a 95% band.

    Several groups get a legend plus direct labels at the right end; a single group gets
    neither, and its n goes in the title.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.axhline(0, color=REFERENCE, linewidth=1)
    ax.axvline(0, color=REFERENCE, linewidth=1, linestyle=(0, (3, 3)))
    groups = list(paths.groupby("group", sort=True))
    for color, (group, g) in zip(SERIES_COLORS, groups):
        ax.fill_between(
            g["rel_day"], g["caar"] - 1.96 * g["se"], g["caar"] + 1.96 * g["se"],
            color=color, alpha=0.15, linewidth=0,
        )  # fmt: skip
        ax.plot(g["rel_day"], g["caar"], color=color, linewidth=2,
                label=f"{group} (n={g['n'].iloc[0]})")  # fmt: skip
    if len(groups) == 1:
        title = f"{title}, n={groups[0][1]['n'].iloc[0]}"
    else:
        lo, hi = ax.get_ylim()
        ends = [g["caar"].iloc[-1] for _, g in groups]
        for (group, g), y in zip(groups, _spread(ends, 0.04 * (hi - lo))):
            ax.annotate(
                str(group), (g["rel_day"].iloc[-1], y), xytext=(6, 0),
                textcoords="offset points", va="center", color=TEXT_PRIMARY, fontsize=9,
            )  # fmt: skip
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY, loc="upper left")
    _style(ax, title, "Mean cumulative abnormal return")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_abs_ar(table: pd.DataFrame, out: Path) -> None:
    """Mean |AR| by relative day, one line per timing group."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.axvline(0, color=REFERENCE, linewidth=1, linestyle=(0, (3, 3)))
    for color, (timing, row) in zip(SERIES_COLORS, table.sort_index().iterrows()):
        ax.plot(row.index, row.to_numpy(), color=color, linewidth=2, marker="o",
                markersize=5, label=timing)  # fmt: skip
    _style(ax, "Day-0 diagnostic: mean |abnormal return| by timing", "Mean |AR| (market model)")
    ax.set_xticks(list(table.columns))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """df as a markdown table, index first. NaN prints as an empty cell."""

    def cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if math.isnan(v) else floatfmt.format(v)
        return str(v)

    header = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    # itertuples keeps each column's dtype; iterrows would upcast int columns to float.
    for idx, row in zip(df.index, df.itertuples(index=False, name=None)):
        lines.append("| " + " | ".join([str(idx)] + [cell(v) for v in row]) + " |")
    return "\n".join(lines)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/figures"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    events = pd.read_csv(args.in_dir / "events.csv")
    ar_daily = pd.read_csv(args.in_dir / "ar_daily.csv")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    abs_ar = abs_ar_by_day(ar_daily, events)
    sections = [
        f"# Event study summary\n\nEvents with status ok: {len(ok_events(events))}. "
        f"CARs winsorized at {WINSOR:.0%} / {1 - WINSOR:.0%} in the *_w columns.",
        "## Day-0 diagnostic: mean |AR| (market model)\n\n" + markdown_table(abs_ar),
        "## CAAR, market model\n\n" + markdown_table(caar_table(events, "mm")),
        "## CAAR, market-adjusted\n\n" + markdown_table(caar_table(events, "ma")),
        "## Log volatility, post-call vs pre-call\n\n" + markdown_table(vol_table(events)),
    ]
    summary = "\n\n".join(sections) + "\n"
    print(summary)
    (args.out_dir / "summary.md").write_text(summary)

    plot_abs_ar(abs_ar, args.out_dir / "day0_diagnostic.png")
    for model, name in (("mm", "market model"), ("ma", "market-adjusted")):
        plot_caar(
            caar_paths(ar_daily, events, model=model),
            f"CAAR around earnings calls ({name})",
            args.out_dir / f"caar_all_{model}.png",
        )
    plot_caar(
        caar_paths(ar_daily, events, group_col="timing"),
        "CAAR by call timing (market model)",
        args.out_dir / "caar_by_timing_mm.png",
    )
    print(f"figures written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

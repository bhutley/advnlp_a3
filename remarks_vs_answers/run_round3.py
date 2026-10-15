"""Round 3 (pre-registered: docs/plans/2026-10-13-round3-design.md) → results/round3/.

Every rung on every target with 5 × 10 firm-grouped CV on the 457 calls, then the
comparisons: H6–H9 (confirmatory, Holm across the volatility horizons), and the
exploratory ones listed in round3.py.

--quick is the registered timing trial: 2 repeats, and it writes only
results/round3_trial/timing.json (seconds per rung and target, and the projected full
run). It saves no predictions, metrics or comparisons, so nothing in it can be read
against the targets. The runtime rule then sets --audio-repeats (10, or 5 if the
projection exceeds 8 hours).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

import round3 as r3
from cv import outer_splits  # ../modelling, via round3
from evaluate import compare, holm, metrics

LIMIT_HOURS = 8.0  # the runtime rule


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(".."))
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("results/round3"))
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--audio-repeats", type=int, default=10, help="repeats for AR and AA (runtime rule)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--targets", default=",".join(r3.TARGETS_VOL + r3.TARGETS_DIR))
    p.add_argument("--rungs", default=",".join(r3.RUNGS))
    p.add_argument("--quick", action="store_true", help="timing trial: 2 repeats, timings only")
    args = p.parse_args(argv)
    args.targets, args.rungs = args.targets.split(","), args.rungs.split(",")
    return args


def comparisons(oof: pd.DataFrame, groups: pd.Series, targets: list[str]) -> pd.DataFrame:
    vol = [t for t in targets if r3.TASKS[t] == "regression"]
    have = lambda a, b, t: {a, b} <= set(oof.loc[oof["target"] == t, "rung"])  # noqa: E731
    rows = []
    for name, (a, b) in r3.CONFIRMATORY.items():
        results = {t: compare(oof, a, b, t, groups) for t in vol if have(a, b, t)}
        adjusted = holm({t: r["p"] for t, r in results.items()})
        rows += [r | {"status": "confirmatory", "hypothesis": name, "p_holm": adjusted[t]} for t, r in results.items()]
    for t in targets:
        pairs = r3.EXPLORATORY_VOL if r3.TASKS[t] == "regression" else r3.EXPLORATORY_DIR
        rows += [compare(oof, a, b, t, groups) | {"status": "exploratory", "hypothesis": "exploratory", "p_holm": np.nan}
                 for a, b in pairs if have(a, b, t)]  # fmt: skip
    cols = ["status", "hypothesis", "target", "a", "b", "mean_diff", "ci_low", "ci_high", "p", "p_holm",
            "n_calls", "n_firms"]  # fmt: skip
    return pd.DataFrame(rows, columns=cols)


def main(argv=None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    repeats = 2 if args.quick else args.repeats
    data = r3.load_round3(args.root, args.data)
    splits = outer_splits(data.groups, args.splits, repeats, args.seed)
    audio_splits = [s for s in splits if s.repeat < (repeats if args.quick else args.audio_repeats)]
    print(f"{len(data.table)} calls, {len(set(data.groups))} firms; {len(splits)} outer folds", flush=True)

    frames, timing = [], []
    for target in args.targets:
        for rung in args.rungs:
            t0 = time.perf_counter()
            use = audio_splits if rung in r3.AUDIO_RUNGS else splits
            oof = r3.run_rung(data, rung, target, use, inner_seed=args.seed, n_jobs=args.n_jobs)
            seconds = time.perf_counter() - t0
            timing.append({"target": target, "rung": rung, "repeats": len(use) // args.splits, "seconds": seconds})
            print(f"  {target} {rung}: {seconds:.0f}s", flush=True)
            if not args.quick:
                frames.append(oof)
            del oof

    if args.quick:
        t = pd.DataFrame(timing)
        per_repeat = t["seconds"] / t["repeats"]
        full = float((per_repeat * np.where(t["rung"].isin(r3.AUDIO_RUNGS), 10, args.repeats)).sum()) / 3600
        audio5 = float((per_repeat * np.where(t["rung"].isin(r3.AUDIO_RUNGS), 5, args.repeats)).sum()) / 3600
        decision = {"projected_hours_all_10": round(full, 2), "projected_hours_audio_5": round(audio5, 2),
                    "limit_hours": LIMIT_HOURS, "audio_repeats": 10 if full <= LIMIT_HOURS else 5}  # fmt: skip
        out = args.out.parent / "round3_trial"
        out.mkdir(parents=True, exist_ok=True)
        (out / "timing.json").write_text(json.dumps({"decision": decision, "timing": timing}, indent=2) + "\n")
        print(json.dumps(decision, indent=2))
        print(f"timings only, written to {out}/timing.json; no predictions were kept")
        return 0

    oof = pd.concat(frames, ignore_index=True)
    groups = data.table.set_index("event_id")["ticker"]
    args.out.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.out / "oof.parquet", index=False)
    metrics(oof).to_csv(args.out / "metrics.csv", index=False)
    comparisons(oof, groups, args.targets).to_csv(args.out / "comparisons.csv", index=False)
    meta = {
        "design": "docs/plans/2026-10-13-round3-design.md", "targets": args.targets, "rungs": args.rungs,
        "repeats": args.repeats, "audio_repeats": args.audio_repeats, "splits": args.splits, "seed": args.seed,
        "n_calls": len(data.table), "n_firms": int(len(set(data.groups))), "sklearn": sklearn.__version__,
        "timing": timing, "runtime_s": round(time.perf_counter() - start, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }  # fmt: skip
    (args.out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwritten to {args.out} ({meta['runtime_s']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

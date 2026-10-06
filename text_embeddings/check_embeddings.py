"""Acceptance checks for the FinBERT outputs; writes <data-dir>/acceptance.md.

1. Direction: fixed positive and negative sentences must score the right way round.
2. MPS vs CPU: the first sentences embedded on both devices must agree closely.
3. Cross-model: agreement of call tone between the two models (reported only).
4. First look at the targets: Spearman correlation of call tone with event-study
   targets (reported only; never use it to choose a model or pooling).

Exits with 1 if check 1 or 2 fails.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

import embed
from transcripts import read_sentences

POSITIVE = [
    "Revenue grew strongly and we raised our full-year guidance.",
    "Margins expanded to a record level this quarter.",
    "Demand was robust across all of our regions.",
    "We delivered double-digit earnings growth and returned cash to shareholders.",
]
NEGATIVE = [
    "We missed expectations and cut our outlook for the year.",
    "Sales declined sharply as demand weakened.",
    "Margins contracted due to higher costs and pricing pressure.",
    "We recorded a significant impairment charge and lowered guidance.",
]
DEVICE_SENTENCES = 200
DEVICE_TOLERANCE = 1e-2
TARGETS = ["car_mm_0_1", "car_mm_0_5", "vol_3", "vol_30"]


def direction_failures(texts: list[str], probs: np.ndarray, positive: bool) -> list[str]:
    """Texts whose p_pos vs p_neg ordering is wrong. probs columns follow PROB_COLUMNS."""
    p_pos, p_neg = probs[:, 0], probs[:, 1]
    wrong = p_pos <= p_neg if positive else p_neg <= p_pos
    return [t for t, w in zip(texts, wrong) if w]


def correlations(x: pd.Series, y: pd.Series) -> dict:
    """Pearson and Spearman correlation over rows where both are present."""
    both = pd.concat([x, y], axis=1).dropna()
    a, b = both.iloc[:, 0], both.iloc[:, 1]
    spearman = stats.spearmanr(a, b)
    return {
        "n": len(both),
        "pearson": float(stats.pearsonr(a, b).statistic),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def target_table(calls: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of call tone and frac_neg with each target, over ok events."""
    joined = calls.merge(events[events["status"] == "ok"], on="event_id")
    rows = []
    for feature in ("tone", "frac_neg"):
        for target in TARGETS:
            c = correlations(joined[feature], joined[target])
            rows.append({"feature": feature, "target": target, "n": c["n"],
                         "spearman": c["spearman"], "p": c["spearman_p"]})  # fmt: skip
    return pd.DataFrame(rows)


def markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--events", type=Path, default=Path("../event_study/data/events.csv"))
    parser.add_argument(
        "--dataset", type=Path, default=Path.home() / "var/data/earnings_calls/ACL19_Release"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cpu = torch.device("cpu")
    mps = embed.pick_device("auto")
    _, sentences = read_sentences(args.dataset, limit_calls=2)
    device_texts = sentences["text"].tolist()[:DEVICE_SENTENCES]

    failed = False
    direction_rows, device_rows = [], []
    for key in sorted(embed.MODELS):
        model, tokenizer = embed.load_model(key, cpu)
        for texts, positive in ((POSITIVE, True), (NEGATIVE, False)):
            probs = embed.embed_sentences(texts, model, tokenizer, cpu).probs
            wrong = direction_failures(texts, probs, positive)
            failed |= bool(wrong)
            direction_rows.append({
                "model": key, "set": "positive" if positive else "negative",
                "correct": f"{len(texts) - len(wrong)}/{len(texts)}",
                "wrong": "; ".join(wrong) or "-",
            })  # fmt: skip

        on_cpu = embed.embed_sentences(device_texts, model, tokenizer, cpu)
        on_mps = embed.embed_sentences(device_texts, model.to(mps), tokenizer, mps)
        emb_diff = float(np.abs(on_cpu.emb - on_mps.emb).max())
        prob_diff = float(np.abs(on_cpu.probs - on_mps.probs).max())
        ok = max(emb_diff, prob_diff) < DEVICE_TOLERANCE
        failed |= not ok
        device_rows.append({"model": key, "device": str(mps), "n": len(device_texts),
                            "max_emb_diff": emb_diff, "max_prob_diff": prob_diff,
                            "pass": "yes" if ok else "NO"})  # fmt: skip

    calls = {k: pd.read_csv(args.data_dir / k / "calls.csv") for k in sorted(embed.MODELS)}
    tone = pd.concat({k: c.set_index("event_id")["tone"] for k, c in calls.items()}, axis=1)
    cross = correlations(tone["finbert_tone"], tone["prosus_finbert"])
    events = pd.read_csv(args.events)
    targets = pd.concat(
        [target_table(c, events).assign(model=k) for k, c in calls.items()], ignore_index=True
    )[["model", "feature", "target", "n", "spearman", "p"]]

    sections = [
        "# FinBERT acceptance checks",
        "## 1. Direction (must pass)\n\n" + markdown(pd.DataFrame(direction_rows)),
        f"## 2. MPS vs CPU (must pass, tolerance {DEVICE_TOLERANCE})\n\n"
        + markdown(pd.DataFrame(device_rows), "{:.2e}"),
        "## 3. Cross-model agreement of call tone (reported only)\n\n"
        + markdown(pd.DataFrame([cross])),
        "## 4. First look at the targets (reported only)\n\n"
        "Spearman correlation over `ok` events. This is a sanity check, not a basis for "
        "choosing a model or pooling.\n\n" + markdown(targets),
    ]
    report = "\n\n".join(sections) + "\n"
    print(report)
    (args.data_dir / "acceptance.md").write_text(report)
    print("FAILED" if failed else "all must-pass checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

"""Acceptance checks for the remarks/answers labels → data/acceptance.md.

Must pass (exit 1 otherwise): the release matched the saved outputs, the labels cover
exactly the saved sentences, each call's sections come in order, every modelling call
has prepared remarks, and calls.csv agrees with sentences.csv. Reported: rule use,
section sizes, how many calls have both sections, and the hand audit once it is marked.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

from sections import SECTIONS

ORDER = {"remarks": 0, "answers": 1, "closing": 2}  # headings may sit at either boundary


def table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion k/n."""
    if n == 0:
        return math.nan, math.nan
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def out_of_order(sentences: pd.DataFrame) -> list[str]:
    """Calls whose sections are not remarks → answers → closing (headings ignored)."""
    bad = []
    for event_id, g in sentences[sentences["section"] != "heading"].groupby("event_id"):
        rank = g.sort_values("sent_idx")["section"].map(ORDER)
        if not rank.is_monotonic_increasing:
            bad.append(event_id)
    return bad


def must_pass(data: Path, root: Path, expected_calls: int = 488) -> pd.DataFrame:
    meta = json.loads((data / "meta.json").read_text())
    sentences = pd.read_csv(data / "sentences.csv")
    calls = pd.read_csv(data / "calls.csv")
    saved = pd.read_csv(root / "audio_embeddings/data/praat/sentences.csv", usecols=["event_id", "sent_idx"])
    keys = sentences[["event_id", "sent_idx"]].merge(saved, how="outer", indicator=True)["_merge"]
    counts = sentences.pivot_table(index="event_id", columns="section", values="sent_idx", aggfunc="size",
                                   fill_value=0).reindex(columns=list(SECTIONS), fill_value=0)  # fmt: skip
    mine = calls.set_index("event_id")[[f"n_{s}" for s in SECTIONS]].sort_index()
    mine.columns = list(SECTIONS)
    in_sample = calls[calls["in_sample"]]
    checks = [
        ("release matches saved outputs (build-time check)", meta["release_check"]["ok"]),
        ("labels cover exactly the saved sentences", bool((keys == "both").all())),
        ("sections in order in every call", not out_of_order(sentences)),
        (f"{expected_calls} modelling calls, each with prepared remarks",
         len(in_sample) == expected_calls and bool((in_sample["n_remarks"] > 0).all())),
        ("calls.csv counts agree with sentences.csv", counts.sort_index().astype(int).equals(mine.astype(int))),
    ]  # fmt: skip
    return pd.DataFrame(checks, columns=["check", "pass"]).assign(**{"pass": lambda d: d["pass"].map({True: "yes", False: "no"})})


def reported(calls: pd.DataFrame) -> list[str]:
    s = calls[calls["in_sample"]]
    rules = s["rule"].value_counts().rename_axis("rule").reset_index(name="calls")
    sizes = pd.DataFrame({"section": list(SECTIONS), "sentences": [int(s[f"n_{x}"].sum()) for x in SECTIONS],
                          "median per call": [s[f"n_{x}"].median() for x in SECTIONS]})  # fmt: skip
    both = pd.DataFrame({"min sentences in each": [1, 5, 10, 20],
                         "calls": [int(((s.n_remarks >= k) & (s.n_answers >= k)).sum()) for k in (1, 5, 10, 20)]})  # fmt: skip
    share = s["n_answers"] / (s["n_remarks"] + s["n_answers"])
    q = share.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
    return [
        "## 2. Rules used (modelling sample, reported)\n\n" + table(rules),
        "## 3. Section sizes (modelling sample, reported)\n\n" + table(sizes),
        "## 4. Calls with both sections (reported)\n\n" + table(both),
        "## 5. Answers' share of a call's remarks + answers (reported)\n\n"
        + table(pd.DataFrame({"quantile": q.index, "share": q.values})),
    ]


def audit_section(path: Path) -> str:
    title = "## 6. Hand audit (reported)\n\n"
    if not path.exists():
        return title + "Not yet generated: run make_audit.py."
    sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for name, col in (("remarks/answers boundary", "boundary_remarks_answers_ok"),
                      ("answers/final-paragraph boundary", "boundary_answers_finalpara_ok")):
        marks = sheet[col].str.strip().str.upper()
        marked = marks.isin(["Y", "N"])
        k, n = int((marks == "Y").sum()), int(marked.sum())
        lo, hi = wilson(k, n)
        rows.append({"item": name, "marked": n, "correct": k,
                     "share": f"{k / n:.2f}" if n else "-", "95% CI": f"{lo:.2f}–{hi:.2f}" if n else "-"})  # fmt: skip
    by_rule = sheet.assign(ok=sheet["boundary_remarks_answers_ok"].str.strip().str.upper()).query("ok in ['Y', 'N']")
    by_rule = by_rule.groupby("rule")["ok"].agg(marked="size", correct=lambda x: int((x == "Y").sum())).reset_index()
    text = title + f"{len(sheet)} calls sampled. Wilson intervals.\n\n" + table(pd.DataFrame(rows))
    if len(by_rule):
        text += "\n\nBoundary by rule:\n\n" + table(by_rule)
    return text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--root", type=Path, default=Path(".."))
    p.add_argument("--expected-calls", type=int, default=488, help="size of the modelling sample")
    args = p.parse_args(argv)
    checks = must_pass(args.data, args.root, args.expected_calls)
    calls = pd.read_csv(args.data / "calls.csv")
    report = "\n\n".join(
        ["# Remarks vs answers: acceptance checks", "## 1. Must pass\n\n" + table(checks)]
        + reported(calls)
        + [audit_section(args.data / "audit" / "audit_sheet.csv")]
    )
    (args.data / "acceptance.md").write_text(report + "\n")
    print(report)
    return 0 if (checks["pass"] == "yes").all() else 1


if __name__ == "__main__":
    sys.exit(main())

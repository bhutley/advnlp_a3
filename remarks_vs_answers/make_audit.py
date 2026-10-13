"""Write the hand-audit sheet for the remarks/answers split.

A simple random sample of calls from the 488-call modelling sample (so the share marked
correct estimates the rule's accuracy on that sample), with the transcript around each
call's boundary:

- data/audit/audit_sheet.csv  one row per call; fill in the marking columns
- data/audit/audit_sheet.md   the same calls, laid out for reading

Each sentence is shown as [sent_idx p<paragraph> <section>] text.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import DEFAULT_DATASET, read_dataset
from sections import CLOSING_MAX_SENTENCES, label_sections

MARK_COLUMNS = [
    "boundary_remarks_answers_ok", "correct_boundary_remarks_answers",
    "boundary_answers_finalpara_ok", "correct_boundary_answers_finalpara", "notes",
]
CONTEXT = 4  # sentences shown on each side of a boundary
ENDING = 3  # sentences shown from each end of a final paragraph too long to be closing

INSTRUCTIONS = """\
# Hand audit: prepared remarks vs answers

For each call, read the sentences around the boundary and fill in the CSV:

- **boundary_remarks_answers_ok**: `Y` if the last `remarks` sentence is the executive's
  last prepared sentence before Q&A, so the first `answers` sentence is the start of their
  first answer. `N` otherwise. For a call with no `answers`, `Y` means the call really has
  no answers.
- **correct_boundary_remarks_answers**: if `N`, the sent_idx of the true last prepared
  sentence (or a short note if it is not visible here).
- **boundary_answers_finalpara_ok**: `Y` if the split between the answers and the closing
  remarks is right: `closing` starts exactly where scripted closing remarks start, and a
  final paragraph that is an answer stays `answers`. `N` otherwise. "Before the final
  paragraph" shows the sentences leading into it. A final paragraph of up to 12 sentences
  (the most `closing` can cover) is shown whole; a longer one shows its first and last 3.
  Leave blank if the final paragraph is part of the remarks.
- **correct_boundary_answers_finalpara**: if `N`, the sent_idx of the first true closing
  sentence, or `none` if the call has no closing remarks.
- **notes**: anything else, e.g. a sentence that is the operator or an analyst.

Rules: `para1` = boundary at the end of paragraph 1; `handover` = at a handover to
questions; `handback` = at the end of paragraph 2 after a hand-off to a colleague.
For `handover` and `handback` calls, the end of paragraph 1 is also shown, because the
rule moved the boundary past it.
"""


def fmt(rows: pd.DataFrame) -> str:
    return "\n".join(f"[{r.sent_idx} p{r.para} {r.section}] {r.text}" for r in rows.itertuples())


def audit_rows(labelled: pd.DataFrame, calls: pd.DataFrame, event_ids) -> pd.DataFrame:
    out = []
    by_call = dict(tuple(labelled.groupby("event_id")))
    summary = calls.set_index("event_id")
    for k, event_id in enumerate(event_ids, start=1):
        g, c = by_call[event_id].reset_index(drop=True), summary.loc[event_id]
        b = int(c["boundary"])
        p1_end = int(g.loc[g["para"] == g["para"].min(), "sent_idx"].max())
        last = g[g["para"] == g["para"].max()]
        short = len(last) <= CLOSING_MAX_SENTENCES  # could be closing: show it whole
        ending = last if short else pd.concat([last.head(ENDING), last.tail(ENDING)])
        f = int(last["sent_idx"].min())
        after_b = f > b
        out.append({
            "audit_id": k, "event_id": event_id, "rule": c["rule"], "boundary": b,
            "n_sentences": int(c["n_sentences"]), "n_remarks": int(c["n_remarks"]),
            "n_answers": int(c["n_answers"]), "n_closing": int(c["n_closing"]),
            "end_of_para1": fmt(g.iloc[max(0, p1_end - CONTEXT + 1) : p1_end + 2]) if b > p1_end else "",
            "before_boundary": fmt(g.iloc[max(0, b - CONTEXT + 1) : b + 1]),
            "after_boundary": fmt(g.iloc[b + 1 : b + 1 + CONTEXT]),
            "before_final": fmt(g.iloc[max(b + 1, f - CONTEXT) : f]) if after_b else "",
            "final_paragraph": fmt(ending) if after_b else "",
        } | {m: "" for m in MARK_COLUMNS})  # fmt: skip
    return pd.DataFrame(out)


def to_markdown(sheet: pd.DataFrame) -> str:
    parts = [INSTRUCTIONS]
    for r in sheet.itertuples():
        parts.append(f"## {r.audit_id}. {r.event_id}\n\nrule `{r.rule}`, boundary after sentence "
                     f"{r.boundary}; {r.n_remarks} remarks, {r.n_answers} answers, {r.n_closing} closing")  # fmt: skip
        for title, text in (("End of paragraph 1", r.end_of_para1), ("Before the boundary", r.before_boundary),
                            ("After the boundary", r.after_boundary), ("Before the final paragraph", r.before_final),
                            ("Final paragraph", r.final_paragraph)):  # fmt: skip
            if text:
                parts.append(f"**{title}**\n\n```\n{text}\n```")
    return "\n\n".join(parts) + "\n"


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--calls", type=Path, default=Path("data/calls.csv"))
    p.add_argument("--out", type=Path, default=Path("data/audit"))
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="overwrite an existing sheet, marks included")
    p.add_argument("--refresh", action="store_true",
                   help="rebuild the text of an existing sheet for the same calls, keeping the marks")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sheet_path = args.out / "audit_sheet.csv"
    if sheet_path.exists() and not (args.force or args.refresh):
        print(f"{sheet_path} exists (it may hold your marks); use --force to replace it", file=sys.stderr)
        return 1
    calls = pd.read_csv(args.calls)
    frame = sorted(calls.loc[calls["in_sample"], "event_id"])
    if args.refresh:
        old = pd.read_csv(sheet_path, dtype=str, keep_default_na=False)
        picked = old["event_id"].to_numpy()
    else:
        picked = np.random.default_rng(args.seed).choice(frame, size=args.n, replace=False)
    labelled, labelled_calls = label_sections(read_dataset(args.dataset.expanduser()).query("event_id in @picked"))
    saved = calls.set_index("event_id").loc[list(picked), ["rule", "boundary"]]
    if not saved.equals(labelled_calls.set_index("event_id").loc[list(picked), ["rule", "boundary"]]):
        print("data/calls.csv is out of date with sections.py; rerun build_sections.py", file=sys.stderr)
        return 1
    sheet = audit_rows(labelled, labelled_calls, picked)
    if args.refresh:
        marks = old.set_index("event_id")[MARK_COLUMNS]
        sheet[MARK_COLUMNS] = marks.loc[sheet["event_id"]].to_numpy()
    args.out.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(sheet_path, index=False)
    (args.out / "audit_sheet.md").write_text(to_markdown(sheet))
    print(f"{len(sheet)} calls written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

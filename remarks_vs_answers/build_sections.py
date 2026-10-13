"""Label every sentence of the ACL19 release as remarks, answers, closing or heading.

Reads the release, checks it is the one the earlier stages used (it refuses to write
anything otherwise), labels each call, and writes:

- data/sentences.csv  event_id, sent_idx, para, section: one row per sentence
- data/calls.csv      one row per call: in_sample, rule, boundary and section counts
- data/meta.json      settings, the release check and counts

No transcript text is written: the dataset itself is not redistributed.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dataset import DEFAULT_DATASET, check_against_saved, modelling_sample, read_dataset
from sections import SECTIONS, label_sections

SENTENCE_COLUMNS = ["event_id", "sent_idx", "para", "section"]
CALL_COLUMNS = ["event_id", "in_sample", "n_sentences", "n_paragraphs", "rule", "boundary"] + [
    f"n_{s}" for s in SECTIONS
]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="ACL19_Release folder")
    p.add_argument("--root", type=Path, default=Path(".."), help="repository root")
    p.add_argument("--out", type=Path, default=Path("data"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    sentences = read_dataset(args.dataset.expanduser())
    check = check_against_saved(sentences, args.root)
    print(json.dumps(check, indent=2))
    if not check["ok"]:
        print("The release does not match the saved outputs; nothing written.", file=sys.stderr)
        return 1

    labelled, calls = label_sections(sentences)
    sample = modelling_sample(args.root)
    calls.insert(1, "in_sample", calls["event_id"].isin(sample))

    args.out.mkdir(parents=True, exist_ok=True)
    labelled[SENTENCE_COLUMNS].to_csv(args.out / "sentences.csv", index=False)
    calls[CALL_COLUMNS].to_csv(args.out / "calls.csv", index=False)
    in_sample = calls[calls["in_sample"]]
    meta = {
        "dataset": str(args.dataset),
        "release_check": check,
        "n_calls": len(calls),
        "n_in_sample": int(len(in_sample)),
        "rules_in_sample": in_sample["rule"].value_counts().to_dict(),
        "sections_in_sample": {s: int(in_sample[f"n_{s}"].sum()) for s in SECTIONS},
        "runtime_s": round(time.perf_counter() - start, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"written to {args.out}: {len(calls)} calls ({len(in_sample)} in the modelling sample)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

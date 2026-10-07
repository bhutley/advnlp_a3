"""Record each call's audio format: data/formats.csv.

The dataset's clips come in several formats (8–48 kHz, 8–128 kbps), and format affects
voice-quality features and SSL embeddings. Every clip is probed; a call's format is its
most common one, and clips in another format or unreadable are counted, not assumed away.
"""

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from clips import list_clips
from decode import probe_format

COLUMNS = ["event_id", "sample_rate", "kbps", "channels", "n_clips", "n_other_format",
           "n_unreadable"]  # fmt: skip


def call_formats(clips: pd.DataFrame, threads: int) -> pd.DataFrame:
    with ThreadPoolExecutor(threads) as ex:
        probed = list(ex.map(probe_format, clips["path"]))
    rows = []
    for event_id, idx in clips.groupby("event_id", sort=False).indices.items():
        found = [probed[i] for i in idx]
        readable = [tuple(f.values()) for f in found if f is not None]
        (sr, kbps, ch), n = Counter(readable).most_common(1)[0]
        rows.append([event_id, sr, kbps, ch, len(idx), len(readable) - n, len(idx) - len(readable)])
    return pd.DataFrame(rows, columns=COLUMNS)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path.home() / "var/data/earnings_calls/ACL19_Release"
    )
    parser.add_argument("--out", type=Path, default=Path("data/formats.csv"))
    parser.add_argument("--threads", type=int, default=16)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    formats = call_formats(list_clips(args.dataset), args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    formats.to_csv(args.out, index=False)
    counts = formats.groupby(["sample_rate", "kbps", "channels"]).size().rename("calls")
    print(counts.to_string())
    print(f"clips in a non-majority format: {formats['n_other_format'].sum()}; "
          f"unreadable: {formats['n_unreadable'].sum()}")  # fmt: skip
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

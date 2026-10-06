"""Read the ACL19 transcripts into one row per sentence.

Each call folder holds TextSequence.txt with one sentence per line. sent_idx is the
0-based index among non-blank lines, which is the alignment key for the audio clips.
"""

from pathlib import Path

import pandas as pd

TEXT_FILE = "TextSequence.txt"
COLUMNS = ["event_id", "sent_idx", "text"]


def read_call(path: Path) -> list[str]:
    """Non-blank, stripped lines of one transcript. Invalid UTF-8 is replaced."""
    lines = (line.strip() for line in path.read_text(errors="replace").splitlines())
    return [line for line in lines if line]


def read_sentences(root: Path, limit_calls: int | None = None) -> tuple[list[str], pd.DataFrame]:
    """(event_ids, sentences) for the call folders under root, in sorted order.

    event_ids lists every call, including any without sentences; such a call has no rows
    in sentences. limit_calls keeps only the first N calls, for trial runs.
    """
    folders = sorted(p for p in root.iterdir() if p.is_dir())[:limit_calls]
    event_ids, rows = [], []
    for folder in folders:
        event_ids.append(folder.name)
        path = folder / TEXT_FILE
        lines = read_call(path) if path.exists() else []
        rows.extend((folder.name, i, text) for i, text in enumerate(lines))
    return event_ids, pd.DataFrame(rows, columns=COLUMNS)

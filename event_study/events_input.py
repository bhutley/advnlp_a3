"""Scan the ACL19 earnings-call dataset into one row per call, with call timing.

The dataset gives call dates but not times. The greeting in the transcript is the best
clue: "good morning" means the call ran before the market opened ('pre'), "good
afternoon" or "good evening" means after the close ('post'). Without a greeting, a
reference to the release, e.g. "we reported results this morning", is used instead.
"""

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

GREETING_CHARS = 3000  # only the opening of the transcript is searched
TEXT_FILE = "TextSequence.txt"

MORNING_RE = re.compile(r"good\s+morning", re.IGNORECASE)
LATER_RE = re.compile(r"good\s+(afternoon|evening)", re.IGNORECASE)
THIS_MORNING_RE = re.compile(r"this\s+morning", re.IGNORECASE)
THIS_LATER_RE = re.compile(r"this\s+(afternoon|evening)", re.IGNORECASE)


def classify_timing(text: str) -> str:
    """'pre' for a morning greeting, 'post' for afternoon/evening, else 'unknown'.

    Without a greeting, "this morning" / "this afternoon" / "this evening" decide instead.
    Only the first GREETING_CHARS characters are searched. Within a tier, morning wins.
    """
    head = text[:GREETING_CHARS]
    for morning_re, later_re in ((MORNING_RE, LATER_RE), (THIS_MORNING_RE, THIS_LATER_RE)):
        if morning_re.search(head):
            return "pre"
        if later_re.search(head):
            return "post"
    return "unknown"


def parse_event_id(folder_name: str) -> tuple[str, date]:
    """'3M Company_20170425' → ('3M Company', date(2017, 4, 25)). Splits on the last '_'."""
    company, sep, ymd = folder_name.rpartition("_")
    if not sep or not company or len(ymd) != 8:
        raise ValueError(f"not a <company>_<YYYYMMDD> folder name: {folder_name!r}")
    return company, datetime.strptime(ymd, "%Y%m%d").date()


def scan_dataset(root: Path) -> pd.DataFrame:
    """One row per call folder under root: event_id, company, call_date, timing.

    Transcripts are read with errors='replace', since a few contain invalid UTF-8.
    A folder without a transcript gets timing 'unknown'.
    """
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        company, call_date = parse_event_id(d.name)
        text_path = d / TEXT_FILE
        text = text_path.read_text(errors="replace") if text_path.exists() else ""
        rows.append(
            {
                "event_id": d.name,
                "company": company,
                "call_date": call_date,
                "timing": classify_timing(text),
            }
        )
    return pd.DataFrame(rows, columns=["event_id", "company", "call_date", "timing"])

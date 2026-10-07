"""List the dataset's audio clips, one row per transcript sentence.

Each call folder holds CEO/<Speaker>_<para>_<sent>.mp3, one clip per non-blank line of
TextSequence.txt. Sorted by numeric (para, sent), the k-th clip is sentence k, which is
the same sent_idx that text_embeddings uses. Any mismatch is an error, never a guess.
"""

import math
import re
from pathlib import Path

import pandas as pd

CLIP_RE = re.compile(r"^(?P<speaker>.+)_(?P<para>\d+)_(?P<sent>\d+)\.mp3$")
TEXT_FILE = "TextSequence.txt"
AUDIO_DIR = "CEO"
COLUMNS = ["event_id", "sent_idx", "path", "n_words"]

# Quality rules. A few clips in the dataset are truncated (e.g. 0.073 s of audio for a
# 53-word line), and two calls have broken segmentation. Real speech stays under about
# 5 words/s at the 99th percentile; 8 words/s is beyond the 99.9th.
MAX_WORDS_PER_SEC = 8.0
MAX_BAD_SHARE = 0.10  # a call with more implausible clips than this is unreliable


def plausible_rate(n_words: int, duration_s: float) -> bool:
    """False if the clip is missing, empty, or too short to hold its transcript line."""
    if not (duration_s > 0) or math.isnan(duration_s):
        return False
    return n_words / duration_s <= MAX_WORDS_PER_SEC


def call_ok(n_bad: int, n_clips: int) -> bool:
    return n_bad <= MAX_BAD_SHARE * n_clips


def list_call_clips(call_dir: Path) -> list[Path]:
    """The call's MP3 clips sorted by numeric (para, sent).

    Non-MP3 entries (e.g. .ipynb_checkpoints) are ignored. Raises ValueError if an MP3
    name doesn't parse.
    """
    keyed = []
    for path in (call_dir / AUDIO_DIR).glob("*.mp3"):
        m = CLIP_RE.match(path.name)
        if not m:
            raise ValueError(f"{call_dir.name}: unexpected clip name {path.name!r}")
        keyed.append(((int(m["para"]), int(m["sent"])), path))
    return [path for _, path in sorted(keyed)]


def read_lines(path: Path) -> list[str]:
    """Non-blank, stripped transcript lines (same rule as text_embeddings/transcripts.py)."""
    lines = (line.strip() for line in path.read_text(errors="replace").splitlines())
    return [line for line in lines if line]


def list_clips(root: Path, limit_calls: int | None = None) -> pd.DataFrame:
    """One row per clip (event_id, sent_idx, path, n_words), calls in sorted order.

    Raises ValueError, naming the call, if its clip count differs from its line count.
    """
    rows = []
    for call_dir in sorted(p for p in root.iterdir() if p.is_dir())[:limit_calls]:
        clips = list_call_clips(call_dir)
        lines = read_lines(call_dir / TEXT_FILE)
        if len(clips) != len(lines):
            raise ValueError(
                f"{call_dir.name}: {len(clips)} clips but {len(lines)} transcript lines"
            )
        rows.extend(
            (call_dir.name, i, str(clip), len(line.split()))
            for i, (clip, line) in enumerate(zip(clips, lines))
        )
    return pd.DataFrame(rows, columns=COLUMNS)

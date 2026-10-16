"""Read the ACL19 release with each sentence's paragraph number, and check it against
the outputs the earlier stages saved.

Each call folder holds CEO/<Speaker>_<para>_<sent>.mp3, one clip per non-blank line of
TextSequence.txt. Sorted by numeric (para, sent), the k-th clip is sentence k: the same
sent_idx as text_embeddings and audio_embeddings (see audio_embeddings/clips.py). The
paragraph number, which those stages discard, is what separates prepared remarks from
answers.
"""

import re
from pathlib import Path

import pandas as pd

CLIP_RE = re.compile(r"^(?P<speaker>.+)_(?P<para>\d+)_(?P<sent>\d+)\.mp3$")
TEXT_FILE = "TextSequence.txt"
AUDIO_DIR = "CEO"
COLUMNS = ["event_id", "sent_idx", "para", "sent_in_para", "speaker", "n_words", "text"]
DEFAULT_DATASET = Path("~/var/data/earnings_calls/ACL19_Release").expanduser()


def read_lines(path: Path) -> list[str]:
    """Non-blank, stripped transcript lines (the rule every earlier stage uses)."""
    lines = (line.strip() for line in path.read_text(errors="replace").splitlines())
    return [line for line in lines if line]


def read_call(call_dir: Path) -> pd.DataFrame:
    """One row per sentence: sent_idx, para, sent_in_para, speaker, n_words, text.

    Raises ValueError, naming the call, if a clip name doesn't parse or the clip count
    differs from the line count.
    """
    keyed = []
    for path in (call_dir / AUDIO_DIR).glob("*.mp3"):
        m = CLIP_RE.match(path.name)
        if not m:
            raise ValueError(f"{call_dir.name}: unexpected clip name {path.name!r}")
        keyed.append((int(m["para"]), int(m["sent"]), m["speaker"]))
    keyed.sort()
    lines = read_lines(call_dir / TEXT_FILE)
    if len(keyed) != len(lines):
        raise ValueError(f"{call_dir.name}: {len(keyed)} clips but {len(lines)} transcript lines")
    rows = [
        (call_dir.name, i, para, sent, speaker, len(line.split()), line)
        for i, ((para, sent, speaker), line) in enumerate(zip(keyed, lines))
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def read_dataset(root: Path, limit_calls: int | None = None) -> pd.DataFrame:
    """read_call for every call folder under root, in sorted order."""
    calls = sorted(p for p in Path(root).iterdir() if p.is_dir())[:limit_calls]
    if not calls:
        raise ValueError(f"no call folders under {root}")
    return pd.concat([read_call(c) for c in calls], ignore_index=True)


def check_against_saved(sentences: pd.DataFrame, repo_root: Path) -> dict:
    """Compare the release with audio_embeddings' saved sentence table.

    Returns counts; 'ok' is True only if every (event_id, sent_idx) matches in both
    directions and every word count agrees, i.e. this is the release the earlier stages
    used and sent_idx means the same sentence everywhere.
    """
    saved = pd.read_csv(Path(repo_root) / "audio_embeddings/data/praat/sentences.csv",
                        usecols=["event_id", "sent_idx", "n_words"])  # fmt: skip
    m = sentences[["event_id", "sent_idx", "n_words"]].merge(
        saved, on=["event_id", "sent_idx"], how="outer", suffixes=("", "_saved"), indicator=True
    )
    both = m[m["_merge"] == "both"]
    result = {
        "calls": int(sentences["event_id"].nunique()),
        "sentences": len(sentences),
        "only_in_release": int((m["_merge"] == "left_only").sum()),
        "only_in_saved": int((m["_merge"] == "right_only").sum()),
        "word_count_mismatches": int((both["n_words"] != both["n_words_saved"]).sum()),
        "calls_with_several_speakers": int((sentences.groupby("event_id")["speaker"].nunique() > 1).sum()),
    }
    result["ok"] = not (result["only_in_release"] or result["only_in_saved"] or result["word_count_mismatches"])
    return result


def modelling_sample(repo_root: Path) -> set[str]:
    """The 488 calls every model uses: event-study status 'ok' and audio_call_ok
    (the same rule as modelling/data.py)."""
    root = Path(repo_root)
    events = pd.read_csv(root / "event_study/data/events.csv", usecols=["event_id", "status"])
    audio = pd.read_csv(root / "audio_embeddings/data/hubert/calls.csv", usecols=["event_id", "audio_call_ok"])
    ok_audio = set(audio.loc[audio["audio_call_ok"].astype(bool), "event_id"])
    return set(events.loc[events["status"] == "ok", "event_id"]) & ok_audio

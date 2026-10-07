"""Targets, ladder rungs, and assembly of a rung's feature blocks into one matrix.

Each rung is a list of named blocks. assemble() stacks the blocks' columns side by side
and returns the column slice of each block, which the pipeline uses to treat blocks
differently (scaling, PCA, layer selection).
"""

import numpy as np

from data import PROSODY_COLS, SENT_COLS, CallData

TARGETS_VOL = ["vol_3", "vol_7", "vol_15", "vol_30"]
TARGETS_DIR = ["dir_0_1", "dir_0_5"]
TASKS = {t: "regression" for t in TARGETS_VOL} | {t: "classification" for t in TARGETS_DIR}

TEXT = ["text_emb", "text_sent"]
AUDIO = ["audio_emb", "prosody"]
# B0 is not fitted: vol_pre_n for volatility, the training majority class for direction.
RUNGS = {
    "B0": [],
    "R1": ["past", "format"],
    "R2": ["past", "format"] + TEXT,
    "R3": ["past", "format"] + AUDIO,
    "R4": ["past", "format"] + TEXT + AUDIO,
    "R4_nopast": ["format"] + TEXT + AUDIO,
}
EMBEDDING_BLOCKS = {"text_emb", "audio_emb"}


def task_of(target: str) -> str:
    return TASKS[target]


def past_column(target: str) -> str:
    """The past-volatility column matching a volatility target's horizon."""
    if task_of(target) != "regression":
        raise ValueError(f"{target} has no past-volatility column")
    return target.replace("vol_", "vol_pre_")


def rung_blocks(rung: str, target: str) -> list[str]:
    """The rung's blocks; direction targets drop 'past' (past volatility isn't a direction cue)."""
    blocks = RUNGS[rung]
    return [b for b in blocks if b != "past"] if task_of(target) == "classification" else list(blocks)


def _block(data: CallData, block: str, target: str) -> np.ndarray:
    t = data.table
    columns = {
        "past": lambda: t[[past_column(target)]].to_numpy(),
        "format": lambda: t[data.format_cols].to_numpy(),
        "text_sent": lambda: t[SENT_COLS].to_numpy(),
        "prosody": lambda: t[PROSODY_COLS].to_numpy(),
        "text_emb": lambda: data.text_emb,
        "audio_emb": lambda: data.audio_emb,
    }
    return np.asarray(columns[block](), dtype=np.float64)


def assemble(data: CallData, blocks: list[str], target: str) -> tuple[np.ndarray, dict[str, slice]]:
    """(X, {block: column slice}) with blocks stacked in the given order."""
    parts, slices, start = [], {}, 0
    for block in blocks:
        part = _block(data, block, target)
        parts.append(part)
        slices[block] = slice(start, start + part.shape[1])
        start += part.shape[1]
    X = np.hstack(parts) if parts else np.empty((len(data.table), 0))
    return X, slices

"""Join the earlier stages' outputs into one call-level dataset.

The sample is every event_study call with status 'ok' whose audio is usable
(audio_call_ok): 488 calls from 240 firms. Rows are sorted by event_id, and every array
is row-aligned with the table. Any missing, duplicated or NaN input is an error.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VOL_HORIZONS = (3, 7, 15, 30)
VOL_COLS = [f"vol_{h}" for h in VOL_HORIZONS]
PAST_COLS = [f"vol_pre_{h}" for h in VOL_HORIZONS]
CAR_COLS = ["car_mm_0_1", "car_mm_0_5"]
DIR_COLS = ["dir_0_1", "dir_0_5"]  # 1 if the matching CAR is positive
SENT_COLS = ["tone", "frac_pos", "frac_neg", "p_pos", "p_neg", "p_neu"]
PROSODY_COLS = ["f0_sd_st", "speech_rate", "intensity_sd_db", "voiced_frac"]  # format-robust


@dataclass
class CallData:
    table: pd.DataFrame  # one row per call: ids, targets, past, format, sentiment, prosody
    text_emb: np.ndarray  # (n, text_dim) float32
    audio_emb: np.ndarray  # (n, audio_layers * audio_dim) float32, layer-major
    groups: np.ndarray  # ticker per row, for grouped CV
    format_cols: list[str]
    audio_layers: int
    audio_dim: int


def _unique(df: pd.DataFrame, source: str) -> pd.DataFrame:
    dup = df["event_id"][df["event_id"].duplicated()]
    if len(dup):
        raise ValueError(f"{source}: duplicate event_id {dup.iloc[0]!r}")
    return df


def _rows_for(event_ids: pd.Series, calls: pd.DataFrame, source: str) -> np.ndarray:
    """Positions in calls of each event_id, in order; raises if any is missing."""
    pos = pd.Series(np.arange(len(calls)), index=calls["event_id"])
    missing = event_ids[~event_ids.isin(pos.index)]
    if len(missing):
        raise ValueError(f"{source}: no row for {missing.iloc[0]!r}")
    return pos.loc[event_ids].to_numpy()


def _load_calls_with_array(directory: Path, source: str) -> tuple[pd.DataFrame, np.ndarray]:
    calls = _unique(pd.read_csv(directory / "calls.csv"), source)
    emb = np.load(directory / "call_emb.npy")
    if len(emb) != len(calls):
        raise ValueError(f"{source}: {len(calls)} rows in calls.csv but {len(emb)} in call_emb.npy")
    return calls, emb


def format_columns(formats: pd.DataFrame) -> pd.DataFrame:
    """One-hot fmt_<sample_rate>_<kbps> columns (sorted) plus a stereo flag."""
    label = "fmt_" + formats["sample_rate"].astype(str) + "_" + formats["kbps"].astype(str)
    onehot = pd.get_dummies(label).astype(int)
    onehot = onehot[sorted(onehot.columns)]
    onehot["stereo"] = (formats["channels"] > 1).astype(int).to_numpy()
    return onehot


def load_call_data(root: Path, text_model: str = "finbert_tone", audio_model: str = "hubert") -> CallData:
    """Load and join event_study, text_embeddings and audio_embeddings under the repo root."""
    root = Path(root)
    events = _unique(pd.read_csv(root / "event_study/data/events.csv"), "events.csv")
    text_calls, text_emb = _load_calls_with_array(
        root / "text_embeddings/data" / text_model, f"text_embeddings/{text_model}")
    audio_calls, audio_emb = _load_calls_with_array(
        root / "audio_embeddings/data" / audio_model, f"audio_embeddings/{audio_model}")  # fmt: skip
    praat = _unique(pd.read_csv(root / "audio_embeddings/data/praat/calls.csv"), "praat")
    formats = _unique(pd.read_csv(root / "audio_embeddings/data/formats.csv"), "formats.csv")

    ok_audio = audio_calls.loc[audio_calls["audio_call_ok"].astype(bool), "event_id"]
    sample = events[(events["status"] == "ok") & events["event_id"].isin(ok_audio)]
    table = sample.sort_values("event_id").reset_index(drop=True)
    ids = table["event_id"]

    table = table[["event_id", "ticker", "call_date"] + VOL_COLS + PAST_COLS + CAR_COLS].copy()
    for car, d in zip(CAR_COLS, DIR_COLS):
        table[d] = (table[car] > 0).astype(int)
    fmt = format_columns(formats.iloc[_rows_for(ids, formats, "formats.csv")].reset_index(drop=True))
    sent = text_calls.iloc[_rows_for(ids, text_calls, "text")][SENT_COLS].reset_index(drop=True)
    prosody = praat.iloc[_rows_for(ids, praat, "praat")][PROSODY_COLS].reset_index(drop=True)
    table = pd.concat([table, fmt, sent, prosody], axis=1)

    features = VOL_COLS + PAST_COLS + CAR_COLS + list(fmt.columns) + SENT_COLS + PROSODY_COLS
    nan_cols = [c for c in features if table[c].isna().any()]
    if nan_cols:
        raise ValueError(f"NaN in {nan_cols}")

    text = text_emb[_rows_for(ids, text_calls, "text")].astype(np.float32)
    audio = audio_emb[_rows_for(ids, audio_calls, "audio")].astype(np.float32)
    _, layers, dim = audio.shape
    if not (np.isfinite(text).all() and np.isfinite(audio).all()):
        raise ValueError("non-finite values in the embeddings")

    return CallData(
        table=table,
        text_emb=text,
        audio_emb=audio.reshape(len(audio), layers * dim),
        groups=table["ticker"].to_numpy(),
        format_cols=list(fmt.columns),
        audio_layers=layers,
        audio_dim=dim,
    )

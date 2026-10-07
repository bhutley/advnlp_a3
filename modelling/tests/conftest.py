"""Synthetic stand-ins for the earlier stages' outputs."""

from pathlib import Path

import numpy as np
import pandas as pd

TEXT_DIM, AUDIO_DIM, AUDIO_LAYERS = 8, 6, 13
FORMATS = [(44100, 64, 1), (22050, 32, 1), (11025, 16, 1), (44100, 128, 2)]


def make_project(root: Path, n_firms: int = 6, calls_per_firm: int = 3, seed: int = 0) -> list[str]:
    """Write events.csv, text/audio/praat call files and formats.csv under root.

    The last call is no_price and the second-to-last has audio_call_ok False, so the
    loaded sample is n_firms * calls_per_firm - 2 calls. Returns all event_ids.
    """
    rng = np.random.default_rng(seed)
    ids, tickers = [], []
    for f in range(n_firms):
        for c in range(calls_per_firm):
            ids.append(f"Firm{f} Inc_2017{c + 1:02d}15")
            tickers.append(f"F{f}")
    n = len(ids)
    order = rng.permutation(n)  # sources list calls in different orders

    events = pd.DataFrame({"event_id": ids, "ticker": tickers,
                           "call_date": [f"2017-{c + 1:02d}-15" for c in range(calls_per_firm)] * n_firms,
                           "status": ["ok"] * (n - 1) + ["no_price"]})  # fmt: skip
    for h in (3, 7, 15, 30):
        events[f"vol_{h}"] = rng.normal(-4.3, 0.4, n)
        events[f"vol_pre_{h}"] = rng.normal(-4.6, 0.4, n)
    for w in ("0_1", "0_5"):
        events[f"car_mm_{w}"] = rng.normal(0, 0.05, n)
    events.loc[n - 1, [c for c in events if c.startswith(("vol", "car"))]] = np.nan
    _write(root / "event_study/data/events.csv", events.iloc[order])

    for model in ("finbert_tone", "prosus_finbert"):
        d = root / "text_embeddings/data" / model
        calls = pd.DataFrame({"event_id": np.array(ids)[order], "n_sentences": 50})
        for col in ("p_pos", "p_neg", "p_neu", "tone", "frac_pos", "frac_neg"):
            calls[col] = rng.uniform(0, 1, n)
        _write(d / "calls.csv", calls)
        np.save(d / "call_emb.npy", rng.normal(size=(n, TEXT_DIM)).astype(np.float32))

    audio_ok = [True] * n
    audio_ok[n - 2] = False
    for model in ("hubert", "wav2vec2"):
        d = root / "audio_embeddings/data" / model
        calls = pd.DataFrame({"event_id": np.array(ids)[order], "n_sentences": 40, "n_bad": 0,
                              "audio_call_ok": np.array(audio_ok)[order], "duration_s": 300.0})  # fmt: skip
        _write(d / "calls.csv", calls)
        emb = rng.normal(size=(n, AUDIO_LAYERS, AUDIO_DIM)).astype(np.float16)
        np.save(d / "call_emb.npy", emb)

    praat = pd.DataFrame({"event_id": np.array(ids)[order], "n_sentences": 40, "n_bad": 0,
                          "audio_call_ok": np.array(audio_ok)[order]})  # fmt: skip
    for col in ("f0_sd_st", "speech_rate", "intensity_sd_db", "voiced_frac", "jitter_local"):
        praat[col] = rng.uniform(0, 1, n)
    _write(root / "audio_embeddings/data/praat/calls.csv", praat)

    fmts = [FORMATS[i % len(FORMATS)] for i in range(n)]
    formats = pd.DataFrame({"event_id": ids, "sample_rate": [f[0] for f in fmts],
                            "kbps": [f[1] for f in fmts], "channels": [f[2] for f in fmts],
                            "n_clips": 40, "n_other_format": 0, "n_unreadable": 0})  # fmt: skip
    _write(root / "audio_embeddings/data/formats.csv", formats.iloc[order])
    return ids


def _write(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_call_data(n_firms: int = 40, calls_per_firm: int = 3, text_dim: int = 20,
                   audio_layers: int = 13, audio_dim: int = 10, signal: float = 0.5,
                   seed: int = 0):  # fmt: skip
    """In-memory CallData with a planted signal.

    Every vol_n = 0.5 * vol_pre_n + signal * (sum of text_emb[:, :3]) + noise; the audio
    block is pure noise. dir_* = 1 where the same text signal is positive (plus noise).
    """
    from data import PAST_COLS, PROSODY_COLS, SENT_COLS, VOL_HORIZONS, CallData

    rng = np.random.default_rng(seed)
    n = n_firms * calls_per_firm
    text = rng.normal(size=(n, text_dim)).astype(np.float32)
    audio = rng.normal(size=(n, audio_layers * audio_dim)).astype(np.float32)
    planted = text[:, :3].sum(axis=1)
    table = pd.DataFrame({
        "event_id": [f"E{i:04d}" for i in range(n)],
        "ticker": np.repeat([f"F{f}" for f in range(n_firms)], calls_per_firm),
    })  # fmt: skip
    for h, past in zip(VOL_HORIZONS, PAST_COLS):
        table[past] = rng.normal(-4.6, 0.5, n)
        table[f"vol_{h}"] = 0.5 * table[past] + signal * planted + rng.normal(0, 0.3, n)
    for w in ("0_1", "0_5"):
        table[f"dir_{w}"] = (signal * planted + rng.normal(0, 0.5, n) > 0).astype(int)
    format_cols = ["fmt_a", "fmt_b"]
    table["fmt_a"] = rng.integers(0, 2, n)
    table["fmt_b"] = 1 - table["fmt_a"]
    for col in SENT_COLS + PROSODY_COLS:
        table[col] = rng.uniform(0, 1, n)
    return CallData(table=table, text_emb=text, audio_emb=audio, groups=table["ticker"].to_numpy(),
                    format_cols=format_cols, audio_layers=audio_layers, audio_dim=audio_dim)  # fmt: skip

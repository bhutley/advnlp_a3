"""Decode MP3 clips to float32 mono audio at 16 kHz with ffmpeg.

The dataset's clips are MP3 at several sample rates (8–48 kHz, constant within a call;
see probe_format); wav2vec2 and HuBERT expect 16 kHz. Resampling is left to ffmpeg, so
every feature set sees identical audio.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def check_ffmpeg() -> None:
    """Raise RuntimeError if ffmpeg isn't on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install it, e.g. `port install ffmpeg`")


def decode_clip(path: Path) -> np.ndarray | None:
    """Float32 mono samples in [-1, 1] at SAMPLE_RATE, or None if decoding fails or is empty."""
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
         "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
    )  # fmt: skip
    if result.returncode != 0 or not result.stdout:
        return None
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def probe_format(path: Path) -> dict | None:
    """{sample_rate, kbps, channels} of a clip's audio stream, or None if unreadable."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,bit_rate,channels", "-of", "json", str(path)],
        capture_output=True, text=True,
    )  # fmt: skip
    streams = json.loads(result.stdout or "{}").get("streams") if result.returncode == 0 else None
    if not streams:
        return None
    s = streams[0]
    return {
        "sample_rate": int(s["sample_rate"]),
        "kbps": int(s.get("bit_rate", 0)) // 1000,
        "channels": int(s["channels"]),
    }

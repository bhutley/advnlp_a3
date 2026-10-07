"""Interpretable acoustic features for one clip, computed with Praat via parselmouth.

Pitch-based features (F0, jitter, shimmer, HNR) are NaN when no frame is voiced.
Intensity is summarised as the arithmetic mean and SD of the dB contour, excluding
zero-energy frames, which Praat reports as -300 dB.
"""

import math

import numpy as np
import parselmouth
from parselmouth.praat import call

PITCH_FLOOR, PITCH_CEILING, TIME_STEP = 75.0, 500.0, 0.01
# Praat's standard jitter/shimmer arguments: period floor, period ceiling, max period
# factor, (shimmer only) max amplitude factor.
PERIOD_ARGS = (0.0001, 0.02, 1.3)
MAX_AMPLITUDE_FACTOR = 1.6
HNR_UNDEFINED = -200.0  # Praat's value for frames where harmonicity is undefined
INTENSITY_UNDEFINED = -300.0  # Praat's value for zero-energy (digitally silent) frames
MIN_INTENSITY_DURATION = 6.4 / PITCH_FLOOR  # Praat's intensity window; shorter clips get NaN

PRAAT_COLUMNS = [
    "duration_s", "n_words", "speech_rate", "voiced_frac", "f0_mean_hz", "f0_sd_st",
    "intensity_mean_db", "intensity_sd_db", "jitter_local", "shimmer_local", "hnr_db",
]  # fmt: skip


def _finite(x) -> float:
    x = float(x)
    return x if math.isfinite(x) else math.nan


def praat_features(audio: np.ndarray, sr: int, n_words: int) -> dict:
    """The PRAAT_COLUMNS values for one clip, in that order."""
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    duration = sound.get_total_duration()
    f = dict.fromkeys(PRAAT_COLUMNS, math.nan)
    f.update(duration_s=duration, n_words=n_words, speech_rate=n_words / duration)

    f0 = sound.to_pitch(
        time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING
    ).selected_array["frequency"]
    voiced = f0[f0 > 0]
    f["voiced_frac"] = len(voiced) / len(f0) if len(f0) else 0.0

    if duration >= MIN_INTENSITY_DURATION:
        db = sound.to_intensity(minimum_pitch=PITCH_FLOOR, time_step=TIME_STEP).values[0]
        db = db[np.isfinite(db) & (db > INTENSITY_UNDEFINED)]
        if len(db):
            f.update(intensity_mean_db=float(db.mean()), intensity_sd_db=float(db.std()))

    if len(voiced) >= 2:
        semitones = 12 * np.log2(voiced)
        f.update(f0_mean_hz=float(voiced.mean()), f0_sd_st=float(semitones.std()))
        points = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        f["jitter_local"] = _finite(call(points, "Get jitter (local)", 0, 0, *PERIOD_ARGS))
        f["shimmer_local"] = _finite(
            call([sound, points], "Get shimmer (local)", 0, 0, *PERIOD_ARGS, MAX_AMPLITUDE_FACTOR)
        )
        hnr = call(sound, "To Harmonicity (cc)", TIME_STEP, PITCH_FLOOR, 0.1, 1.0).values[0]
        hnr = hnr[hnr != HNR_UNDEFINED]
        if len(hnr):
            f["hnr_db"] = float(hnr.mean())
    return f

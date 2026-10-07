import math

import numpy as np
import pytest

from conftest import tone
from praat import PRAAT_COLUMNS, praat_features

SR = 16000


def vibrato(base_hz: float, depth: float, rate_hz: float, seconds: float) -> np.ndarray:
    """A sine whose frequency swings ±depth (fraction) around base_hz at rate_hz."""
    t = np.arange(int(seconds * SR)) / SR
    freq = base_hz * (1 + depth * np.sin(2 * np.pi * rate_hz * t))
    return (0.3 * np.sin(2 * np.pi * np.cumsum(freq) / SR)).astype(np.float32)


def test_columns():
    f = praat_features(tone(200, 1.0), SR, n_words=3)
    assert list(f) == PRAAT_COLUMNS


def test_steady_tone():
    f = praat_features(tone(200, 2.0), SR, n_words=5)

    assert f["duration_s"] == pytest.approx(2.0)
    assert f["f0_mean_hz"] == pytest.approx(200, abs=2)
    assert f["f0_sd_st"] < 0.1
    assert f["voiced_frac"] > 0.9
    assert f["jitter_local"] < 0.005
    assert f["shimmer_local"] < 0.02
    assert f["hnr_db"] > 20


def test_vibrato_raises_pitch_variation():
    steady = praat_features(tone(200, 2.0), SR, n_words=5)
    wobbly = praat_features(vibrato(200, 0.05, 5, 2.0), SR, n_words=5)

    assert wobbly["f0_mean_hz"] == pytest.approx(200, abs=5)
    # ±5% is about ±0.85 semitones; a sine swing has SD amplitude / sqrt(2) ≈ 0.6
    assert 0.4 < wobbly["f0_sd_st"] < 0.8
    assert wobbly["f0_sd_st"] > 5 * steady["f0_sd_st"]


def test_silence_is_unvoiced_without_raising():
    f = praat_features(np.zeros(SR, dtype=np.float32), SR, n_words=0)

    assert f["voiced_frac"] == 0
    for col in ("f0_mean_hz", "f0_sd_st", "jitter_local", "shimmer_local", "hnr_db"):
        assert math.isnan(f[col]), col
    # Praat reports zero energy as -300 dB; that is undefined, not a loudness.
    assert math.isnan(f["intensity_mean_db"]) and math.isnan(f["intensity_sd_db"])
    assert f["speech_rate"] == 0


def test_digital_silence_does_not_drag_intensity():
    clean = praat_features(tone(200, 1.0), SR, n_words=1)
    padded = praat_features(np.concatenate([np.zeros(SR // 2, np.float32), tone(200, 1.0)]),
                            SR, n_words=1)  # fmt: skip

    assert padded["intensity_mean_db"] == pytest.approx(clean["intensity_mean_db"], abs=1)
    assert padded["intensity_sd_db"] < 5


def test_speech_rate():
    f = praat_features(tone(200, 2.0), SR, n_words=5)
    assert f["n_words"] == 5
    assert f["speech_rate"] == pytest.approx(2.5)


def test_louder_is_more_intense():
    quiet = praat_features(tone(200, 1.0, amp=0.05), SR, n_words=1)
    loud = praat_features(tone(200, 1.0, amp=0.5), SR, n_words=1)

    # 10x amplitude is +20 dB
    assert loud["intensity_mean_db"] - quiet["intensity_mean_db"] == pytest.approx(20, abs=1)
    assert loud["intensity_sd_db"] < 3  # a steady tone has a flat intensity contour


def test_works_at_dataset_rate_after_mp3(tmp_path):
    from conftest import make_mp3
    from decode import decode_clip

    path = tmp_path / "clip.mp3"
    make_mp3(path, tone(150, 2.0))
    f = praat_features(decode_clip(path), SR, n_words=4)

    assert f["f0_mean_hz"] == pytest.approx(150, abs=2)  # pitch survives the codec


def test_clip_shorter_than_intensity_window():
    # Praat's intensity analysis needs 6.4 / 75 Hz ≈ 0.085 s; the dataset has a 0.073 s clip.
    f = praat_features(tone(200, 0.07), SR, n_words=1)
    assert math.isnan(f["intensity_mean_db"]) and math.isnan(f["intensity_sd_db"])
    assert f["duration_s"] == pytest.approx(0.07)

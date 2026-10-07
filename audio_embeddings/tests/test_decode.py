import numpy as np
import pytest

import decode
from conftest import make_mp3, tone
from decode import SAMPLE_RATE, check_ffmpeg, decode_clip, probe_format


def test_decode_clip(tmp_path):
    path = tmp_path / "clip.mp3"
    make_mp3(path, tone(200, 2.0))

    audio = decode_clip(path)

    assert audio.dtype == np.float32 and audio.ndim == 1
    # MP3 encoders add a little padding; allow 0.05 s.
    assert abs(len(audio) / SAMPLE_RATE - 2.0) < 0.05
    assert 0.2 < np.abs(audio).max() <= 1.0  # the tone's amplitude is 0.3


def test_decode_clip_keeps_pitch(tmp_path):
    path = tmp_path / "clip.mp3"
    make_mp3(path, tone(200, 2.0))

    audio = decode_clip(path)

    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.argmax(spectrum) * SAMPLE_RATE / len(audio)
    assert abs(peak_hz - 200) < 2


def test_decode_clip_corrupt_file(tmp_path):
    path = tmp_path / "bad.mp3"
    path.write_bytes(np.random.default_rng(0).bytes(2000))
    assert decode_clip(path) is None


def test_decode_clip_missing_file(tmp_path):
    assert decode_clip(tmp_path / "nope.mp3") is None


def test_check_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(decode.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        check_ffmpeg()


def test_check_ffmpeg_present():
    check_ffmpeg()  # ffmpeg is installed on this machine


def test_probe_format(tmp_path):
    path = tmp_path / "clip.mp3"
    make_mp3(path, tone(200, 1.0))
    assert probe_format(path) == {"sample_rate": 11025, "kbps": 16, "channels": 1}


def test_probe_format_corrupt(tmp_path):
    path = tmp_path / "bad.mp3"
    path.write_bytes(b"")
    assert probe_format(path) is None

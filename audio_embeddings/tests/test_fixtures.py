import subprocess

import torch

from conftest import DATASET_SR, make_mp3, tone


def test_make_mp3_matches_dataset_format(tmp_path):
    path = tmp_path / "clip.mp3"
    make_mp3(path, tone(200, 1.0))

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels",
         "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    assert "codec_name=mp3" in out
    assert f"sample_rate={DATASET_SR}" in out
    assert "channels=1" in out


def test_tiny_ssl_forward(tiny_ssl):
    model = tiny_ssl()
    with torch.no_grad():
        out = model(torch.from_numpy(tone(200, 1.0))[None], output_hidden_states=True)

    assert len(out.hidden_states) == 3  # CNN output + 2 transformer layers
    n_frames = out.hidden_states[-1].shape[1]
    assert n_frames == 798  # (16000 − 10) // 5 + 1 = 3199, then (3199 − 8) // 4 + 1
    assert out.hidden_states[-1].shape[2] == 16

import subprocess

import pandas as pd

from build_formats import main
from conftest import tone


def mp3(path, sr, kbps):
    pcm = (tone(200, 1.0) * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
                    "-i", "-", "-ar", str(sr), "-b:a", f"{kbps}k", str(path)],
                   input=pcm, check=True)  # fmt: skip


def make_call(root, event_id, formats):
    d = root / event_id / "CEO"
    d.mkdir(parents=True)
    for i, (sr, kbps) in enumerate(formats, start=1):
        mp3(d / f"Jane Doe_1_{i}.mp3", sr, kbps)
    (root / event_id / "TextSequence.txt").write_text("word\n" * len(formats))


def test_build_formats(tmp_path):
    dataset = tmp_path / "dataset"
    make_call(dataset, "Alpha_20170101", [(11025, 16), (11025, 16)])
    make_call(dataset, "Beta_20170102", [(22050, 32), (22050, 32), (11025, 16)])  # one odd clip
    (dataset / "Alpha_20170101" / "CEO" / "Jane Doe_1_2.mp3").write_bytes(b"")  # empty file
    out = tmp_path / "formats.csv"

    assert main(["--dataset", str(dataset), "--out", str(out), "--threads", "2"]) == 0

    df = pd.read_csv(out)
    assert list(df.columns) == ["event_id", "sample_rate", "kbps", "channels", "n_clips",
                                "n_other_format", "n_unreadable"]  # fmt: skip
    assert df.values.tolist() == [
        ["Alpha_20170101", 11025, 16, 1, 2, 0, 1],
        ["Beta_20170102", 22050, 32, 1, 3, 1, 0],
    ]

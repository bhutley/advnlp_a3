import json

import parselmouth

import numpy as np
import pandas as pd
import pytest

import build_audio
import ssl_embed
from build_audio import main
from conftest import make_mp3, tone
from praat import PRAAT_COLUMNS

TEST_LAYER = 1  # the tiny model has hidden states 0..2


@pytest.fixture
def project(tmp_path, monkeypatch, tiny_ssl):
    """Two calls: Alpha (2 clips) and Beta (3 clips), tones of known pitch."""
    dataset = tmp_path / "dataset"
    calls = {
        "Alpha Corp_20170425": [(200, 1.0, "revenue grew"), (220, 1.5, "we raised guidance")],
        "Beta Inc_20170426": [(150, 1.2, "demand was weak"), (160, 1.0, "costs rose"),
                              (170, 1.3, "we cut the outlook today")],
    }  # fmt: skip
    for event_id, clips in calls.items():
        (dataset / event_id / "CEO").mkdir(parents=True)
        lines = []
        for i, (hz, seconds, text) in enumerate(clips, start=1):
            make_mp3(dataset / event_id / "CEO" / f"Jane Doe_1_{i}.mp3", tone(hz, seconds))
            lines.append(text)
        (dataset / event_id / "TextSequence.txt").write_text("\n".join(lines) + "\n")

    monkeypatch.setattr(ssl_embed, "load_model", lambda key, device: (tiny_ssl(), True))
    monkeypatch.setattr(ssl_embed, "SENTENCE_LAYER", TEST_LAYER)
    out_dir = tmp_path / "out"
    argv = ["--dataset", str(dataset), "--out-dir", str(out_dir), "--device", "cpu",
            "--workers", "0"]  # fmt: skip
    return argv, dataset, out_dir


def test_ssl_end_to_end(project):
    argv, _, out_dir = project

    assert main(["--features", "wav2vec2"] + argv) == 0

    d = out_dir / "wav2vec2"
    sentences = pd.read_csv(d / "sentences.csv")
    assert list(sentences.columns) == ["event_id", "sent_idx", "duration_s", "n_frames", "audio_ok"]
    assert sentences["audio_ok"].all()
    assert sentences["sent_idx"].tolist() == [0, 1, 0, 1, 2]
    assert sentences["duration_s"].iloc[1] == pytest.approx(1.5, abs=0.05)
    assert (sentences["n_frames"] > 0).all()

    sent = np.load(d / "sentence_emb.npy")
    assert sent.shape == (5, 16) and sent.dtype == np.float16
    calls = pd.read_csv(d / "calls.csv")
    assert calls["event_id"].tolist() == ["Alpha Corp_20170425", "Beta Inc_20170426"]
    assert calls["n_sentences"].tolist() == [2, 3]
    assert calls["duration_s"].iloc[1] == pytest.approx(3.5, abs=0.15)

    call = np.load(d / "call_emb.npy")
    assert call.shape == (2, 3, 16) and call.dtype == np.float16
    # The call vector at the sentence layer is the mean of that call's sentence rows.
    np.testing.assert_allclose(
        call[1, TEST_LAYER].astype(np.float32), sent[2:].astype(np.float32).mean(0), atol=2e-3
    )

    meta = json.loads((d / "meta.json").read_text())
    assert meta["checkpoint"] == "facebook/wav2vec2-base"
    assert meta["sentence_layer"] == TEST_LAYER
    assert (meta["n_calls"], meta["n_clips"], meta["n_failed"]) == (2, 5, 0)


def test_praat_end_to_end(project):
    argv, _, out_dir = project

    assert main(["--features", "praat"] + argv) == 0

    d = out_dir / "praat"
    assert sorted(p.name for p in d.iterdir()) == ["calls.csv", "meta.json", "sentences.csv"]
    sentences = pd.read_csv(d / "sentences.csv")
    assert list(sentences.columns) == ["event_id", "sent_idx"] + PRAAT_COLUMNS + ["audio_ok"]
    assert sentences["f0_mean_hz"].iloc[0] == pytest.approx(200, abs=3)
    assert sentences["n_words"].tolist() == [2, 3, 3, 2, 5]

    calls = pd.read_csv(d / "calls.csv")
    assert list(calls.columns) == ["event_id", "n_sentences", "n_bad", "audio_call_ok"] + PRAAT_COLUMNS
    assert calls["f0_mean_hz"].iloc[1] == pytest.approx(160, abs=3)  # mean of 150, 160, 170


def test_corrupt_clip_is_nan_and_counted(project, capsys):
    argv, dataset, out_dir = project
    (dataset / "Beta Inc_20170426" / "CEO" / "Jane Doe_1_2.mp3").write_bytes(b"not audio")

    assert main(["--features", "wav2vec2"] + argv) == 0

    d = out_dir / "wav2vec2"
    sent = np.load(d / "sentence_emb.npy")
    assert np.isnan(sent[3]).all() and not np.isnan(sent[2]).any()
    assert pd.read_csv(d / "calls.csv")["n_sentences"].tolist() == [2, 2]
    assert json.loads((d / "meta.json").read_text())["n_failed"] == 1
    assert "Jane Doe_1_2.mp3" in capsys.readouterr().out


def test_refuses_existing_output(project, capsys):
    argv, _, out_dir = project
    (out_dir / "praat").mkdir(parents=True)
    (out_dir / "praat" / "calls.csv").write_text("old")

    assert main(["--features", "praat"] + argv) == 1
    assert (out_dir / "praat" / "calls.csv").read_text() == "old"
    assert "--force" in capsys.readouterr().err

    assert main(["--features", "praat"] + argv + ["--force"]) == 0
    assert (out_dir / "praat" / "calls.csv").read_text().startswith("event_id,")


def test_worker_pool_matches_in_process(project):
    argv, _, out_dir = project
    main(["--features", "praat"] + argv)
    in_process = pd.read_csv(out_dir / "praat" / "sentences.csv")

    pooled_argv = [a if a != "0" else "2" for a in argv]  # --workers 2
    main(["--features", "praat", "--force"] + pooled_argv)
    pooled = pd.read_csv(out_dir / "praat" / "sentences.csv")

    pd.testing.assert_frame_equal(pooled, in_process)


def truncate_beta_clip(dataset):
    """Give Beta's second clip (1.0 s of audio) a 30-word line: 30 words/s, implausible."""
    path = dataset / "Beta Inc_20170426" / "TextSequence.txt"
    lines = path.read_text().splitlines()
    lines[1] = " ".join(["word"] * 30)
    path.write_text("\n".join(lines) + "\n")


def test_ssl_excludes_implausible_clip(project):
    argv, dataset, out_dir = project
    truncate_beta_clip(dataset)

    assert main(["--features", "wav2vec2"] + argv) == 0

    d = out_dir / "wav2vec2"
    sentences = pd.read_csv(d / "sentences.csv")
    assert sentences["audio_ok"].tolist() == [True, True, True, False, True]
    sent = np.load(d / "sentence_emb.npy")
    assert not np.isnan(sent[3]).any()  # the bad clip keeps its vector, flagged

    calls = pd.read_csv(d / "calls.csv")
    assert list(calls.columns) == ["event_id", "n_sentences", "n_bad", "audio_call_ok", "duration_s"]
    assert calls["n_sentences"].tolist() == [2, 2]
    assert calls["n_bad"].tolist() == [0, 1]
    assert calls["audio_call_ok"].tolist() == [True, False]  # 1 of 3 is over 10%
    call = np.load(d / "call_emb.npy")
    good = sent[[2, 4]].astype(np.float32).mean(0)  # Beta's plausible clips only
    np.testing.assert_allclose(call[1, TEST_LAYER].astype(np.float32), good, atol=2e-3)

    meta = json.loads((d / "meta.json").read_text())
    assert (meta["n_failed"], meta["n_bad"], meta["n_bad_calls"]) == (0, 1, 1)


def test_praat_excludes_implausible_clip(project):
    argv, dataset, out_dir = project
    truncate_beta_clip(dataset)

    assert main(["--features", "praat"] + argv) == 0

    calls = pd.read_csv(out_dir / "praat" / "calls.csv")
    beta = calls.iloc[1]
    assert (beta["n_sentences"], beta["n_bad"], beta["audio_call_ok"]) == (2, 1, False)
    assert beta["f0_mean_hz"] == pytest.approx(160, abs=3)  # mean of 150 and 170 only


def test_praat_error_marks_clip_failed(project, monkeypatch, capsys):
    argv, _, out_dir = project
    real, calls = build_audio.praat_features, []

    def flaky(audio, sr, n_words):
        calls.append(n_words)
        if len(calls) == 4:  # in-process order: Beta's second clip, "costs rose"
            raise parselmouth.PraatError("Sound: shorter than window length.")
        return real(audio, sr, n_words)

    monkeypatch.setattr(build_audio, "praat_features", flaky)

    assert main(["--features", "praat"] + argv) == 0

    meta = json.loads((out_dir / "praat" / "meta.json").read_text())
    assert meta["n_failed"] == 1
    sentences = pd.read_csv(out_dir / "praat" / "sentences.csv")
    assert not sentences["audio_ok"].iloc[3] and np.isnan(sentences["f0_mean_hz"].iloc[3])
    assert "shorter than window length" in capsys.readouterr().out

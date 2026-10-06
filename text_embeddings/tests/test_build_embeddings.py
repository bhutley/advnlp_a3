import json

import numpy as np
import pandas as pd
import pytest

import embed
from build_embeddings import main


@pytest.fixture
def project(tmp_path, monkeypatch, tiny_model, tiny_tokenizer):
    dataset = tmp_path / "dataset"
    for folder, text in [
        ("Alpha Corp_20170425", "revenue grew strongly\nwe raised guidance\n"),
        ("Beta Inc_20170426", "we missed expectations\n\ncut our outlook\nmargin was weak\n"),
    ]:
        (dataset / folder).mkdir(parents=True)
        (dataset / folder / "TextSequence.txt").write_text(text)

    loaded = []

    def fake_load(key, device):
        loaded.append((key, device))
        return tiny_model(), tiny_tokenizer

    monkeypatch.setattr(embed, "load_model", fake_load)
    out_dir = tmp_path / "out"
    argv = ["--model", "finbert_tone", "--dataset", str(dataset), "--out-dir", str(out_dir),
            "--device", "cpu", "--batch-size", "2"]  # fmt: skip
    return argv, out_dir / "finbert_tone", loaded


def test_main_writes_outputs(project):
    argv, model_dir, loaded = project

    assert main(argv) == 0

    assert loaded == [("finbert_tone", embed.pick_device("cpu"))]
    assert sorted(p.name for p in model_dir.iterdir()) == [
        "call_emb.npy", "calls.csv", "meta.json", "sentence_emb.npy", "sentences.csv",
    ]  # fmt: skip

    sentences = pd.read_csv(model_dir / "sentences.csv")
    assert list(sentences.columns) == ["event_id", "sent_idx", "n_tokens", "p_pos", "p_neg", "p_neu"]
    assert sentences[["event_id", "sent_idx"]].values.tolist() == [
        ["Alpha Corp_20170425", 0], ["Alpha Corp_20170425", 1],
        ["Beta Inc_20170426", 0], ["Beta Inc_20170426", 1], ["Beta Inc_20170426", 2],
    ]  # fmt: skip
    sentence_emb = np.load(model_dir / "sentence_emb.npy")
    assert sentence_emb.shape == (5, 32) and sentence_emb.dtype == np.float16

    calls = pd.read_csv(model_dir / "calls.csv")
    assert calls["event_id"].tolist() == ["Alpha Corp_20170425", "Beta Inc_20170426"]
    assert calls["n_sentences"].tolist() == [2, 3]
    call_emb = np.load(model_dir / "call_emb.npy")
    assert call_emb.shape == (2, 32) and call_emb.dtype == np.float32
    # The call vector is the mean of its sentence vectors (up to float16 storage).
    np.testing.assert_allclose(call_emb[1], sentence_emb[2:].astype(np.float32).mean(0), atol=1e-2)

    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta["checkpoint"] == "yiyanghkust/finbert-tone"
    assert meta["revision"] == embed.MODELS["finbert_tone"][1]
    assert (meta["n_calls"], meta["n_sentences"], meta["n_truncated"]) == (2, 5, 0)
    assert meta["device"] == "cpu"
    assert {"pooling", "torch", "transformers", "runtime_s", "max_length"} <= set(meta)


def test_main_refuses_existing_output(project, capsys):
    argv, model_dir, _ = project
    model_dir.mkdir(parents=True)
    (model_dir / "calls.csv").write_text("old")

    assert main(argv) == 1
    assert (model_dir / "calls.csv").read_text() == "old"
    assert "--force" in capsys.readouterr().err


def test_main_force_overwrites(project):
    argv, model_dir, _ = project
    model_dir.mkdir(parents=True)
    (model_dir / "calls.csv").write_text("old")

    assert main(argv + ["--force"]) == 0
    assert (model_dir / "calls.csv").read_text().startswith("event_id,")


def test_main_limit_calls(project):
    argv, model_dir, _ = project

    assert main(argv + ["--limit-calls", "1"]) == 0
    calls = pd.read_csv(model_dir / "calls.csv")
    assert calls["event_id"].tolist() == ["Alpha Corp_20170425"]
    assert json.loads((model_dir / "meta.json").read_text())["n_sentences"] == 2


def test_main_rejects_unknown_model(project):
    argv, _, _ = project
    with pytest.raises(SystemExit):
        main(["--model", "bert_base"] + argv[2:])

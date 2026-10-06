import math

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification

import embed
from conftest import PROSUS_LABELS, TONE_LABELS
from embed import (
    MODELS,
    PROB_COLUMNS,
    SentenceOutput,
    aggregate_calls,
    check_vocab,
    embed_sentences,
    label_order,
    mean_pool,
    pick_device,
)

# --- registry ---


def test_models_are_pinned():
    assert set(MODELS) == {"finbert_tone", "prosus_finbert"}
    for name, revision in MODELS.values():
        assert "/" in name and len(revision) == 40  # a full commit hash, not a branch


# --- label_order ---


def test_label_order_tone():
    # finbert-tone: 0 Neutral, 1 Positive, 2 Negative → p_pos, p_neg, p_neu
    assert PROB_COLUMNS == ["p_pos", "p_neg", "p_neu"]
    assert label_order(TONE_LABELS) == [1, 2, 0]


def test_label_order_prosus():
    assert label_order(PROSUS_LABELS) == [0, 1, 2]


def test_label_order_string_keys():
    # id2label read straight from JSON has string keys
    assert label_order({"0": "neutral", "1": "positive", "2": "negative"}) == [1, 2, 0]


@pytest.mark.parametrize(
    "id2label",
    [
        {0: "bullish", 1: "bearish", 2: "neutral"},
        {0: "positive", 1: "negative"},
        {0: "positive", 1: "negative", 2: "neutral", 3: "mixed"},
        {0: "positive", 1: "positive", 2: "neutral"},
    ],
)
def test_label_order_rejects_unexpected_labels(id2label):
    with pytest.raises(ValueError):
        label_order(id2label)


# --- pick_device ---


def test_pick_device_cpu():
    assert pick_device("cpu") == torch.device("cpu")


def test_pick_device_auto_without_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto") == torch.device("cpu")


def test_pick_device_auto_with_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert pick_device("auto") == torch.device("mps")


def test_pick_device_mps_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        pick_device("mps")


def test_pick_device_unknown():
    with pytest.raises(ValueError):
        pick_device("cuda:7")


# --- check_vocab ---


def test_check_vocab_matches(tiny_model, tiny_tokenizer):
    check_vocab(tiny_model(), tiny_tokenizer)  # no error


def test_check_vocab_mismatch(tiny_tokenizer):
    model = BertForSequenceClassification(
        BertConfig(vocab_size=len(tiny_tokenizer) + 8, hidden_size=32, num_hidden_layers=1,
                   num_attention_heads=2, intermediate_size=64, num_labels=3)
    )  # fmt: skip
    with pytest.raises(ValueError, match="vocab"):
        check_vocab(model, tiny_tokenizer)


def test_load_model_rejects_unknown_key():
    with pytest.raises(KeyError):
        embed.load_model("bert_base", torch.device("cpu"))


# --- mean_pool ---


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])  # last token is padding
    mask = torch.tensor([[1, 1, 0]])
    assert mean_pool(hidden, mask).tolist() == [[2.0, 3.0]]


# --- embed_sentences ---

SHORT = "revenue grew"
MEDIUM = "we raised guidance and margin was strong"
LONG = " ".join(["the quarter was in line with flat demand and higher costs"] * 3)
CPU = torch.device("cpu")


def one_at_a_time(texts, model, tokenizer):
    outs = [embed_sentences([t], model, tokenizer, CPU) for t in texts]
    return np.vstack([o.emb for o in outs]), np.vstack([o.probs for o in outs])


def test_embed_sentences_shapes(tiny_model, tiny_tokenizer):
    out = embed_sentences([SHORT, MEDIUM], tiny_model(), tiny_tokenizer, CPU)
    assert out.emb.shape == (2, 32) and out.emb.dtype == np.float32
    assert out.probs.shape == (2, 3)
    assert out.n_tokens.tolist() == [4, 9]  # [CLS] + words + [SEP]
    assert out.n_truncated == 0


def test_embed_sentences_padding_invariance(tiny_model, tiny_tokenizer):
    model = tiny_model()
    alone = embed_sentences([SHORT], model, tiny_tokenizer, CPU)
    padded = embed_sentences([SHORT, LONG], model, tiny_tokenizer, CPU, batch_size=2)

    np.testing.assert_allclose(padded.emb[0], alone.emb[0], atol=1e-5)
    np.testing.assert_allclose(padded.probs[0], alone.probs[0], atol=1e-5)


def test_embed_sentences_restores_order(tiny_model, tiny_tokenizer):
    model = tiny_model()
    texts = [LONG, SHORT, MEDIUM]
    out = embed_sentences(texts, model, tiny_tokenizer, CPU, batch_size=2)
    emb, probs = one_at_a_time(texts, model, tiny_tokenizer)

    np.testing.assert_allclose(out.emb, emb, atol=1e-5)
    np.testing.assert_allclose(out.probs, probs, atol=1e-5)
    assert out.n_tokens[1] < out.n_tokens[2] < out.n_tokens[0]


@pytest.mark.parametrize("id2label", [PROSUS_LABELS, TONE_LABELS])
def test_embed_sentences_maps_probabilities_by_label(tiny_model, tiny_tokenizer, id2label):
    model = tiny_model(id2label)
    out = embed_sentences([SHORT, MEDIUM], model, tiny_tokenizer, CPU)

    batch = tiny_tokenizer([SHORT, MEDIUM], padding=True, return_tensors="pt")
    with torch.no_grad():
        raw = torch.softmax(model(**batch).logits, dim=-1).numpy()
    np.testing.assert_allclose(out.probs, raw[:, label_order(id2label)], atol=1e-5)
    np.testing.assert_allclose(out.probs.sum(axis=1), 1.0, atol=1e-5)


def test_embed_sentences_truncates(tiny_model, tiny_tokenizer):
    out = embed_sentences([SHORT, LONG], tiny_model(), tiny_tokenizer, CPU, max_length=8)
    assert out.n_tokens.tolist() == [4, 8]
    assert out.n_truncated == 1


def test_embed_sentences_empty(tiny_model, tiny_tokenizer):
    out = embed_sentences([], tiny_model(), tiny_tokenizer, CPU)
    assert out.emb.shape == (0, 32) and out.probs.shape == (0, 3)
    assert out.n_tokens.shape == (0,) and out.n_truncated == 0


def test_embed_sentences_reports_progress(tiny_model, tiny_tokenizer):
    seen = []
    embed_sentences([SHORT, MEDIUM, LONG], tiny_model(), tiny_tokenizer, CPU, batch_size=2,
                    on_batch=lambda done, total: seen.append((done, total)))  # fmt: skip
    assert seen == [(1, 2), (2, 2)]


# --- aggregate_calls ---


def make_output():
    """Three sentences: two for call A, one for call C; call B has none."""
    sentences = pd.DataFrame(
        {"event_id": ["A", "A", "C"], "sent_idx": [0, 1, 0], "text": ["a0", "a1", "c0"]}
    )
    out = SentenceOutput(
        emb=np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 5.0]], dtype=np.float32),
        probs=np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]], dtype=np.float32
        ),  # A: pos, neu ; C: neg
        n_tokens=np.array([5, 6, 7]),
        n_truncated=0,
    )
    return sentences, out


def test_aggregate_calls():
    sentences, out = make_output()

    calls, call_emb = aggregate_calls(["A", "B", "C"], sentences, out)

    assert list(calls.columns) == [
        "event_id", "n_sentences", "p_pos", "p_neg", "p_neu", "tone", "frac_pos", "frac_neg",
    ]  # fmt: skip
    assert calls["event_id"].tolist() == ["A", "B", "C"]
    a = calls.iloc[0]
    assert a["n_sentences"] == 2
    assert a["p_pos"] == pytest.approx(0.4)
    assert a["p_neg"] == pytest.approx(0.25)
    assert a["tone"] == pytest.approx(0.15)  # mean of (0.5, -0.2)
    assert (a["frac_pos"], a["frac_neg"]) == (0.5, 0.0)
    c = calls.iloc[2]
    assert c["tone"] == pytest.approx(-0.3)
    assert (c["frac_pos"], c["frac_neg"]) == (0.0, 1.0)

    assert call_emb.dtype == np.float32 and call_emb.shape == (3, 2)
    np.testing.assert_allclose(call_emb[0], [2.0, 4.0])
    np.testing.assert_allclose(call_emb[2], [5.0, 5.0])


def test_aggregate_calls_empty_call():
    sentences, out = make_output()

    calls, call_emb = aggregate_calls(["A", "B", "C"], sentences, out)

    b = calls.iloc[1]
    assert b["event_id"] == "B" and b["n_sentences"] == 0
    assert all(math.isnan(b[c]) for c in ["p_pos", "p_neg", "p_neu", "tone", "frac_pos", "frac_neg"])
    assert np.isnan(call_emb[1]).all()

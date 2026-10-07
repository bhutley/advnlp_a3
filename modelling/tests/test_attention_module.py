import numpy as np
import pandas as pd
import pytest
import torch

from attention import AttentionPool, load_sentences, pad
from conftest import AUDIO_DIM, TEXT_DIM, make_project, n_sentences


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    ids = make_project(root, n_firms=4, calls_per_firm=3)
    return root, ids


# --- load_sentences ---


def test_load_sentences_alignment(project):
    root, ids = project
    text, audio = load_sentences(root, ids)

    assert len(text) == len(audio) == len(ids)
    for i, (t, a) in enumerate(zip(text, audio)):
        # Sentence 0 of every call is not audio_ok, so it is dropped from both modalities.
        assert t.shape == (n_sentences(i) - 1, TEXT_DIM) and a.shape == (n_sentences(i) - 1, AUDIO_DIM)
        assert t.dtype == a.dtype == np.float32 and np.isfinite(a).all()

    # Values follow (event_id, sent_idx), not file order: check one sentence by hand.
    rows = pd.read_csv(root / "text_embeddings/data/finbert_tone/sentences.csv")
    emb = np.load(root / "text_embeddings/data/finbert_tone/sentence_emb.npy")
    r = rows.index[(rows["event_id"] == ids[2]) & (rows["sent_idx"] == 3)][0]
    np.testing.assert_array_equal(text[2][2], emb[r].astype(np.float32))  # sent_idx 3 → row 2

    rows = pd.read_csv(root / "audio_embeddings/data/hubert/sentences.csv")
    emb = np.load(root / "audio_embeddings/data/hubert/sentence_emb.npy")
    r = rows.index[(rows["event_id"] == ids[2]) & (rows["sent_idx"] == 3)][0]
    np.testing.assert_array_equal(audio[2][2], emb[r].astype(np.float32))


def test_load_sentences_order_follows_event_ids(project):
    root, ids = project
    text, _ = load_sentences(root, ids[::-1])
    forward, _ = load_sentences(root, ids)
    np.testing.assert_array_equal(text[0], forward[-1])


def test_load_sentences_missing_text_row_raises(project, tmp_path):
    root, ids = project
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    path = copy / "text_embeddings/data/finbert_tone/sentences.csv"
    rows = pd.read_csv(path)
    drop = rows.index[(rows["event_id"] == ids[0]) & (rows["sent_idx"] == 1)][0]
    rows.loc[drop, "sent_idx"] = 999  # the audio_ok sentence (ids[0], 1) now has no text row
    rows.to_csv(path, index=False)

    with pytest.raises(ValueError, match="text"):
        load_sentences(copy, ids)


# --- pad ---


def test_pad():
    x, mask = pad([np.ones((2, 3)), np.full((4, 3), 2.0)])
    assert x.shape == (2, 4, 3) and mask.tolist() == [[True, True, False, False], [True] * 4]
    assert x[0, 2:].abs().sum() == 0


# --- AttentionPool ---


def batch(seed=0, lengths=(3, 5, 1), d_text=4, d_audio=3, k=2):
    rng = np.random.default_rng(seed)
    text = [rng.normal(size=(n, d_text)) for n in lengths]
    audio = [rng.normal(size=(n, d_audio)) for n in lengths]
    small = torch.tensor(rng.normal(size=(len(lengths), k)), dtype=torch.float32)
    return text, audio, small


def test_uniform_attention_is_masked_mean():
    text, audio, small = batch()
    model = AttentionPool(4, 3, 2, learn_attention=False)
    t, mask = pad(text)
    a, _ = pad(audio)

    p_text, p_audio = model.pooled(t, a, mask)

    for i in range(len(text)):
        np.testing.assert_allclose(p_text[i].detach().numpy(), text[i].mean(0), atol=1e-6)
        np.testing.assert_allclose(p_audio[i].detach().numpy(), audio[i].mean(0), atol=1e-6)
    assert not model.w_text.requires_grad and not model.w_text.abs().sum()


def test_padding_invariance():
    torch.manual_seed(0)
    text, audio, small = batch()
    model = AttentionPool(4, 3, 2, learn_attention=True)
    t, mask = pad(text)
    a, _ = pad(audio)
    t_long, mask_long = pad(text, length=9)  # extra padding
    a_long, _ = pad(audio, length=9)

    np.testing.assert_allclose(model(t, a, mask, small).detach().numpy(),
                               model(t_long, a_long, mask_long, small).detach().numpy(), atol=1e-6)  # fmt: skip


def test_weights_sum_to_one_and_ignore_padding():
    torch.manual_seed(0)
    text, audio, _ = batch()
    model = AttentionPool(4, 3, 2, learn_attention=True)
    t, mask = pad(text)
    a, _ = pad(audio)

    w_text, w_audio = model.weights(t, a, mask)

    for w in (w_text, w_audio):
        np.testing.assert_allclose(w.sum(1).detach().numpy(), 1.0, atol=1e-6)
        assert (w[~mask] == 0).all()
    assert not torch.allclose(w_text[1, :5], torch.full((5,), 0.2))  # learned ≠ uniform

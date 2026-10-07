from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import tone
from ssl_embed import (
    MIN_SAMPLES,
    SENTENCE_LAYER,
    SSL_MODELS,
    check_extractor,
    layer_means,
    load_model,
    n_frames,
    normalize,
)

CPU = torch.device("cpu")


def test_registry_is_pinned():
    assert set(SSL_MODELS) == {"wav2vec2", "hubert"}
    for name, revision, _ in SSL_MODELS.values():
        assert name.startswith("facebook/") and len(revision) == 40
    assert SENTENCE_LAYER == 6


def test_normalize():
    x = tone(200, 1.0) * 5 + 2
    z = normalize(x)
    assert z.dtype == np.float32
    assert abs(z.mean()) < 1e-5 and z.std() == pytest.approx(1.0, abs=1e-4)


# --- check_extractor ---


def extractor(**overrides):
    fields = {"sampling_rate": 16000, "return_attention_mask": False, "do_normalize": True}
    return SimpleNamespace(**fields | overrides)


def test_check_extractor_returns_do_normalize():
    assert check_extractor(extractor()) is True
    assert check_extractor(extractor(do_normalize=False)) is False


@pytest.mark.parametrize("bad", [{"sampling_rate": 8000}, {"return_attention_mask": True}])
def test_check_extractor_rejects(bad):
    with pytest.raises(ValueError):
        check_extractor(extractor(**bad))


def test_load_model_unknown_key():
    with pytest.raises(KeyError):
        load_model("whisper", CPU)


# --- layer_means ---


def test_layer_means_shape(tiny_ssl):
    out = layer_means(tone(200, 1.0), tiny_ssl(), CPU, do_normalize=True)
    assert out.shape == (3, 16) and out.dtype == np.float32


def test_layer_means_is_frame_mean_of_each_layer(tiny_ssl):
    model = tiny_ssl()
    audio = tone(200, 1.0)

    out = layer_means(audio, model, CPU, do_normalize=False)

    with torch.no_grad():
        hidden = model(torch.from_numpy(audio)[None], output_hidden_states=True).hidden_states
    expected = np.stack([h[0].mean(dim=0).numpy() for h in hidden])
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_layer_means_normalization(tiny_ssl):
    model = tiny_ssl()
    x = tone(200, 1.0) + 0.05 * tone(310, 1.0)
    shifted = 3 * x + 0.1

    same = layer_means(x, model, CPU, do_normalize=True)
    np.testing.assert_allclose(layer_means(shifted, model, CPU, do_normalize=True), same, atol=1e-4)

    raw = layer_means(x, model, CPU, do_normalize=False)
    assert not np.allclose(layer_means(shifted, model, CPU, do_normalize=False), raw, atol=1e-4)


def test_layer_means_too_short(tiny_ssl):
    assert layer_means(np.zeros(MIN_SAMPLES - 1, np.float32), tiny_ssl(), CPU, True) is None


@pytest.mark.parametrize("n_samples", [400, 16000, 23456])
def test_n_frames_matches_model(tiny_ssl, n_samples):
    model = tiny_ssl()
    with torch.no_grad():
        out = model(torch.zeros(1, n_samples), output_hidden_states=True)
    assert n_frames(n_samples, model.config) == out.hidden_states[-1].shape[1]

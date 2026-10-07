"""Synthetic audio, MP3 encoding in the dataset's format, and a tiny wav2vec2 model."""

import subprocess

import numpy as np
import pytest
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

DATASET_SR = 11025  # the dataset's lowest-quality common format (92 calls): 11,025 Hz, 16 kbps
DATASET_BITRATE = "16k"


def tone(freq: float, seconds: float, sr: int = 16000, amp: float = 0.3) -> np.ndarray:
    """A float32 sine wave."""
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def make_mp3(path, audio: np.ndarray, sr: int = 16000) -> None:
    """Encode float audio in [-1, 1] to MP3 the way the dataset's clips are encoded."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "-",
         "-ar", str(DATASET_SR), "-b:a", DATASET_BITRATE, str(path)],
        input=pcm, check=True,
    )  # fmt: skip


@pytest.fixture
def tiny_ssl():
    """Factory: a seeded 2-layer Wav2Vec2Model with a 20-sample hop (strides 5 × 4), in eval mode."""

    def make():
        torch.manual_seed(0)
        config = Wav2Vec2Config(
            conv_dim=(8, 8),
            conv_stride=(5, 4),
            conv_kernel=(10, 8),
            num_feat_extract_layers=2,
            feat_extract_norm="group",
            do_stable_layer_norm=False,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=2,
        )
        return Wav2Vec2Model(config).eval()

    return make

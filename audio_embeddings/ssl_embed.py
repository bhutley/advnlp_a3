"""wav2vec2 / HuBERT embeddings: every hidden state, mean-pooled over frames.

Both checkpoints normalise inside their first convolution over the whole input and take
no attention mask, so clips are embedded one at a time: padding a batch would change the
outputs of its shorter clips.
"""

import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model

from decode import SAMPLE_RATE

# key → (Hugging Face checkpoint, pinned commit, model class)
SSL_MODELS = {
    "wav2vec2": ("facebook/wav2vec2-base", "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8", Wav2Vec2Model),
    "hubert": ("facebook/hubert-base-ls960", "dba3bb02fda4248b6e082697eee756de8fe8aa8a", HubertModel),
}
SENTENCE_LAYER = 6  # of hidden states 0 (CNN output) .. 12; fixed before seeing any targets
MIN_SAMPLES = 400  # the convolutional front end needs at least one 25 ms window


def normalize(audio: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance, as Wav2Vec2FeatureExtractor does when do_normalize is set."""
    x = audio.astype(np.float64)
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)


def check_extractor(extractor) -> bool:
    """Raise unless the feature extractor expects 16 kHz and no attention mask.

    Returns its do_normalize flag.
    """
    if extractor.sampling_rate != SAMPLE_RATE:
        raise ValueError(f"model expects {extractor.sampling_rate} Hz, decoder gives {SAMPLE_RATE}")
    if extractor.return_attention_mask:
        raise ValueError("model expects an attention mask; one-clip-at-a-time design assumes none")
    return bool(extractor.do_normalize)


def load_model(key: str, device: torch.device):
    """(model in eval mode on device, do_normalize) for an SSL_MODELS key at its revision."""
    name, revision, model_class = SSL_MODELS[key]
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(name, revision=revision)
    do_normalize = check_extractor(extractor)
    model = model_class.from_pretrained(name, revision=revision)
    return model.eval().to(device), do_normalize


def n_frames(n_samples: int, config) -> int:
    """Frames the convolutional front end produces for n_samples of audio."""
    for kernel, stride in zip(config.conv_kernel, config.conv_stride):
        n_samples = (n_samples - kernel) // stride + 1
    return n_samples


@torch.no_grad()
def layer_means(
    audio: np.ndarray, model, device: torch.device, do_normalize: bool
) -> np.ndarray | None:
    """(n_layers + 1, hidden) float32: each hidden state averaged over frames.

    None if the clip is shorter than MIN_SAMPLES.
    """
    if len(audio) < MIN_SAMPLES:
        return None
    x = normalize(audio) if do_normalize else audio.astype(np.float32)
    out = model(torch.from_numpy(x)[None].to(device), output_hidden_states=True)
    return torch.stack([h[0].mean(dim=0) for h in out.hidden_states]).float().cpu().numpy()

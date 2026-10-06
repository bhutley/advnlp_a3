"""FinBERT models, sentence embeddings and call-level aggregation.

Both models are BertForSequenceClassification checkpoints with three sentiment labels,
but they order those labels differently, so probabilities are always mapped by label
name (label_order), never by index.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# key → (Hugging Face checkpoint, pinned commit)
MODELS = {
    "finbert_tone": ("yiyanghkust/finbert-tone", "4921590d3c0c3832c0efea24c8381ce0bda7844b"),
    "prosus_finbert": ("ProsusAI/finbert", "4556d13015211d73dccd3fdd39d39232506f3e43"),
}
PROB_COLUMNS = ["p_pos", "p_neg", "p_neu"]
LABEL_NAMES = ["positive", "negative", "neutral"]  # same order as PROB_COLUMNS


def label_order(id2label: dict) -> list[int]:
    """The logit index for each of PROB_COLUMNS.

    Raises ValueError unless the lowercased labels are exactly positive, negative and
    neutral, so a model with different labels can never be mapped silently.
    """
    by_name = {str(name).lower(): int(idx) for idx, name in id2label.items()}
    if len(id2label) != len(LABEL_NAMES) or set(by_name) != set(LABEL_NAMES):
        raise ValueError(f"expected labels {LABEL_NAMES}, got {id2label}")
    return [by_name[name] for name in LABEL_NAMES]


def pick_device(pref: str) -> torch.device:
    """'auto' → mps if available, else cpu. 'mps' raises if unavailable. 'cpu'."""
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "mps":
            raise RuntimeError("--device mps requested, but MPS is not available")
        return torch.device("cpu")
    raise ValueError(f"unknown device {pref!r}; use auto, mps or cpu")


def check_vocab(model, tokenizer) -> None:
    """Raise if the tokenizer's vocabulary doesn't match the model's embedding table.

    A mismatched tokenizer doesn't fail on its own; it silently yields wrong token ids.
    """
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(
            f"tokenizer vocab size {len(tokenizer)} != model vocab_size "
            f"{model.config.vocab_size}"
        )


def load_model(key: str, device: torch.device):
    """(model, tokenizer) for a MODELS key at its pinned revision, in eval mode on device.

    Uses the Bert* classes rather than Auto*: finbert-tone's config.json has no
    model_type, which transformers 5 requires for Auto* resolution. Both checkpoints
    declare BertForSequenceClassification.
    """
    name, revision = MODELS[key]
    tokenizer = BertTokenizer.from_pretrained(name, revision=revision)
    model = BertForSequenceClassification.from_pretrained(name, revision=revision)
    check_vocab(model, tokenizer)
    label_order(model.config.id2label)  # fail now, not after embedding 90k sentences
    return model.eval().to(device), tokenizer


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of token vectors over unmasked positions: (batch, seq, dim) → (batch, dim)."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


@dataclass
class SentenceOutput:
    emb: np.ndarray  # (n, hidden) float32, mean-pooled last layer
    probs: np.ndarray  # (n, 3) float32, columns in PROB_COLUMNS order
    n_tokens: np.ndarray  # (n,) tokens fed to the model, after truncation
    n_truncated: int  # sentences longer than max_length


@torch.no_grad()
def embed_sentences(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 512,
    on_batch: Callable[[int, int], None] | None = None,
) -> SentenceOutput:
    """Embed and score sentences, returning results in the order of texts.

    Sentences are batched in length order to minimise padding. max_length must always be
    passed explicitly: finbert-tone's tokenizer sets no model_max_length of its own.
    on_batch(done, total) is called after each batch.
    """
    hidden_size = model.config.hidden_size
    order = label_order(model.config.id2label)
    ids = tokenizer(texts)["input_ids"] if texts else []  # the tokenizer rejects []
    full_lengths = np.array([len(i) for i in ids], dtype=int)
    by_length = np.argsort(full_lengths, kind="stable")
    n_batches = -(-len(texts) // batch_size)

    emb = np.empty((len(texts), hidden_size), dtype=np.float32)
    probs = np.empty((len(texts), len(PROB_COLUMNS)), dtype=np.float32)
    for b in range(n_batches):
        idx = by_length[b * batch_size : (b + 1) * batch_size]
        batch = tokenizer(
            [texts[i] for i in idx],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**batch, output_hidden_states=True)
        emb[idx] = mean_pool(out.hidden_states[-1], batch["attention_mask"]).float().cpu().numpy()
        probs[idx] = torch.softmax(out.logits.float(), dim=-1)[:, order].cpu().numpy()
        if on_batch:
            on_batch(b + 1, n_batches)

    return SentenceOutput(
        emb=emb,
        probs=probs,
        n_tokens=np.minimum(full_lengths, max_length),
        n_truncated=int((full_lengths > max_length).sum()),
    )


CALL_COLUMNS = ["event_id", "n_sentences"] + PROB_COLUMNS + ["tone", "frac_pos", "frac_neg"]


def aggregate_calls(
    event_ids: list[str], sentences: pd.DataFrame, out: SentenceOutput
) -> tuple[pd.DataFrame, np.ndarray]:
    """Call-level sentiment and mean sentence vector, one row per event_id, in order.

    tone is mean(p_pos − p_neg); frac_pos / frac_neg are the shares of sentences whose
    most likely label is positive / negative. A call without sentences gets NaNs.
    sentences must be row-aligned with out.
    """
    positions = sentences.groupby("event_id").indices  # event_id → row positions
    top = out.probs.argmax(axis=1)  # 0 = positive, 1 = negative, 2 = neutral
    rows = []
    call_emb = np.full((len(event_ids), out.emb.shape[1]), np.nan, dtype=np.float32)
    for i, event_id in enumerate(event_ids):
        idx = positions.get(event_id, np.array([], dtype=int))
        row = dict.fromkeys(CALL_COLUMNS, np.nan) | {"event_id": event_id, "n_sentences": len(idx)}
        if len(idx):
            p = out.probs[idx]
            row |= dict(zip(PROB_COLUMNS, p.mean(axis=0)))
            row |= {
                "tone": float((p[:, 0] - p[:, 1]).mean()),
                "frac_pos": float((top[idx] == 0).mean()),
                "frac_neg": float((top[idx] == 1).mean()),
            }
            call_emb[i] = out.emb[idx].mean(axis=0)
        rows.append(row)
    return pd.DataFrame(rows, columns=CALL_COLUMNS), call_emb

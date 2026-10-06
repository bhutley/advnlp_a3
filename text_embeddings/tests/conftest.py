"""A tiny random BERT and tokenizer, built offline, standing in for FinBERT in tests."""

import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = (
    "revenue grew strongly and we raised guidance missed expectations cut our outlook "
    "margin the quarter was in line with flat demand weak strong growth costs higher lower"
).split()

PROSUS_LABELS = {0: "positive", 1: "negative", 2: "neutral"}
TONE_LABELS = {0: "Neutral", 1: "Positive", 2: "Negative"}


@pytest.fixture
def tiny_tokenizer():
    # transformers 5 takes the vocabulary as `vocab`; an old-style `vocab_file=` is
    # silently ignored and leaves only the special tokens.
    tokens = SPECIAL + sorted(set(WORDS))
    return BertTokenizer(vocab={t: i for i, t in enumerate(tokens)})


@pytest.fixture
def tiny_model(tiny_tokenizer):
    """Factory: tiny_model(id2label) → a seeded BertForSequenceClassification in eval mode."""

    def make(id2label=PROSUS_LABELS):
        torch.manual_seed(0)
        config = BertConfig(
            vocab_size=tiny_tokenizer.vocab_size,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=64,
            num_labels=3,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
        )
        return BertForSequenceClassification(config).eval()

    return make

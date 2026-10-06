import torch


def test_tiny_model_forward(tiny_model, tiny_tokenizer):
    model = tiny_model()
    batch = tiny_tokenizer(
        ["revenue grew strongly", "we missed expectations and cut our outlook"],
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)

    assert out.logits.shape == (2, 3)
    assert out.hidden_states[-1].shape[:2] == batch["input_ids"].shape
    assert out.hidden_states[-1].shape[2] == 32
    # Known words tokenize to themselves, not [UNK].
    assert tiny_tokenizer.unk_token_id not in batch["input_ids"][0].tolist()
    assert batch["attention_mask"][0].sum() < batch["attention_mask"][1].sum()  # padded

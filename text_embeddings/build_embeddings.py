"""Embed every transcript sentence with one FinBERT model and aggregate to calls.

Writes <out-dir>/<model>/: sentences.csv + sentence_emb.npy (float16), row-aligned;
calls.csv + call_emb.npy (float32), row-aligned; and meta.json describing the run.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers

import embed
from transcripts import read_sentences

PROGRESS_EVERY = 100  # batches
POOLING = "mean of the last hidden layer over attention-masked tokens, incl. [CLS] and [SEP]"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(embed.MODELS))
    parser.add_argument(
        "--dataset", type=Path, default=Path.home() / "var/data/earnings_calls/ACL19_Release"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto", help="auto, mps or cpu (default auto)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit-calls", type=int, help="only the first N calls, for a trial run")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output dir")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model_dir = args.out_dir / args.model
    if model_dir.exists() and not args.force:
        print(f"error: {model_dir} exists; use --force to overwrite", file=sys.stderr)
        return 1

    start = time.perf_counter()
    event_ids, sentences = read_sentences(args.dataset, args.limit_calls)
    print(f"{len(event_ids)} calls, {len(sentences)} sentences")
    device = embed.pick_device(args.device)
    model, tokenizer = embed.load_model(args.model, device)
    print(f"model {args.model} on {device}")

    def progress(done, total):
        if done % PROGRESS_EVERY == 0 or done == total:
            print(f"  batch {done}/{total} ({time.perf_counter() - start:.0f}s)")

    out = embed.embed_sentences(
        sentences["text"].tolist(), model, tokenizer, device,
        batch_size=args.batch_size, max_length=args.max_length, on_batch=progress,
    )  # fmt: skip
    calls, call_emb = embed.aggregate_calls(event_ids, sentences, out)

    model_dir.mkdir(parents=True, exist_ok=True)
    table = sentences[["event_id", "sent_idx"]].assign(n_tokens=out.n_tokens)
    table[embed.PROB_COLUMNS] = out.probs
    table.to_csv(model_dir / "sentences.csv", index=False)
    np.save(model_dir / "sentence_emb.npy", out.emb.astype(np.float16))
    calls.to_csv(model_dir / "calls.csv", index=False)
    np.save(model_dir / "call_emb.npy", call_emb)

    checkpoint, revision = embed.MODELS[args.model]
    runtime = time.perf_counter() - start
    meta = {
        "model": args.model,
        "checkpoint": checkpoint,
        "revision": revision,
        "pooling": POOLING,
        "sentence_emb_dtype": "float16",
        "call_emb_dtype": "float32",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "n_calls": len(event_ids),
        "n_sentences": len(sentences),
        "n_truncated": out.n_truncated,
        "n_empty_calls": int((calls["n_sentences"] == 0).sum()),
        "runtime_s": round(runtime, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print()
    print(f"written to {model_dir}")
    for key in ("n_calls", "n_sentences", "n_truncated", "n_empty_calls", "runtime_s"):
        print(f"{key}: {meta[key]}")
    for event_id in calls.loc[calls["n_sentences"] == 0, "event_id"]:
        print(f"warning: {event_id} has no sentences")
    return 0


if __name__ == "__main__":
    sys.exit(main())

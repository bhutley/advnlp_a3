"""Frozen embeddings from segments.csv: call_id,text_path,audio_path.

Each row pairs a UTF-8 text segment with its audio file. Paths are relative to
the manifest. Multiple segments per call are pooled into one vector per modality.
Requires torch, transformers, soundfile, scipy, numpy, pandas.

The cache is saved every --save-every calls and reloaded on restart, so an
interrupted run resumes instead of starting over.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import (AutoModel, AutoTokenizer, BertModel, BertTokenizer,
                          Wav2Vec2FeatureExtractor)


def commit_hash(model, revision):
    """Resolve a branch name such as "main" to the cached commit it points at."""
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(model, revision=revision, local_files_only=True)).name
    except Exception:
        return revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-model", default="yiyanghkust/finbert-tone")
    parser.add_argument("--audio-model", default="facebook/wav2vec2-base")
    # Pin checkpoints so the representation is reproducible; "main" records the
    # resolved commit hash in the cache.
    parser.add_argument("--text-revision", default="main")
    parser.add_argument("--audio-revision", default="main")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.text_model, revision=args.text_revision)
        text_model = AutoModel.from_pretrained(args.text_model, revision=args.text_revision)
    except ValueError:
        # finbert-tone ships only vocab.txt and a config without model_type,
        # which the Auto classes cannot resolve; it is a BERT encoder.
        tokenizer = BertTokenizer.from_pretrained(args.text_model, revision=args.text_revision)
        text_model = BertModel.from_pretrained(args.text_model, revision=args.text_revision)
    text_model = text_model.to(device).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.audio_model, revision=args.audio_revision)
    audio_model = AutoModel.from_pretrained(args.audio_model, revision=args.audio_revision).to(device).eval()
    revisions = {
        "text_model": args.text_model,
        "text_revision": commit_hash(args.text_model, args.text_revision),
        "audio_model": args.audio_model,
        "audio_revision": commit_hash(args.audio_model, args.audio_revision),
    }
    manifest = Path(args.segments)
    segments = pd.read_csv(manifest, dtype={"call_id": str})
    if segments[["call_id", "text_path", "audio_path"]].isna().any().any():
        raise ValueError("Every segment needs an ID, text path and audio path")
    ids, text_vectors, audio_vectors, audio_skipped = [], [], [], []
    out = Path(args.out)
    if out.exists():
        with np.load(out, allow_pickle=False) as cache:
            stored = {k: str(cache[k]) for k in revisions if k in cache}
            if stored != {k: str(v) for k, v in revisions.items()}:
                raise ValueError(f"{out} was made with different models {stored}; use a new --out")
            ids = list(cache["call_id"].astype(str))
            text_vectors, audio_vectors = list(cache["text"]), list(cache["audio"])
            # Caches written before skips were recorded had none: a skip used to be fatal.
            audio_skipped = list(cache["audio_skipped"]) if "audio_skipped" in cache else [0] * len(ids)
        print(f"Resuming: {len(ids)} calls already in {out}", flush=True)
    done = set(ids)

    def save():
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, call_id=np.asarray(ids), text=np.asarray(text_vectors),
                            audio=np.asarray(audio_vectors),
                            audio_skipped=np.asarray(audio_skipped, dtype=int), **revisions)
        tmp.replace(out)

    with torch.inference_mode():
        for call_id, group in segments.groupby("call_id", sort=True):
            if call_id in done:
                continue
            text_sum, audio_sum = None, None
            n_tokens, n_samples, skipped = 0, 0, 0
            for row in group.itertuples(index=False):
                text = (manifest.parent / row.text_path).read_text(encoding="utf-8")
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if not token_ids:
                    raise ValueError(f"{call_id}: empty transcript segment")
                for start in range(0, len(token_ids), 510):
                    chunk = token_ids[start:start + 510]
                    chunk_ids = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]
                    tokens = torch.tensor([chunk_ids], device=device)
                    hidden = text_model(input_ids=tokens).last_hidden_state[0, 1:-1]
                    vector = hidden.sum(0).cpu().numpy()
                    text_sum = vector if text_sum is None else text_sum + vector
                    n_tokens += len(chunk)
                audio_path = manifest.parent / row.audio_path
                if audio_path.stat().st_size == 0:
                    skipped += 1  # the release has empty clips; same handling as below
                    continue
                waveform, rate = sf.read(audio_path, dtype="float32", always_2d=True)
                waveform = waveform.mean(axis=1)
                if rate != 16000:
                    from math import gcd
                    divisor = gcd(rate, 16000)
                    waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
                if not np.isfinite(waveform).all():
                    raise ValueError(f"{call_id}: audio must be finite")
                if len(waveform) < 1600:
                    # A sub-0.1 s clip is a forced-alignment failure: keep the
                    # sentence's text, drop its audio, and record the skip.
                    skipped += 1
                    continue
                # Process at most ten seconds at once; merge a short final tail.
                chunks = list(np.array_split(waveform, max(1, int(np.ceil(len(waveform) / 160000)))))
                for chunk in chunks:
                    inputs = extractor(chunk, sampling_rate=16000, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    vector = audio_model(**inputs).last_hidden_state[0].mean(0).cpu().numpy()
                    weighted = vector * len(chunk)
                    audio_sum = weighted if audio_sum is None else audio_sum + weighted
                    n_samples += len(chunk)
            if audio_sum is None:
                raise ValueError(f"{call_id}: no usable audio segments")
            ids.append(call_id)
            text_vectors.append(text_sum / n_tokens)
            audio_vectors.append(audio_sum / n_samples)
            audio_skipped.append(skipped)
            if skipped:
                print(f"  {call_id}: skipped {skipped} empty or sub-0.1 s audio clips", flush=True)
            print(f"Embedded {call_id} ({len(ids)}/{segments.call_id.nunique()})", flush=True)
            if len(ids) % args.save_every == 0:
                save()
    if not ids:
        raise ValueError("Manifest contains no segments")
    save()


if __name__ == "__main__":
    main()

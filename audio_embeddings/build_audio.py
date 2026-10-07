"""Extract one audio feature set for every clip and aggregate to calls.

--features wav2vec2 | hubert: data/<features>/ sentences.csv + sentence_emb.npy (layer
ssl_embed.SENTENCE_LAYER, float16), calls.csv + call_emb.npy (all hidden states,
float16), meta.json.
--features praat: data/praat/ sentences.csv, calls.csv, meta.json.

Clips are decoded (and, for praat, measured) in a process pool; the SSL model runs in
the main process. A clip that fails to decode or measure becomes a NaN row and is listed.

Every sentence row has audio_ok: decoded, and no faster than clips.MAX_WORDS_PER_SEC for
its transcript line (a few dataset clips are truncated). Call aggregates use only
audio_ok clips; calls.csv records n_bad and audio_call_ok (clips.call_ok).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from itertools import batched
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import torch
import transformers

import ssl_embed
from clips import MAX_BAD_SHARE, MAX_WORDS_PER_SEC, call_ok, list_clips, plausible_rate
from decode import SAMPLE_RATE, check_ffmpeg, decode_clip
from praat import PRAAT_COLUMNS, praat_features

FEATURES = ["hubert", "praat", "wav2vec2"]
BLOCK = 512  # clips submitted to the pool at a time; bounds decoded audio held in memory
PROGRESS_EVERY = 2000  # clips


def default_workers() -> int:
    """Performance-core count on Apple Silicon, else all CPUs."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"],
                             capture_output=True, text=True, check=True).stdout  # fmt: skip
        return int(out)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return os.cpu_count() or 1


def pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "mps":
        raise RuntimeError("--device mps requested, but MPS is not available")
    if pref == "auto":
        return torch.device("cpu")
    raise ValueError(f"unknown device {pref!r}; use auto, mps or cpu")


# Worker tasks: top level so the spawn start method can pickle them.
def _decode_task(path: str) -> np.ndarray | None:
    return decode_clip(Path(path))


def _praat_task(item: tuple[str, int]) -> dict | None:
    """Praat features, or None if the clip can't be decoded or Praat rejects it."""
    path, n_words = item
    audio = decode_clip(Path(path))
    if audio is None:
        return None
    try:
        return praat_features(audio, SAMPLE_RATE, n_words)
    except parselmouth.PraatError as e:
        print(f"warning: Praat failed on {path}: {str(e).splitlines()[0]}", flush=True)
        return None


def run_tasks(fn, items: list, workers: int):
    """fn over items in order: in-process if workers == 0, else in a pool fed BLOCK at a time."""
    if workers == 0:
        yield from map(fn, items)
        return
    with Pool(workers) as pool:
        for block in batched(items, BLOCK):
            yield from pool.imap(fn, block, chunksize=8)


def progress(i: int, total: int, start: float) -> None:
    if i % PROGRESS_EVERY == 0 or i == total:
        print(f"  {i}/{total} clips ({time.perf_counter() - start:.0f}s)", flush=True)


def quality(sentences: pd.DataFrame) -> pd.DataFrame:
    """Per call, in order of appearance: n_sentences (audio_ok clips), n_bad, audio_call_ok."""
    g = sentences.groupby("event_id", sort=False)["audio_ok"]
    calls = pd.DataFrame({"n_sentences": g.sum(), "n_bad": g.size() - g.sum()}).reset_index()
    calls["audio_call_ok"] = [call_ok(b, n + b) for n, b in zip(calls["n_sentences"], calls["n_bad"])]
    return calls


def build_ssl(clips: pd.DataFrame, key: str, device, workers: int, start: float):
    """Sentence rows, layer-SENTENCE_LAYER sentence vectors, per-call all-layer means."""
    model, do_normalize = ssl_embed.load_model(key, device)
    layer = ssl_embed.SENTENCE_LAYER
    n_layers, hidden = model.config.num_hidden_layers + 1, model.config.hidden_size
    sentence_emb = np.full((len(clips), hidden), np.nan, dtype=np.float16)
    sums = np.zeros((clips["event_id"].nunique(), n_layers, hidden), dtype=np.float64)
    call_pos = {e: i for i, e in enumerate(clips["event_id"].unique())}
    counts = np.zeros(len(call_pos), dtype=int)
    rows, failed = [], []

    for i, (clip, audio) in enumerate(
        zip(clips.itertuples(index=False), run_tasks(_decode_task, clips["path"].tolist(), workers)),
        start=1,
    ):
        means = None if audio is None else ssl_embed.layer_means(audio, model, device, do_normalize)
        duration = np.nan if audio is None else len(audio) / SAMPLE_RATE
        ok = means is not None and plausible_rate(clip.n_words, duration)
        rows.append({"event_id": clip.event_id, "sent_idx": clip.sent_idx, "duration_s": duration,
                     "n_frames": 0 if means is None else ssl_embed.n_frames(len(audio), model.config),
                     "audio_ok": ok})  # fmt: skip
        if means is None:
            failed.append(clip.path)
        else:
            sentence_emb[i - 1] = means[layer]  # kept even if implausible; audio_ok flags it
        if ok:
            sums[call_pos[clip.event_id]] += means
            counts[call_pos[clip.event_id]] += 1
        progress(i, len(clips), start)

    sentences = pd.DataFrame(rows)
    with warnings.catch_warnings():  # a call whose clips all failed divides 0/0 → NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        call_emb = (sums / counts[:, None, None]).astype(np.float16)
    calls = quality(sentences).assign(
        duration_s=sentences.groupby("event_id", sort=False)["duration_s"].sum().to_numpy()
    )
    return sentences, sentence_emb, calls, call_emb, failed, layer


def build_praat(clips: pd.DataFrame, workers: int, start: float):
    items = list(zip(clips["path"], clips["n_words"].astype(int)))
    rows, failed = [], []
    for i, (clip, feats) in enumerate(
        zip(clips.itertuples(index=False), run_tasks(_praat_task, items, workers)), start=1
    ):
        if feats is None:
            failed.append(clip.path)
            feats = dict.fromkeys(PRAAT_COLUMNS, np.nan) | {"n_words": clip.n_words}
        ok = feats is not None and plausible_rate(clip.n_words, feats["duration_s"])
        rows.append({"event_id": clip.event_id, "sent_idx": clip.sent_idx} | feats | {"audio_ok": ok})
        progress(i, len(clips), start)

    sentences = pd.DataFrame(rows, columns=["event_id", "sent_idx"] + PRAAT_COLUMNS + ["audio_ok"])
    means = sentences[sentences["audio_ok"]].groupby("event_id", sort=False)[PRAAT_COLUMNS].mean()
    calls = quality(sentences).merge(means, on="event_id", how="left")  # pandas means skip NaN
    return sentences, calls, failed


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, choices=FEATURES)
    parser.add_argument(
        "--dataset", type=Path, default=Path.home() / "var/data/earnings_calls/ACL19_Release"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", default="auto", help="auto, mps or cpu (SSL features only)")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="decode processes; 0 = in-process (default: performance cores)")  # fmt: skip
    parser.add_argument("--limit-calls", type=int, help="only the first N calls, for a trial run")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output dir")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out_dir / args.features
    if out.exists() and not args.force:
        print(f"error: {out} exists; use --force to overwrite", file=sys.stderr)
        return 1
    check_ffmpeg()

    start = time.perf_counter()
    clips = list_clips(args.dataset, args.limit_calls)
    print(f"{clips['event_id'].nunique()} calls, {len(clips)} clips, {args.workers} workers")
    meta = {"features": args.features}

    if args.features == "praat":
        sentences, calls, failed = build_praat(clips, args.workers, start)
        meta |= {"parselmouth": parselmouth.__version__, "praat": parselmouth.PRAAT_VERSION}
    else:
        device = pick_device(args.device)
        print(f"model {args.features} on {device}")
        sentences, sentence_emb, calls, call_emb, failed, layer = build_ssl(
            clips, args.features, device, args.workers, start
        )
        checkpoint, revision, _ = ssl_embed.SSL_MODELS[args.features]
        meta |= {
            "checkpoint": checkpoint, "revision": revision, "device": str(device),
            "sentence_layer": layer, "pooling": "mean over frames of each hidden state",
            "call_emb": "mean over the call's sentences, all hidden states (index 0 = CNN)",
            "dtype": "float16", "torch": torch.__version__,
            "transformers": transformers.__version__,
        }  # fmt: skip

    out.mkdir(parents=True, exist_ok=True)
    sentences.to_csv(out / "sentences.csv", index=False)
    calls.to_csv(out / "calls.csv", index=False)
    if args.features != "praat":
        np.save(out / "sentence_emb.npy", sentence_emb)
        np.save(out / "call_emb.npy", call_emb)
    meta |= {
        "n_calls": len(calls), "n_clips": len(clips), "n_failed": len(failed),
        "n_bad": int(calls["n_bad"].sum()), "n_bad_calls": int((~calls["audio_call_ok"]).sum()),
        "max_words_per_sec": MAX_WORDS_PER_SEC, "max_bad_share": MAX_BAD_SHARE,
        "workers": args.workers, "runtime_s": round(time.perf_counter() - start, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nwritten to {out}")
    for key in ("n_calls", "n_clips", "n_failed", "n_bad", "n_bad_calls", "runtime_s"):
        print(f"{key}: {meta[key]}")
    for path in failed:
        print(f"warning: no features for {path} (decode or Praat failed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

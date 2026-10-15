"""HuBERT (or wav2vec 2.0) means per call and section, for all 13 hidden states.

Round 3 needs each section's mean over every hidden state; audio_embeddings kept all
layers only at call level. This reruns the same extraction (same decoding, model,
revision, pooling and audio_ok rule, imported from ../audio_embeddings) and sums each
audio_ok clip into its call's section instead.

Writes data/features/<model>/:
- section_emb.npy   (n_calls, 4, n_layers, hidden) float16, sections in sections.SECTIONS
                    order; NaN where a section has no audio_ok clip
- section_counts.csv  audio_ok clips per call and section
- sentences.csv     event_id, sent_idx, section, duration_s, audio_ok
- meta.json         settings, runtime, and the check against the committed call means

The check: each call's clip-weighted mean of its section means must reproduce
audio_embeddings/data/<model>/call_emb.npy, and every clip's audio_ok must agree.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "audio_embeddings"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import ssl_embed  # noqa: E402
from build_audio import _decode_task, default_workers, pick_device, progress, run_tasks  # noqa: E402
from clips import list_clips, plausible_rate  # noqa: E402
from decode import SAMPLE_RATE, check_ffmpeg  # noqa: E402
from dataset import DEFAULT_DATASET  # noqa: E402
from sections import SECTIONS  # noqa: E402

RTOL, ATOL = 2e-3, 1e-2  # float16 storage (relative error < 5e-4) plus MPS differences


class SectionSums:
    """Running per-call, per-section sums of clip-level layer means."""

    def __init__(self, event_ids: list[str], n_layers: int, hidden: int):
        self.pos = {e: i for i, e in enumerate(event_ids)}
        self.sums = np.zeros((len(event_ids), len(SECTIONS), n_layers, hidden), dtype=np.float64)
        self.counts = np.zeros((len(event_ids), len(SECTIONS)), dtype=int)

    def add(self, event_id: str, section: str, means: np.ndarray) -> None:
        i, s = self.pos[event_id], SECTIONS.index(section)
        self.sums[i, s] += means
        self.counts[i, s] += 1

    def section_means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):  # empty section → NaN
            return self.sums / self.counts[:, :, None, None]

    def call_means(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.sums.sum(axis=1) / self.counts.sum(axis=1)[:, None, None]


def compare_calls(call_means: np.ndarray, committed: np.ndarray) -> dict:
    """Recomputed vs committed call means, over calls that have audio."""
    have = ~np.isnan(call_means).any(axis=(1, 2))
    a, b = call_means[have], committed[have].astype(np.float64)
    return {
        "calls_compared": int(have.sum()),
        "max_abs_diff": float(np.abs(a - b).max()) if have.any() else None,
        "max_abs_value": float(np.abs(b).max()) if have.any() else None,
        "ok": bool(have.any() and np.allclose(a, b, rtol=RTOL, atol=ATOL)),
    }


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--model", choices=sorted(ssl_embed.SSL_MODELS), default="hubert")
    p.add_argument("--root", type=Path, default=Path(".."), help="repository root")
    p.add_argument("--sections", type=Path, default=Path("data/sentences.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/features"))
    p.add_argument("--device", default="auto", help="auto, mps or cpu")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--limit-calls", type=int, default=None, help="first N calls only, for a timing trial")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = args.out_dir / args.model
    if out.exists() and not args.force:
        print(f"error: {out} exists; use --force to overwrite", file=sys.stderr)
        return 1
    check_ffmpeg()
    start = time.perf_counter()

    clips = list_clips(args.dataset.expanduser(), args.limit_calls)
    sections = pd.read_csv(args.sections)
    clips = clips.merge(sections[["event_id", "sent_idx", "section"]], on=["event_id", "sent_idx"],
                        how="left", validate="one_to_one")  # fmt: skip
    if clips["section"].isna().any():
        print("error: clips without a section label; rerun build_sections.py", file=sys.stderr)
        return 1
    event_ids = list(dict.fromkeys(clips["event_id"]))
    device = pick_device(args.device)
    model, do_normalize = ssl_embed.load_model(args.model, device)
    n_layers, hidden = model.config.num_hidden_layers + 1, model.config.hidden_size
    print(f"{len(event_ids)} calls, {len(clips)} clips, {args.workers} workers; {args.model} on {device}")

    sums, rows = SectionSums(event_ids, n_layers, hidden), []
    audio_iter = run_tasks(_decode_task, clips["path"].tolist(), args.workers)
    for i, (clip, audio) in enumerate(zip(clips.itertuples(index=False), audio_iter), start=1):
        means = None if audio is None else ssl_embed.layer_means(audio, model, device, do_normalize)
        duration = np.nan if audio is None else len(audio) / SAMPLE_RATE
        ok = means is not None and plausible_rate(clip.n_words, duration)  # the audio_embeddings rule
        if ok:
            sums.add(clip.event_id, clip.section, means)
        rows.append({"event_id": clip.event_id, "sent_idx": clip.sent_idx, "section": clip.section,
                     "duration_s": duration, "audio_ok": ok})  # fmt: skip
        progress(i, len(clips), start)

    sentences = pd.DataFrame(rows)
    stage = args.root / "audio_embeddings/data" / args.model
    saved_ok = pd.read_csv(stage / "sentences.csv", usecols=["event_id", "sent_idx", "audio_ok"])
    flags = sentences.merge(saved_ok, on=["event_id", "sent_idx"], suffixes=("", "_saved"))
    committed_calls = pd.read_csv(stage / "calls.csv", usecols=["event_id"])["event_id"].tolist()
    committed = np.load(stage / "call_emb.npy")[[committed_calls.index(e) for e in event_ids]]
    check = compare_calls(sums.call_means(), committed)
    check["audio_ok_mismatches"] = int((flags["audio_ok"] != flags["audio_ok_saved"]).sum())
    check["ok"] = check["ok"] and check["audio_ok_mismatches"] == 0 and len(flags) == len(sentences)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "section_emb.npy", sums.section_means().astype(np.float16))
    counts = pd.DataFrame(sums.counts, columns=list(SECTIONS)).assign(event_id=event_ids)
    counts[["event_id", *SECTIONS]].to_csv(out / "section_counts.csv", index=False)
    sentences.to_csv(out / "sentences.csv", index=False)
    checkpoint, revision, _ = ssl_embed.SSL_MODELS[args.model]
    meta = {
        "model": args.model, "checkpoint": checkpoint, "revision": revision, "device": str(device),
        "pooling": "mean over frames of each hidden state, then mean over a section's audio_ok clips",
        "sections": list(SECTIONS), "dtype": "float16", "n_calls": len(event_ids),
        "n_clips": len(clips), "limit_calls": args.limit_calls, "check": check,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "workers": args.workers, "runtime_s": round(time.perf_counter() - start, 1),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }  # fmt: skip
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwritten to {out} ({meta['runtime_s']} s)")
    print(json.dumps(check, indent=2))
    if not check["ok"]:
        print("CHECK FAILED: the section means do not reproduce the committed call means", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

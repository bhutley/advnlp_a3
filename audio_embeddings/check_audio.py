"""Acceptance checks for the audio features; writes <data-dir>/acceptance.md.

Must pass (exit 1 otherwise):
  1. Counts: 572 calls, 89,722 clips; the only failed clips are the dataset's empty
     (0-byte) files; finite arrays apart from those rows.
  2. Decoding: decoded vs ffprobe durations differ by an offset that is constant within
     each call (it depends on the call's sample rate) and bounded.
  3. MPS vs CPU: SSL layer means agree within tolerance.
Reported:
  4. Codec noise floor: Praat voice-quality features of a synthetic voiced signal with
     known jitter, before and after the dataset's MP3 encoding.
  5. Clipping in the source clips, and clip/call quality (implausible speaking rates).
  6. Plausibility of call-level pitch, speech rate and voicing.
  7. Cross-model similarity (RSA) of wav2vec2 and HuBERT call vectors.
  8. First look at the targets: Praat call features vs event-study targets. Reported
     only; never a basis for choosing features or models.
  9. Recording format (data/formats.csv): format counts, Praat features and targets by
     format, and how well format can be predicted from the SSL call vectors.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

import ssl_embed
from build_audio import _decode_task, run_tasks
from clips import MAX_BAD_SHARE, list_clips
from decode import SAMPLE_RATE, decode_clip
from praat import praat_features

N_CALLS, N_CLIPS = 572, 89722
SSL_FEATURES = ["hubert", "wav2vec2"]
OFFSET_MAX_RANGE, OFFSET_MAX = 0.002, 0.2  # seconds: within-call range, absolute bound
OFFSET_CALLS, OFFSET_CLIPS_PER_CALL = 30, 3
MAIN_FORMATS = 3  # the largest format groups, compared in section 9
DEVICE_TOLERANCE = 1e-2
F0_RANGE = (85.0, 255.0)  # typical adult speaking F0, Hz
CLIP_LEVEL = 0.999  # |sample| at or above this counts as clipped
PRAAT_TARGET_FEATURES = ["f0_mean_hz", "f0_sd_st", "intensity_sd_db", "speech_rate",
                         "voiced_frac", "jitter_local", "shimmer_local", "hnr_db"]  # fmt: skip
TARGETS = ["vol_3", "vol_30", "car_mm_0_1", "car_mm_0_5"]


# --- pure helpers ---


def count_check(meta: dict, n_calls: int, n_clips: int, n_empty: int,
                sentence_emb: np.ndarray | None, call_emb: np.ndarray | None) -> bool:  # fmt: skip
    """Counts match, the failed clips are exactly the dataset's empty files, and arrays are
    finite except for those clips' sentence rows. Praat outputs pass None for the arrays."""
    ok = meta["n_calls"] == n_calls and meta["n_clips"] == n_clips and meta["n_failed"] == n_empty
    if sentence_emb is not None:
        ok &= int((~np.isfinite(sentence_emb)).any(axis=1).sum()) == n_empty
    if call_emb is not None:
        ok &= bool(np.isfinite(call_emb).all())
    return bool(ok)


def offset_check(diffs_by_call: dict[str, np.ndarray]) -> tuple[float, float, bool]:
    """(worst within-call range, largest |offset|, ok).

    The offset is the MP3 encoder delay/padding that ffprobe counts and the decoder trims;
    it depends on the call's sample rate, so it must be constant within each call.
    """
    worst = max(float(d.max() - d.min()) for d in diffs_by_call.values())
    largest = max(float(np.abs(d).max()) for d in diffs_by_call.values())
    return worst, largest, worst < OFFSET_MAX_RANGE and largest < OFFSET_MAX


def kruskal_by_group(df: pd.DataFrame, col: str, group: str) -> tuple[dict, float]:
    """(median of col per group, Kruskal–Wallis p-value across groups), ignoring NaN."""
    groups = {g: d[col].dropna() for g, d in df.groupby(group)}
    p = float(stats.kruskal(*groups.values()).pvalue)
    return {g: float(v.median()) for g, v in groups.items()}, p


def format_predictability(x: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(leave-one-out nearest-centroid accuracy, majority-class rate) on standardised x.

    How much the recording format is encoded in a representation.
    """
    x = (x - x.mean(0)) / x.std(0)
    classes = np.unique(labels)
    sums = {c: x[labels == c].sum(0) for c in classes}
    counts = {c: int((labels == c).sum()) for c in classes}
    correct = 0
    for xi, yi in zip(x, labels):
        cents = {c: (sums[c] - (xi if c == yi else 0)) / (counts[c] - (c == yi)) for c in classes}
        correct += min(cents, key=lambda c: np.linalg.norm(xi - cents[c])) == yi
    majority = max(counts.values()) / len(labels)
    return correct / len(labels), majority


def pulse_train(f0: float = 120.0, jitter: float = 0.01, seconds: float = 2.0,
                sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:  # fmt: skip
    """A voiced-like signal: glottal pulses with periods perturbed uniformly by ±jitter,
    each exciting two damped resonances (700 and 1,200 Hz).

    Pulses sit at exact continuous times, not rounded to samples, so the only jitter is
    the one requested: rounding alone would add about 0.75% at 120 Hz and 16 kHz.
    """
    rng = np.random.default_rng(seed)
    times, t = [], 0.05
    while t < seconds - 0.05:
        times.append(t)
        t += (1 / f0) * (1 + jitter * rng.uniform(-1, 1))
    grid = np.arange(int(seconds * sr)) / sr
    out = np.zeros_like(grid)
    for tk in times:
        lo, hi = int(tk * sr), min(len(grid), int((tk + 0.02) * sr) + 1)
        tau = grid[lo:hi] - tk
        tau = np.where(tau >= 0, tau, 0.0)
        out[lo:hi] += np.exp(-tau / 0.003) * (
            np.sin(2 * np.pi * 700 * tau) + 0.5 * np.sin(2 * np.pi * 1200 * tau)
        )
    return (0.3 * out / np.abs(out).max()).astype(np.float32)


def rsa(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation of the upper triangles of two row-wise cosine-similarity
    matrices: do two representations agree on which items are similar?"""

    def cosine(x):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        return x @ x.T

    iu = np.triu_indices(len(a), k=1)
    return float(stats.spearmanr(cosine(a)[iu], cosine(b)[iu]).statistic)


def plausibility(calls: pd.DataFrame) -> tuple[dict, list[str]]:
    lo, hi = F0_RANGE
    f0 = calls["f0_mean_hz"]
    flagged = calls.loc[~f0.between(lo, hi), "event_id"].tolist()
    summary = {
        "f0_min": f0.min(), "f0_median": f0.median(), "f0_max": f0.max(),
        "n_flagged": len(flagged),
        "speech_rate_median": calls["speech_rate"].median(),
        "voiced_frac_median": calls["voiced_frac"].median(),
    }  # fmt: skip
    return summary, flagged


def target_table(calls, events, features, targets) -> pd.DataFrame:
    joined = calls.merge(events[events["status"] == "ok"], on="event_id")
    rows = []
    for feature in features:
        for target in targets:
            pair = joined[[feature, target]].dropna()
            r = stats.spearmanr(pair[feature], pair[target])
            rows.append({"feature": feature, "target": target, "n": len(pair),
                         "spearman": float(r.statistic), "p": float(r.pvalue)})  # fmt: skip
    return pd.DataFrame(rows)


def markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# --- I/O helpers ---


def encode_mp3(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Encode in the dataset's lowest-quality common format: 11,025 Hz mono MP3 at 16 kbps
    (92 calls). The codec probe therefore measures the worst common case."""
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1",
                    "-i", "-", "-ar", "11025", "-b:a", "16k", str(path)],
                   input=pcm, check=True)  # fmt: skip


def ffprobe_duration(path: str) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)  # fmt: skip
    return float(out.stdout)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--dataset", type=Path, default=Path.home() / "var/data/earnings_calls/ACL19_Release"
    )
    parser.add_argument("--events", type=Path, default=Path("../event_study/data/events.csv"))
    parser.add_argument("--device-clips", type=int, default=50)
    parser.add_argument("--clipping-clips", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    clips = list_clips(args.dataset)
    sample = clips.sample(len(clips), random_state=args.seed)["path"].tolist()  # shuffled
    sections, failed = ["# Audio acceptance checks"], False

    # 1. Counts
    empty = [p for p in clips["path"] if Path(p).stat().st_size == 0]
    rows = []
    for features in ["praat"] + SSL_FEATURES:
        d = args.data_dir / features
        meta = pd.read_json(d / "meta.json", typ="series").to_dict()
        sent, call = (None, None) if features == "praat" else (
            np.load(d / "sentence_emb.npy"), np.load(d / "call_emb.npy"))  # fmt: skip
        ok = count_check(meta, N_CALLS, N_CLIPS, len(empty), sent, call)
        failed |= not ok
        rows.append({"features": features, "n_calls": meta["n_calls"], "n_clips": meta["n_clips"],
                     "n_failed": meta["n_failed"], "n_bad": meta["n_bad"],
                     "n_bad_calls": meta["n_bad_calls"], "runtime_s": meta["runtime_s"],
                     "pass": "yes" if ok else "NO"})  # fmt: skip
    sections.append(
        "## 1. Counts (must pass)\n\nFailed clips must be exactly the dataset's empty files "
        f"({len(empty)}: {', '.join(Path(p).parent.parent.name + '/' + Path(p).name for p in empty) or 'none'}). "
        "`n_bad` counts clips that are not `audio_ok` (failed, or implausibly fast for their "
        "transcript line); `n_bad_calls` counts calls with `audio_call_ok` false.\n\n"
        + markdown(pd.DataFrame(rows), "{:.1f}")
    )

    # 2. Decoded vs ffprobe durations, per call
    diffs_by_call = {}
    for event_id in clips["event_id"].drop_duplicates().sample(OFFSET_CALLS, random_state=args.seed):
        paths = clips.loc[clips["event_id"] == event_id, "path"].sample(
            OFFSET_CLIPS_PER_CALL, random_state=args.seed)  # fmt: skip
        diffs_by_call[event_id] = np.array(
            [len(decode_clip(Path(p))) / SAMPLE_RATE - ffprobe_duration(p) for p in paths]
        )
    worst, largest, ok = offset_check(diffs_by_call)
    failed |= not ok
    offsets = sorted({round(float(np.median(d)), 3) for d in diffs_by_call.values()})
    sections.append(
        "## 2. Decoded vs ffprobe duration (must pass)\n\n"
        f"{OFFSET_CALLS} random calls × {OFFSET_CLIPS_PER_CALL} clips: worst within-call range "
        f"{worst:.4f} s (limit {OFFSET_MAX_RANGE}), largest |offset| {largest:.3f} s (limit "
        f"{OFFSET_MAX}); per-call offsets seen: {offsets}. Pass: {'yes' if ok else 'NO'}. "
        "ffprobe's container duration includes the MP3 encoder delay and padding, which the "
        "decoder trims; its length in seconds depends on the call's sample rate."
    )

    # 3. MPS vs CPU
    rows = []
    cpu, mps = torch.device("cpu"), torch.device("mps")
    audio = [decode_clip(Path(p)) for p in sample[: args.device_clips]]
    for key in SSL_FEATURES:
        model, do_normalize = ssl_embed.load_model(key, cpu)
        on_cpu = [ssl_embed.layer_means(a, model, cpu, do_normalize) for a in audio]
        model = model.to(mps)
        on_mps = [ssl_embed.layer_means(a, model, mps, do_normalize) for a in audio]
        diff = max(float(np.abs(c - m).max()) for c, m in zip(on_cpu, on_mps))
        ok = diff < DEVICE_TOLERANCE
        failed |= not ok
        rows.append({"model": key, "n": len(audio), "max_abs_diff": diff,
                     "pass": "yes" if ok else "NO"})  # fmt: skip
    sections.append(f"## 3. MPS vs CPU, all 13 layers (must pass, tolerance {DEVICE_TOLERANCE})\n\n"
                    + markdown(pd.DataFrame(rows), "{:.2e}"))  # fmt: skip

    # 4. Codec noise floor
    rows = []
    tmp = args.data_dir / "_codec_probe.mp3"
    for label, jitter in [("steady pulses", 0.0), ("±1% period jitter", 0.01)]:
        raw = pulse_train(jitter=jitter, seed=args.seed)
        encode_mp3(tmp, raw)
        for stage, x in [("raw 16 kHz", raw), ("after MP3", decode_clip(tmp))]:
            f = praat_features(x, SAMPLE_RATE, n_words=1)
            rows.append({"signal": label, "stage": stage, "f0_mean_hz": f["f0_mean_hz"],
                         "jitter_local": f["jitter_local"], "shimmer_local": f["shimmer_local"],
                         "hnr_db": f["hnr_db"]})  # fmt: skip
    tmp.unlink(missing_ok=True)
    sections.append(
        "## 4. Codec noise floor (reported)\n\nA synthetic 120 Hz voiced signal, measured "
        "before and after the dataset's lowest-quality common encoding (11,025 Hz, 16 kbps "
        "MP3, used by 92 calls; most calls are 44.1 kHz, 64 kbps), so this is a worst case. "
        "Expected raw "
        "jitter for ±1% uniform period perturbation is about 0.0067.\n\n"
        + markdown(pd.DataFrame(rows), "{:.4f}")
    )

    # 5. Clipping
    paths = sample[: args.clipping_clips]
    clipped_clips = clipped_samples = total_samples = 0
    for a in run_tasks(_decode_task, paths, args.workers):
        hits = int((np.abs(a) >= CLIP_LEVEL).sum())
        clipped_clips += hits > 0
        clipped_samples += hits
        total_samples += len(a)
    sections.append(
        f"## 5. Clipping (reported)\n\n{len(paths)} random clips: "
        f"{clipped_clips / len(paths):.1%} contain at least one sample at |x| ≥ {CLIP_LEVEL}; "
        f"{clipped_samples / total_samples:.3%} of all samples are clipped."
    )

    # 5b. Clip and call quality
    praat_sentences = pd.read_csv(args.data_dir / "praat" / "sentences.csv")
    praat_calls = pd.read_csv(args.data_dir / "praat" / "calls.csv")
    bad_calls = praat_calls.loc[~praat_calls["audio_call_ok"], ["event_id", "n_sentences", "n_bad"]]
    sections.append(
        f"## 5b. Clip and call quality (reported)\n\n{(~praat_sentences['audio_ok']).sum()} of "
        f"{len(praat_sentences)} clips are not `audio_ok` (excluded from call aggregates). Calls "
        f"with `audio_call_ok` false (over {MAX_BAD_SHARE:.0%} of clips bad), to be excluded "
        "from audio models:\n\n" + (markdown(bad_calls, "{:.0f}") if len(bad_calls) else "none")
    )

    # 6. Plausibility
    summary, flagged = plausibility(praat_calls)
    sections.append(
        "## 6. Plausibility (reported)\n\n"
        f"Call-level F0: min {summary['f0_min']:.0f}, median {summary['f0_median']:.0f}, "
        f"max {summary['f0_max']:.0f} Hz; {summary['n_flagged']} calls outside "
        f"{F0_RANGE[0]:.0f}–{F0_RANGE[1]:.0f} Hz{': ' + ', '.join(flagged) if flagged else ''}. "
        f"Median speech rate {summary['speech_rate_median']:.2f} words/s (pauses included); "
        f"median voiced fraction {summary['voiced_frac_median']:.2f}."
    )

    # 7. Cross-model RSA
    layer = ssl_embed.SENTENCE_LAYER
    a, b = (np.load(args.data_dir / k / "call_emb.npy")[:, layer].astype(np.float32)
            for k in SSL_FEATURES)  # fmt: skip
    sections.append(
        f"## 7. Cross-model similarity (reported)\n\nRSA between HuBERT and wav2vec2 call "
        f"vectors at layer {layer}: Spearman {rsa(a, b):.3f} over {len(a)} calls."
    )

    # 8. First look at the targets
    table = target_table(praat_calls, pd.read_csv(args.events), PRAAT_TARGET_FEATURES, TARGETS)
    sections.append(
        "## 8. First look at the targets (reported only)\n\nSpearman correlation of Praat call "
        "features with event-study targets over `ok` events. A sanity check, not a basis for "
        "choosing features or models.\n\n" + markdown(table)
    )

    # 9. Recording format
    formats = pd.read_csv(args.data_dir / "formats.csv")
    formats["format"] = (formats["sample_rate"].astype(str) + " Hz / " + formats["kbps"].astype(str)
                         + " kbps" + np.where(formats["channels"] > 1, " stereo", ""))  # fmt: skip
    counts = formats.groupby("format").size().sort_values(ascending=False).rename("calls")
    main_formats = counts.index[:MAIN_FORMATS]
    rows = []
    with_fmt = praat_calls.merge(formats[["event_id", "format"]], on="event_id")
    with_fmt = with_fmt[with_fmt["format"].isin(main_formats) & with_fmt["audio_call_ok"]]
    for col in PRAAT_TARGET_FEATURES:
        medians, p = kruskal_by_group(with_fmt, col, "format")
        rows.append({"feature": col} | {f: medians[f] for f in main_formats} | {"p": p})
    events = pd.read_csv(args.events)
    ev = events[events["status"] == "ok"].merge(formats[["event_id", "format"]], on="event_id")
    ev = ev[ev["format"].isin(main_formats)]
    for col in TARGETS:
        medians, p = kruskal_by_group(ev, col, "format")
        rows.append({"feature": f"target: {col}"} | {f: medians[f] for f in main_formats} | {"p": p})
    pred_rows = []
    for key in SSL_FEATURES:
        calls_k = pd.read_csv(args.data_dir / key / "calls.csv").merge(
            formats[["event_id", "format"]], on="event_id", how="left")  # fmt: skip
        keep = (calls_k["format"].isin(main_formats) & calls_k["audio_call_ok"]).to_numpy()
        x = np.load(args.data_dir / key / "call_emb.npy")[keep, layer].astype(np.float32)
        acc, majority = format_predictability(x, calls_k["format"].to_numpy()[keep])
        pred_rows.append({"model": key, "layer": layer, "n": int(keep.sum()),
                          "loo_accuracy": acc, "majority_rate": majority})  # fmt: skip
    mixed = int((formats["n_other_format"] > 0).sum())
    sections.append(
        "## 9. Recording format (reported)\n\nClips come in several formats; format is constant "
        f"within a call ({mixed} calls contain a clip in another format).\n\n"
        + markdown(counts.reset_index(), "{:.0f}")
        + f"\n\nPraat call features and targets by format, for the {MAIN_FORMATS} largest "
        "groups (Kruskal–Wallis p across them; `audio_call_ok` calls only):\n\n"
        + markdown(pd.DataFrame(rows), "{:.3f}")
        + "\n\nHow well the format can be predicted from SSL call vectors (leave-one-out "
        "nearest centroid on standardised vectors, same groups):\n\n"
        + markdown(pd.DataFrame(pred_rows), "{:.2f}")
        + "\n\nFormat therefore affects voice-quality features and SSL embeddings. Include it "
        "as a covariate (or fixed effect) in audio models and report results by format."
    )

    report = "\n\n".join(sections) + "\n"
    print(report)
    (args.data_dir / "acceptance.md").write_text(report)
    print("FAILED" if failed else "all must-pass checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

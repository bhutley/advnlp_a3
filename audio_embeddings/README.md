# audio_embeddings — wav2vec2, HuBERT and Praat features for the ACL19 audio clips

Extracts three feature sets from the dataset's 89,722 sentence clips (one clip per
transcript line). Each is also aggregated to one row per call:

- **`wav2vec2`**: `facebook/wav2vec2-base`, pretrained only
- **`hubert`**: `facebook/hubert-base-ls960`, pretrained only
- **`praat`**: interpretable acoustic features (pitch, intensity, voice quality, speech
  rate) via `praat-parselmouth`

The outputs join to `../event_study/data/events.csv` on `event_id`, and sentence by
sentence to `../text_embeddings` on `event_id` + `sent_idx`.

The design, including the data-quality rules, is in
`docs/plans/2026-10-06-audio-embeddings-design.md`.

## Current state

| | `praat` | `wav2vec2` | `hubert` |
|---|---|---|---|
| Calls / clips | 572 / 89,722 | 572 / 89,722 | 572 / 89,722 |
| Failed clips | 1 (an empty file in the dataset) | 1 | 1 |
| Clips not `audio_ok` | 78 | 78 | 78 |
| Calls not `audio_call_ok` | 2 | 2 | 2 |
| Runtime (M4 Pro) | 25 min | 53 min | 53 min |

There are 62 tests, all passing. All must-pass acceptance checks pass (see
`data/acceptance.md`).

## Requirements

- Python 3.13 or newer, and [uv](https://docs.astral.sh/uv/). `uv sync` installs torch,
  transformers, praat-parselmouth, numpy, pandas, scipy and pytest.
- **ffmpeg** on PATH, for decoding.
- The first SSL run downloads each model (about 380 MB) into the Hugging Face cache.

## Usage

Run from this directory:

```sh
uv sync

# Each run decodes every clip again in a process pool (default: one worker per
# performance core). Times on an M4 Pro: praat ~25 min, wav2vec2 and hubert ~53 min each.
uv run python build_audio.py --features praat
uv run python build_audio.py --features wav2vec2
uv run python build_audio.py --features hubert

# each call's recording format → data/formats.csv (~7 min; probes every clip)
uv run python build_formats.py

# acceptance report → data/acceptance.md (exits 1 if a must-pass check fails)
uv run python check_audio.py

uv run pytest
```

`build_audio.py` options:

- `--dataset`: default `~/var/data/earnings_calls/ACL19_Release`
- `--out-dir`: default `data`
- `--device`: `auto`, `mps` or `cpu`, for the SSL models
- `--workers N`: decode processes; `0` runs in-process
- `--limit-calls N`: only the first N calls, for a trial run
- `--force`: overwrite an existing `data/<features>/`; without it the script refuses

### `sentence_emb.npy` is not in git

At about 138 MB each, the SSL sentence arrays exceed GitHub's 100 MB file limit, so
`.gitignore` excludes them. Regenerate them with the two SSL commands above plus
`--force`. Revisions are pinned, so results are reproducible.

## Method

| Step | Choice |
|---|---|
| Clips | `CEO/<Speaker>_<para>_<sent>.mp3`, sorted by numeric (para, sent). The *k*-th clip is `sent_idx = k`, the same index as `text_embeddings`. A clip/line count mismatch aborts the run |
| Decoding | `ffmpeg` → float32 mono at 16 kHz. Source clips are MP3 in several formats (see Recording format) |
| SSL models | One clip per forward pass: both models normalise over the whole input and take no attention mask, so padding a batch would change outputs. Audio is normalised as each model's feature extractor specifies |
| SSL pooling | Each of the 13 hidden states (CNN output + 12 layers) averaged over 20 ms frames |
| SSL sentence vector | **Layer 6**, fixed before any targets were seen |
| SSL call vector | **All 13 layers**: the mean of the call's `audio_ok` sentence vectors. The layer is chosen at the modelling stage |
| Praat | Pitch 75–500 Hz with a 10 ms step. F0 SD in semitones. Intensity excludes Praat's −300 dB zero-energy frames. Jitter and shimmer (local) from a periodic point process; HNR from cross-correlation harmonicity |
| Praat call values | Means over `audio_ok` sentences, ignoring NaN |

### Data quality

The dataset's audio has defects, which are flagged rather than silently kept:

- **`audio_ok`** (per sentence) is false if the clip failed to decode or measure, or
  runs faster than **8 words/s** for its transcript line, i.e. it is truncated. Real
  speech is under 5 words/s at the 99th percentile. Such clips keep their features but
  are left out of call aggregates.
- **`audio_call_ok`** (per call) is false if more than **10%** of the call's clips fail
  `audio_ok`. This flags two calls whose segmentation is broken: **BorgWarner
  2017-07-27** (42 of 91 clips) and **eBay 2017-04-19** (20 of 25). **Exclude these from
  audio models.**
- One clip is an **empty file** in the dataset: Ventas 2017-10-27, `Robert Probst_1_8.mp3`.
  It is the only failed clip, and its sentence rows are NaN.

### Recording format

The clips are not in one format. Format is constant within a call (2 calls each contain
one odd clip), and `data/formats.csv` records it per call:

| Format | Calls |
|---|---|
| 44.1 kHz, 64 kbps | 336 |
| 22.05 kHz, 32 kbps | 104 |
| 11.025 kHz, 16 kbps | 92 |
| 48 kHz, 64 kbps | 16 |
| 32 kHz, 48 kbps | 11 |
| 24 kHz, 32 kbps | 8 |
| 44.1 kHz, 128 kbps, stereo | 4 |
| 8 kHz, 8 kbps | 1 |

**Format is a confound. Control for it in every audio model.**

- The SSL call vectors encode format: it can be predicted from them with 84–86%
  accuracy, against a 63% majority rate.
- Shimmer, HNR and jitter differ by format. Pitch variation, speech rate and intensity
  variation do not.
- **`vol_30` also differs by format** (p = 0.026), and `vol_3` marginally (p = 0.09).
  Lower-quality formats have slightly lower volatility.

So a model could learn "recording format" as a proxy for volatility. Include format as a
covariate or fixed effect (e.g. one-hot `sample_rate`/`kbps`), fit any feature
standardisation within training folds, and report results by format as a robustness
check.

## Files

| Path | Description |
|---|---|
| `clips.py` | Lists clips, aligns them to transcript lines, and holds the quality rules |
| `decode.py` | ffmpeg decoding to 16 kHz float32 |
| `praat.py` | Praat features for one clip |
| `ssl_embed.py` | Model registry (pinned), loading, per-layer mean pooling |
| `build_audio.py` | CLI: one feature set → `data/<features>/` |
| `build_formats.py` | CLI: probes every clip → `data/formats.csv` |
| `check_audio.py` | Acceptance checks → `data/acceptance.md` |
| `tests/` | pytest suite: synthetic audio encoded in the dataset's MP3 format and a tiny wav2vec2. No network access needed |

### `data/formats.csv`

`event_id, sample_rate, kbps, channels, n_clips, n_other_format, n_unreadable`, one row
per call. The format is the call's most common one.

### `data/wav2vec2/` and `data/hubert/`

Row *i* of each `.npy` matches row *i* of its CSV.

| File | Content |
|---|---|
| `sentences.csv` | `event_id, sent_idx, duration_s, n_frames, audio_ok` |
| `sentence_emb.npy` | (89,722, 768) float16, layer 6. **Not in git** |
| `calls.csv` | `event_id, n_sentences, n_bad, audio_call_ok, duration_s` |
| `call_emb.npy` | (572, 13, 768) float16, every hidden state; index 0 = CNN output |
| `meta.json` | Checkpoint, revision, sentence layer, device, versions, counts, quality thresholds, runtime |

### `data/praat/`

| File | Content |
|---|---|
| `sentences.csv` | `event_id, sent_idx, duration_s, n_words, speech_rate, voiced_frac, f0_mean_hz, f0_sd_st, intensity_mean_db, intensity_sd_db, jitter_local, shimmer_local, hnr_db, audio_ok` |
| `calls.csv` | `event_id, n_sentences, n_bad, audio_call_ok`, plus the call mean of each feature |
| `meta.json` | parselmouth and Praat versions, counts, quality thresholds, runtime |

## Loading in Colab

Upload `audio_embeddings/data/` and `event_study/data/events.csv`, then:

```python
import numpy as np
import pandas as pd

model, layer = "hubert", 6                            # choose the layer by CV on training folds
calls = pd.read_csv(f"data/{model}/calls.csv")
X_all = np.load(f"data/{model}/call_emb.npy")         # (572, 13, 768)
events = pd.read_csv("events.csv")

formats = pd.read_csv("data/formats.csv")

df = calls.reset_index().merge(events[events["status"] == "ok"], on="event_id")
df = df[df["audio_call_ok"]].merge(formats, on="event_id")   # 488 calls
X = X_all[df["index"], layer].astype(np.float32)
F = pd.get_dummies(df["sample_rate"].astype(str) + "/" + df["kbps"].astype(str))  # format covariates
y = df["vol_3"]

# Praat features at call level
praat = pd.read_csv("data/praat/calls.csv")

# Sentence level, aligned with text_embeddings on (event_id, sent_idx)
S = np.load(f"data/{model}/sentence_emb.npy", mmap_mode="r")
sentences = pd.read_csv(f"data/{model}/sentences.csv")   # filter on audio_ok
```

## Results

From `data/acceptance.md`:

**Must-pass checks**

| Check | Result |
|---|---|
| Counts | All three feature sets: 572 calls, 89,722 clips. The only failed clip is the dataset's empty file. Arrays are finite apart from its sentence rows |
| Decoding | Within each of 30 sampled calls, the decoded-vs-ffprobe offset varies by at most 0.1 ms. Offsets of −0.011 to −0.048 s across calls are the MP3 encoder delay, which depends on the sample rate |
| MPS vs CPU | All 13 layers on 50 clips: maximum difference 7e-6 (HuBERT) and 1e-5 (wav2vec2) |

**Reported**

- **Codec effect,** measured in the worst common format (11.025 kHz, 16 kbps), on a
  synthetic 120 Hz voiced signal:
  - F0 and jitter are preserved (±1% jitter measured 0.0066 raw, 0.0060 after MP3)
  - shimmer rises by about 1.4 points
  - HNR falls by 4–12 dB
- **Clipping:** 8.5% of clips contain at least one full-scale sample, but only 0.004% of
  samples are clipped.
- **Plausibility:** call-level F0 runs from 95 to 232 Hz (median 137), with none outside
  85–255 Hz. Median speech rate is 2.52 words/s, pauses included. Median voiced fraction
  is 0.53.
- **Cross-model:** the HuBERT and wav2vec2 call vectors (layer 6) agree on which calls are
  similar (RSA Spearman 0.75).
- **First look at the targets** (Praat call features, 490 `ok` events). Reported only and
  not used for any choice. No Praat feature correlates meaningfully with volatility or
  CARs: all |ρ| < 0.09, and none reaches p < 0.05. The strongest are jitter vs `vol_30`
  (ρ = 0.085, p = 0.06) and F0 vs `vol_30` (ρ = −0.077, p = 0.09). Simple prosodic
  averages carry little signal on their own. Whether the embeddings do is a question for
  the modelling stage, which must control for format.

## Caveats for the writeup

- **Recording format is a confound** (see Recording format). It shapes the embeddings
  and is weakly related to `vol_30`.
- **Voice-quality measures are on connected speech over compressed channels.** In the
  worst common format, the MP3 encoding preserves pitch and jitter, adds about 1.4%
  shimmer, and lowers HNR by 4–12 dB. Clinical reference values (e.g. jitter under 1%)
  come from sustained vowels and do not apply. Compare these features across calls, and
  within format.
- **Absolute intensity depends on the recording level.** `intensity_sd_db` (variation
  within a clip) is more meaningful than `intensity_mean_db`.
- **`f0_mean_hz` mostly reflects the speaker's sex.** `f0_sd_st` (pitch variation, in
  semitones) is the speaker-independent pitch feature.
- **One speaker per call.** The clips are a single executive's remarks, not the whole
  call.
- **Layer 6 at sentence level was a fixed prior choice.** Any layer selection on call
  vectors must use training folds only.

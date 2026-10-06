# text_embeddings — FinBERT embeddings and sentiment for the ACL19 transcripts

Embeds every transcript sentence with two FinBERT models. Each sentence gets a
768-dimensional vector and positive, negative and neutral probabilities. These are also
aggregated to one row per call. The outputs join to `../event_study/data/events.csv` on
`event_id`, the dataset folder name, e.g. `3M Company_20170425`.

The design and its decisions are in `docs/plans/2026-10-06-finbert-embeddings-design.md`.

## Current state

| | `finbert_tone` | `prosus_finbert` |
|---|---|---|
| Checkpoint | `yiyanghkust/finbert-tone` | `ProsusAI/finbert` |
| Pinned revision | `4921590…` | `4556d13…` |
| Calls / sentences | 572 / 89,722 | 572 / 89,722 |
| Truncated sentences | 0 | 0 |
| Runtime (M4 Pro, MPS) | 103 s | 155 s |

There are 43 tests, all passing. All must-pass acceptance checks pass (see
`data/acceptance.md`).

## Requirements

Python 3.13 or newer, and [uv](https://docs.astral.sh/uv/). `uv sync` installs torch,
transformers, numpy, pandas, scipy and pytest. The first run downloads each model, about
440 MB, into the Hugging Face cache.

## Usage

Run from this directory:

```sh
uv sync

# about 2–3 minutes each on Apple Silicon; --device cpu works too, but is slower
uv run python build_embeddings.py --model finbert_tone
uv run python build_embeddings.py --model prosus_finbert

# acceptance report → data/acceptance.md (exits 1 if a must-pass check fails)
uv run python check_embeddings.py

uv run pytest
```

`build_embeddings.py` options:

- `--dataset`: default `~/var/data/earnings_calls/ACL19_Release`
- `--out-dir`: default `data`
- `--device`: `auto`, `mps` or `cpu`
- `--batch-size`: default 64
- `--max-length`: default 512
- `--limit-calls N`: process only the first N calls, for a trial run
- `--force`: overwrite an existing `data/<model>/`; without it the script refuses

### `sentence_emb.npy` is not in git

At about 138 MB each, the sentence arrays exceed GitHub's 100 MB file limit, so
`.gitignore` excludes them. To regenerate both, run:

```sh
uv run python build_embeddings.py --model finbert_tone --force
uv run python build_embeddings.py --model prosus_finbert --force
```

Each model's revision is pinned, so the probabilities come out the same on any machine.
The vectors match to within float precision (MPS and CPU agree to about 5e-6).

## Method

| Step | Choice |
|---|---|
| Sentences | Non-blank lines of `TextSequence.txt`. `sent_idx` is the 0-based index among them, and is the alignment key for the audio clips |
| Batching | Sorted by length, 64 per batch, `max_length=512` (passed explicitly: `finbert-tone`'s tokenizer has no limit of its own) |
| Sentence vector | Mean of the last hidden layer over attention-masked tokens, `[CLS]` and `[SEP]` included. Stored as float16 |
| Sentiment | Softmax of the classifier logits, mapped **by label name** to `p_pos, p_neg, p_neu`, because the two models order their labels differently |
| Call vector | Mean of the call's sentence vectors. Stored as float32 |
| Call sentiment | Mean probabilities. `tone = mean(p_pos − p_neg)`. `frac_pos` and `frac_neg` are the shares of sentences whose top label is positive or negative |

**Gotchas with transformers 5:**

- `finbert-tone`'s `config.json` has no `model_type`, so the `Auto*` classes reject it.
  Both models load with `BertTokenizer` and `BertForSequenceClassification`.
- `load_model` checks that the tokenizer's vocabulary matches the model's: 30,873 tokens
  (FinVocab) for `finbert-tone` and 30,522 for ProsusAI. It also checks the label names
  before any computation.

## Files

| Path | Description |
|---|---|
| `transcripts.py` | Reads the dataset into one row per sentence |
| `embed.py` | Model registry, label mapping, device choice, pooling, batching, call aggregation |
| `build_embeddings.py` | CLI: one model → `data/<model>/` |
| `check_embeddings.py` | Acceptance checks → `data/acceptance.md` |
| `tests/` | pytest suite using a tiny random BERT built offline. No network access or checkpoint needed |

### `data/<model>/`

Row *i* of each `.npy` matches row *i* of its CSV.

| File | Content |
|---|---|
| `sentences.csv` | `event_id, sent_idx, n_tokens, p_pos, p_neg, p_neu`, 89,722 rows |
| `sentence_emb.npy` | (89,722, 768) float16. **Not in git** |
| `calls.csv` | `event_id, n_sentences, p_pos, p_neg, p_neu, tone, frac_pos, frac_neg`, 572 rows |
| `call_emb.npy` | (572, 768) float32 |
| `meta.json` | Checkpoint, revision, pooling, dtypes, device, library versions, counts, runtime |

## Loading in Colab

Upload `text_embeddings/data/` and `event_study/data/events.csv`, then:

```python
import numpy as np
import pandas as pd

model = "finbert_tone"
calls = pd.read_csv(f"data/{model}/calls.csv")
X = np.load(f"data/{model}/call_emb.npy")             # row i ↔ calls.iloc[i]
events = pd.read_csv("events.csv")

ok = events[events["status"] == "ok"]
df = calls.reset_index().merge(ok, on="event_id")     # 'index' = row in X
X_ok, y = X[df["index"]], (df["car_mm_0_5"] > 0).astype(int)

# Sentence level: np.load(..., mmap_mode="r") avoids reading 138 MB up front
S = np.load(f"data/{model}/sentence_emb.npy", mmap_mode="r")
sentences = pd.read_csv(f"data/{model}/sentences.csv")
```

## Acceptance results

From `data/acceptance.md`:

- **Direction:** 4 positive and 4 negative sentences, written by hand, all score the
  right way round in both models.
- **MPS vs CPU:** the maximum difference over 200 sentences is about 5e-6 for both
  vectors and probabilities.
- **Cross-model:** call `tone` agrees closely between the models (Pearson 0.90, Spearman
  0.90, n = 572).
- **Sentiment profile:** about 51% of sentences are neutral, 40% positive and 8–9%
  negative. Mean call tone is +0.30, so executives lean positive. Use relative tone,
  e.g. terciles, rather than its sign.

**First look at the targets.** Spearman correlation over the 490 `ok` events. These were
computed once, as a sanity check, and were **not** used to choose the model, pooling or
features. Any model comparison must be made on held-out data.

| Model | Feature | CAR [0,+1] | CAR [0,+5] | vol_3 | vol_30 |
|---|---|---|---|---|---|
| finbert_tone | tone | +0.19 | +0.22 | +0.06 | −0.02 |
| finbert_tone | frac_neg | −0.17 | −0.21 | +0.03 | +0.05 |
| prosus_finbert | tone | +0.18 | +0.22 | +0.05 | −0.04 |
| prosus_finbert | frac_neg | −0.15 | −0.19 | −0.02 | +0.03 |

Tone relates to the *direction* of abnormal returns (p < 0.001 for both CAR windows),
but not to volatility.

## Caveats for the writeup

- **CAR windows include day 0.** [0,+1] and [0,+5] include the reaction to the reported
  numbers, which the call's tone partly reflects. A window starting after the reaction,
  e.g. [+2,+20], would test whether the call predicts drift. That would be a new window
  in `event_study`.
- **Correlated events.** The p-values assume independent events. Earnings-season
  clustering and repeat firms make them optimistic.
- **CMS Energy, 2017-02-02,** has a single sentence in the dataset, in both text and
  audio. Its call-level values rest on that sentence. Use `n_sentences` to set a minimum
  if needed.
- **`finbert-tone` is very confident.** 85% of its sentences have a top probability above
  0.95, and the median is 1.00. For ProsusAI those figures are 11% and 0.88. So for
  `finbert-tone`, `tone` and `frac_*` are close to label counts, while ProsusAI's
  probabilities carry more gradation.

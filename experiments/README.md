# Experiments: does speech help predict post-call volatility?

Bhargav's branch (`dev_bhargav`) of Advanced NLP Assignment 3. This folder holds a complete
study on the Qin & Yang (ACL 2019) earnings-call dataset: building the dataset, labels,
frozen text and audio embeddings, ridge models, and the results.

**Question.** Does the audio of an earnings call (how things are said) improve forecasts of
post-call volatility beyond the transcript (what is said) and the stock's own pre-call
volatility?

**Answer, for this cohort.**
- **Text and audio together help.** Text + audio + past volatility is the best model on the
  main test set, with MSE 0.255 against 0.270 for past volatility alone. The gain is
  significant (one-sided p = 0.018).
- **Audio's own contribution is not proven.** On top of text + past volatility, audio lowers
  MSE by only 0.005, and the gain is not significant (p = 0.087).
- **None of the gains carry over to unseen companies.** On a company-disjoint split, past
  volatility alone is best.

For a one-page, non-technical version, see [project_summary.pdf](project_summary.pdf).

## Reading guide

| Document | What it covers |
|---|---|
| [README.md](README.md) | This overview: changes, experiments, observations |
| [research.md](research.md) | The design, written before the runs: related work, target definition, evaluation plan |
| [RESULTS.md](RESULTS.md) | The full results, every table, the limitations and the exact commands to reproduce |
| [project_summary.pdf](project_summary.pdf) | A one-page summary for a general reader |

## Changes to the repository

| Path | Change |
|---|---|
| `experiments/` | **New.** All code, results and write-ups for this study |
| `README.md` (top level) | The dataset is credited as Qin & Yang (ACL 2019), not 2017 (2017 is the call year). The `acl19_proc1` path is fixed to `acl19_proc`. Adds a pointer to this folder |
| `.gitignore` | Ignores `data/` (the 9.2 GB dataset and generated files) and `experiments/results/*.npz` |
| `acl19_proc/` | **Unchanged.** Its `fetch_prices.py` was reused as is, writing to `data/prices/` |

Nothing under `data/` is committed. It holds the extracted dataset, the re-downloaded prices,
the manifests, the labels and the embeddings, all of which can be regenerated with the steps
below.

## Pipeline

| Step | Script | Output (in `data/`) |
|---|---|---|
| 1. Extract the five-part ACL19 release, with no external zip tool needed | [extract_release.py](extract_release.py) | `ACL19_Release/`: 572 call folders |
| 2. Pair each transcript sentence with its audio clip | [build_manifests.py](build_manifests.py) | `calls.csv`, `segments.csv`, `call_texts.csv` |
| 3. Re-download prices from 3 months before each company's first call | `acl19_proc/fetch_prices.py` | `prices/<TICKER>.csv` |
| 4. Compute volatility labels and the past-volatility control | [volatility_baseline.py](volatility_baseline.py) `labels` | `labels.csv`, plus files listing excluded calls and extreme returns |
| 5. Compute frozen embeddings, mean-pooled to one vector per call per modality | [embed_calls.py](embed_calls.py) | `features.npz` |
| 6. Train the ridge models and evaluate them | [volatility_baseline.py](volatility_baseline.py) `train` | `experiments/results/*.json` |

**Target.** The log standard deviation of daily `Adj Close` returns over the 7 trading days
after the first close strictly after the call date, so the immediate reaction is excluded.
**Control.** The log volatility of the 20 returns before the call.

**Embeddings.** FinBERT-tone encoder hidden states for text, over every token in 510-token
chunks, and wav2vec2-base at 16 kHz for audio. Each is a 768-dimensional vector. Model
revisions are pinned in `features.npz`: FinBERT-tone `4921590d3c0c`, wav2vec2-base
`0b5b8e868dd8`. Embedding took about 2.5 hours on an RTX 4070.

**Data coverage.** The dataset has 572 calls from 280 companies, with 89,722 aligned
sentences. 490 calls from 240 companies have prices and are all labelled. The other 82 calls
come from 40 delisted companies and are excluded. Two of the 89,722 audio clips were broken
and skipped; their sentences' text was kept.

## Experiments run

| Run | Inputs | Split | Result file |
|---|---|---|---|
| 1. First price-only trial, using the 1-month price window from `acl19_proc` | Past volatility | Date split (only 64 test calls survived) | [price_only.json](results/price_only.json) |
| 2. Price-only, with the 3-month window | Past volatility | Date split | [price_only_date.json](results/price_only_date.json) |
| 3. Price-only, with the 3-month window | Past volatility | Season split | [price_only_season.json](results/price_only_season.json) |
| 4. **Full study (main)** | Every combination of text, audio, past volatility and TF-IDF | Season split | [full_season.json](results/full_season.json) |
| 5. Sensitivity check | Same as run 4 | The starter's date split | [full_date.json](results/full_date.json) |
| 6. Sensitivity check | Same as run 4 | Company-disjoint split | [full_company.json](results/full_company.json) |

### What changed from the starter design, and why

1. **Longer pre-call price window.** The 1-month window left only 17–19 returns before many
   first calls, so 109 calls were dropped. With a 3-month window, no calls are lost.
2. **Season split** (now the default). The starter cut at the 60th and 80th percentiles of
   call dates. Calls cluster in reporting seasons, so those cuts fell mid-season and purging
   overlapping label windows removed 41 of the 57 validation calls. The season split
   removes only 6:
   - **train:** January–June, 251 calls
   - **validation:** July–September, 123 calls
   - **test:** October–December, 110 calls
3. **Past volatility is no longer shrunk.** Ridge had penalised the single past-volatility
   column as hard as each of the 1,536 embedding dimensions, which made text + audio + past
   *worse* than past alone. Past volatility is now partialled out (Frisch–Waugh), and only the
   embeddings are penalised.
4. **Additions from research.md:**
   - Spearman correlation
   - paired bootstrap 95% CIs that resample whole companies (2,000 draws)
   - a TF-IDF + ridge text benchmark
   - a company-disjoint split
   - a report of extreme returns

## Results

Main test set: the season split, 110 calls from October–December 2017. Lower MSE is better.

| Model | MSE | Spearman |
|---|---|---|
| Training mean | 0.324 | – |
| Audio | 0.316 | 0.075 |
| Text | 0.287 | 0.294 |
| Past volatility (OLS) | 0.270 | 0.422 |
| TF-IDF + past | 0.268 | 0.417 |
| Audio + past | 0.268 | 0.424 |
| Text + past | 0.260 | **0.449** |
| **Text + audio + past** | **0.255** | 0.439 |

Key comparisons (paired bootstrap by company, ΔMSE, negative favours the first model):

| Comparison | ΔMSE | 95% CI | One-sided p |
|---|---|---|---|
| Text + audio + past vs past | −0.0147 | [−0.0306, −0.0009] | 0.018 |
| Text + past vs past | −0.0101 | [−0.0221, +0.0007] | 0.034 |
| Text + audio + past vs text + past | −0.0046 | [−0.0114, +0.0019] | 0.087 |
| Text + past vs TF-IDF + past | −0.0080 | [−0.0212, +0.0066] | 0.128 |

Sensitivity checks (test MSE):

| Split | Test calls | Past | Text + past | Text + audio + past |
|---|---|---|---|---|
| Season (main) | 110 | 0.270 | 0.260 | **0.255** |
| Starter date split | 100 | 0.283 | 0.274 | **0.272** |
| Company-disjoint | 110 | **0.193** | 0.198 | 0.198 |

The full tables, with MAE and the chosen ridge α, are in [RESULTS.md](RESULTS.md).

## Observations

- **Past volatility is the strongest single predictor.** Every model without it does worse than
  past volatility alone.
- **Text probably adds information.** FinBERT text + past beats past alone by about 0.01 MSE on
  both time-based splits, but the evidence is borderline: both confidence intervals just touch
  zero.
- **Audio alone carries almost no signal** (Spearman about 0.03–0.08). Added to text + past
  volatility, it gives a small gain on both time-based splits, but the gain is significant on
  neither.
- **The gains do not carry over to new companies.** On the company-disjoint split, past
  volatility alone wins. Part of the time-based gain may come from the embeddings recognising
  a company or speaker rather than what was said or how it was said.
- **FinBERT is not clearly better than TF-IDF** (p = 0.13).

## Limitations

- **Small test sets.** 100–110 calls per split, so the intervals are wide. There is no
  correction for multiple comparisons.
- **Bootstrap by company only.** Calls in the same season share market shocks, so the
  intervals may be too narrow.
- **Not comparable to Qin & Yang's model.** They used CRSP prices, calendar-day horizons and an
  aligned sequence model. This study uses Yahoo prices, trading days and mean pooling.
- **Possible pretraining overlap.** FinBERT-tone's pretraining may include 2017 earnings
  calls, so this evaluates representations retrospectively rather than a strategy that could
  have run in 2017.
- **Selection bias.** The 82 calls from delisted companies are missing.

## Reproducing

The data is not in git. Download the five `ACL19_Release` parts from the Google Drive folder
linked in [GeminiLn/EarningsCall_Dataset](https://github.com/GeminiLn/EarningsCall_Dataset)
into `data/archive/`. Then follow the commands in [RESULTS.md](RESULTS.md#reproducing). They
were run on Windows with Python 3.12 and an NVIDIA GPU, using the dependencies pinned in
[requirements.txt](requirements.txt).

The offline unit tests need no data or network access:

```sh
python -m unittest discover -s experiments -p "test_*.py"
```

They cover label timing, the splits and purging, ridge regression, Spearman correlation and
the bootstrap: 12 tests, all passing.

Yahoo revises adjusted prices over time, so a fresh download can shift the labels slightly.

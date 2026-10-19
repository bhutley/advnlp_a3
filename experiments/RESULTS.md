# Earnings-call volatility: results

Run on 5 October 2026. The design and its caveats are in [research.md](research.md). This file
records what was run, what changed from the starter, and what came out.

**Question.** Does speech information improve post-call volatility forecasts beyond the
transcript text and pre-call volatility?

**Answer, for this cohort.** Text and audio together improve on past volatility alone. On the
main split, text + audio + past has the lowest error of any model, and its improvement over
past volatility is outside the 95% bootstrap interval. Audio's own contribution on top of text
and past volatility is small. It points the right way on both chronological splits, but it is
not statistically significant on either, and it disappears on unseen companies.

## Data

| | |
|---|---|
| Calls in the release | 572 from 280 companies, 89,722 aligned sentences (median 146 per call) |
| Calls with price data | 490 from 240 companies |
| Excluded | 82 calls from 40 delisted companies with no Yahoo history |
| Labelled | 490, every call that has price data |
| Audio clips skipped | 2 of 89,722: one empty file and one 0.07 s clip. In both cases forced alignment failed; the sentence's text was kept |

Target: the log standard deviation of daily `Adj Close` returns over 7 trading days. The window
starts from the first close strictly after the call date, so it excludes the immediate reaction.
The control is the log volatility of the 20 returns before the call. The median 7-day daily
volatility is 0.92%.

One call has a daily return above 15%: Illumina, January 2017, a real drop of about 24% in its
pre-call window. It is kept and not clipped.

## Changes from the starter

1. **Longer pre-call price window.** The `acl19_proc` price files start one month before each
   company's first call. That left 17–19 returns, so 109 first calls were dropped. Prices were
   re-downloaded with the same `fetch_prices.py`, starting three months before each company's
   first call, into `data/prices/`. `acl19_proc/` is unchanged, and no calls are now lost to the
   window.
2. **Season split** (`--split season`, now the default). The starter put its 60%/80% cuts at
   quantiles of the distinct call dates. Calls cluster in reporting seasons, so the test
   boundary fell mid-season and purging removed 41 of the 57 validation calls. The season split
   uses three periods, and purging removes only 6 calls:
   - **train:** calls from January to June, n = 251
   - **validation:** the Q2 season, July to September, n = 123
   - **test:** the Q3 season, October to December, n = 110

   Ridge α is chosen on validation, then every model is refit on train + validation. All
   validation outcomes end before October, and every model uses the same observations. The
   starter's date split is reported below as a sensitivity check.
3. **Past volatility is not penalized.** In the starter, ridge shrank the single past-volatility
   column as hard as each of the 1,536 embedding dimensions. As a result, text + audio + past did
   *worse* than past alone (0.284 against 0.270). Past volatility is now partialled out
   (Frisch–Waugh), and only the embedding coefficients are shrunk. "Past" alone is ordinary least
   squares. The α grid now reaches 10⁶, because audio models had been choosing the old upper
   limit.
4. **Final-study additions from research.md:**
   - Spearman correlation
   - paired bootstrap 95% CIs that resample whole companies (2,000 draws)
   - a TF-IDF + ridge text benchmark, with the vocabulary fitted on training calls only
   - a company-disjoint sensitivity split
   - a report of extreme returns

## Main result: season split (test = Q3 2017 season, 110 calls)

| Model | MSE | MAE | Spearman | α |
|---|---|---|---|---|
| Training mean | 0.3241 | 0.4301 | – | – |
| Past persistence | 0.2867 | 0.4084 | 0.422 | – |
| Past (OLS) | 0.2699 | 0.4006 | 0.422 | – |
| Text | 0.2874 | 0.4208 | 0.294 | 10³ |
| Audio | 0.3163 | 0.4296 | 0.075 | 10⁴ |
| Text + audio | 0.2914 | 0.4344 | 0.195 | 10³ |
| TF-IDF | 0.2941 | 0.4278 | 0.225 | 0.1 |
| TF-IDF + past | 0.2678 | 0.4057 | 0.417 | 1 |
| Audio + past | 0.2679 | 0.3996 | 0.424 | 10⁵ |
| Text + past | 0.2598 | 0.3929 | **0.449** | 10⁴ |
| **Text + audio + past** | **0.2552** | **0.3901** | 0.439 | 10⁴ |

Paired bootstrap differences in test MSE (resampled by company; negative favours the first
model):

| Comparison | ΔMSE | 95% CI | One-sided p |
|---|---|---|---|
| Text + audio + past vs past | −0.0147 | [−0.0306, −0.0009] | 0.018 |
| Text + past vs past | −0.0101 | [−0.0221, +0.0007] | 0.034 |
| Audio + past vs past | −0.0020 | [−0.0040, −0.0001] | 0.022 |
| **Text + audio + past vs text + past** | **−0.0046** | **[−0.0114, +0.0019]** | **0.087** |
| Text + past vs TF-IDF + past | −0.0080 | [−0.0212, +0.0066] | 0.128 |
| Past vs training mean | −0.0542 | [−0.1055, −0.0006] | 0.024 |

## Sensitivity checks

The key comparisons on each split (test MSE):

| Split | n test | Past | Text + past | Text + audio + past | Audio vs text + past: ΔMSE [95% CI] |
|---|---|---|---|---|---|
| Season (main) | 110 | 0.2699 | 0.2598 | **0.2552** | −0.0046 [−0.0114, +0.0019] |
| Starter date split | 100 | 0.2828 | 0.2739 | **0.2724** | −0.0015 [−0.0077, +0.0044] |
| Company-disjoint | 110 | **0.1928** | 0.1983 | 0.1980 | −0.0003 [−0.0165, +0.0157] |

Full JSON for every model is in `results/full_{season,date,company}.json`. Price-only runs are
in `results/price_only_{season,date}.json`.

## Interpretation

- **Past volatility is the strongest single predictor.** Every model without it is worse than
  past volatility alone.
- **Text probably adds something beyond past volatility.** FinBERT text + past beats past alone
  by about 0.01 MSE on both chronological splits. The evidence is borderline: one-sided
  p = 0.034 and 0.069, and both intervals just touch zero. Text alone ranks volatility
  moderately (ρ ≈ 0.3).
- **Audio on its own carries almost no usable signal** (ρ ≈ 0.03–0.08). The ridge model shrinks
  it nearly to zero. Combined with text and past volatility, it gives a small further gain on
  both chronological splits, but neither gain is significant. That fits Qin & Yang's claim that
  vocal cues help, but this setup cannot confirm it.
- **None of the gains carry over to companies the model has not seen.** On the company-disjoint
  split, past volatility alone is best. Part of the chronological gain may therefore come from
  embeddings recognising a company or speaker, rather than from what was said or how it was said.
  Frozen speech embeddings encode speaker identity and recording channel as well as delivery.
- **FinBERT is not clearly better than the cheap benchmark.** Its edge over TF-IDF + past is
  inside the noise (p = 0.13).

## Limitations

- **The test sets are small**, at 100–110 calls, and the confidence intervals are wide. Several
  comparisons are reported without a multiple-comparison correction.
- **Bootstrap resampling is by company only.** Calls in the same season share market shocks, and
  no date-block bootstrap or rolling-origin evaluation was run, so the intervals may be too
  narrow.
- **The setup is not comparable to Qin & Yang's MDRM.** That work uses CRSP prices,
  calendar-day horizons and an aligned sequence model. This study uses Yahoo `Adj Close`, trading
  days, and simple mean pooling of each modality, which discards sentence-level alignment.
- **Pretraining may overlap the test period.** FinBERT-tone's pretraining corpora include
  earnings-call transcripts and may cover 2017. This is a retrospective evaluation of
  representations, not a strategy that could have been run in 2017. Model revisions are pinned in
  `features.npz`: FinBERT-tone `4921590d3c0c`, wav2vec2-base `0b5b8e868dd8`.
- **There may be selection bias.** The 82 calls from delisted companies are missing, and those
  firms may have had different volatility.
- **Label timing is approximate.** Only call dates are known, so labels start the day after the
  first post-call close.

## Reproducing

From the repository root, on Windows with an NVIDIA GPU:

```sh
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/Scripts/python -m pip install -r experiments/requirements.txt yfinance

# Download the five ACL19_Release parts from the Google Drive folder linked in
# github.com/GeminiLn/EarningsCall_Dataset into data/archive/, then:
.venv/Scripts/python experiments/extract_release.py
.venv/Scripts/python experiments/build_manifests.py --release data/ACL19_Release --out data

# Prices with a three-month pre-call window, written to data/prices/:
.venv/Scripts/python -c "import pandas as pd; c=pd.read_csv('acl19_proc/data/company.csv',dtype=object,keep_default_na=False); ok=c.status=='ok'; c.loc[ok,'first_call']=(pd.to_datetime(c.loc[ok,'first_call'])-pd.DateOffset(months=2)).dt.strftime('%Y-%m-%d'); c.to_csv('data/company_prewindow.csv',index=False)"
cd acl19_proc && ../.venv/Scripts/python fetch_prices.py --company-csv ../data/company_prewindow.csv --data-dir ../data/prices && cd ..

.venv/Scripts/python experiments/volatility_baseline.py labels --calls data/calls.csv --prices data/prices --out data/labels.csv
.venv/Scripts/python experiments/embed_calls.py --segments data/segments.csv --out data/features.npz   # about 2.5 h on an RTX 4070, resumable
.venv/Scripts/python experiments/volatility_baseline.py train --labels data/labels.csv --features data/features.npz --texts data/call_texts.csv --split season --out experiments/results/full_season.json
.venv/Scripts/python -m unittest discover -s experiments -p "test_*.py"
```

Yahoo revises its adjusted prices over time, so a later re-download can shift the labels
slightly.

# modelling — do earnings-call text and audio predict post-call volatility?

Tests whether FinBERT text features and HuBERT/Praat audio features predict post-call
stock volatility beyond past volatility. The design is an ablation ladder evaluated with
firm-grouped cross-validation, in two pre-registered rounds. Direction (the sign of the
abnormal return) is a secondary target.

**Start with `notebooks/results.ipynb`.** It presents both rounds from the saved results
and runs in seconds, locally or in Colab.

| Document | Content |
|---|---|
| `docs/plans/2026-10-07-modelling-design.md` | Round 1 design: ladder, CV, H1/H2 |
| `docs/plans/2026-10-07-round2-design.md` | Round 2 design, committed before any round-2 model ran: H3–H5, runtime decision, deviations log |
| `docs/plans/*-plan.md` | Implementation plans |

## Results in brief

488 calls from 240 firms, 10 repeats × 5 firm-grouped folds. Out-of-fold MSE:

| | vol_3 | vol_7 | vol_15 | vol_30 |
|---|---|---|---|---|
| B0 raw past volatility | 1.346 | 0.661 | 0.351 | 0.185 |
| **R1** past vol + format | **0.511** | **0.317** | **0.196** | **0.120** |
| R2 + text | 0.480 | 0.318 | 0.200 | 0.121 |
| R4 + text + audio | 0.526 | 0.346 | 0.216 | 0.126 |

| Hypothesis | Result |
|---|---|
| **H1** text helps (R2 vs R1) | Not supported. 3-day gain of 6%, Holm p = 0.34 |
| **H2** audio adds to text (R4 vs R2) | Fails in the opposite direction: audio raises MSE at every horizon |
| **H3** prosody helps (R1 + prosody vs R1) | Not supported; slightly worse at every horizon |
| **H4** format explains audio's harm (residualised vs raw) | Not supported; no difference |
| **H5** attention finds informative sentences (vs uniform) | Not supported; attention is worse at every horizon |

Further results:

- **Descriptive:** under a naive, ungrouped split, text appears to beat past volatility
  significantly at 3 days (p = 0.019). Firm-grouped CV removes that result.
- **Direction (exploratory):** text reaches an AUC of about 0.57; audio adds nothing.
- **Leakage checks:** permuted targets show no spurious gains. Ungrouped CV understates
  audio-only MSE by 3–6%.

## Requirements

Python 3.13 or newer, and [uv](https://docs.astral.sh/uv/). `uv sync` installs
scikit-learn, pandas, numpy, scipy, matplotlib, pyarrow, torch, and (dev) pytest,
nbconvert and ipykernel.

The runs read the committed outputs of `../event_study`, `../text_embeddings` and
`../audio_embeddings`. Round 2's attention model (D) also needs those stages' git-ignored
`sentence_emb.npy` files; see their READMEs to regenerate them.

## Usage

Run from this directory:

```sh
uv sync
uv run pytest                                   # 97 tests, about 1 minute

uv run python run_ladder.py                     # round 1 → results/        (~4.4 h)
uv run python run_round2.py                     # round 2 → results/round2/ (~3.2 h)

uv run python notebooks/build_results_notebook.py
uv run jupyter nbconvert --to notebook --execute notebooks/results.ipynb --output results.ipynb
```

Both run scripts take `--quick` (2 repeats) for a trial run. `run_round2.py` also takes
`--skip-d` (no attention model, so no sentence arrays needed) and `--d-repeats N`.
`--alphas`, `--ks` and `--layers` override the grid, for tests and trials only.

## Method

| Step | Choice |
|---|---|
| Sample | `ok` events with `audio_call_ok`: 488 calls, 240 firms (1–4 calls each) |
| Targets | `vol_3/7/15/30` (log volatility, Qin & Yang's definition); `dir_0_1`, `dir_0_5` (sign of CAR) |
| Blocks | `past` (`vol_pre_n`), `format` (8 MP3 formats, one-hot), `text_emb` (FinBERT-tone 768), `text_sent` (6 sentiment summaries), `audio_emb` (HuBERT 13 × 768), `prosody` (4 format-robust Praat features) |
| Ladder | B0 (`vol_pre_n` itself), R1 past + format, R2 + text, R3 + audio, R4 both, R4 without past; robustness: R4 with ProsusAI FinBERT or wav2vec2 |
| CV | 5 outer folds × 10 repeats, **grouped by firm**; inner grouped 4-fold CV for tuning |
| Models | Ridge (logistic for direction). Small blocks standardised; embeddings standardised then PCA k ∈ {8, 16, 32, 64, none}; audio layer ∈ {3, 6, 9, 12}; 13 α values |
| Comparisons | Per-call loss averaged over repeats; paired differences; firm-clustered bootstrap (10,000 draws); Holm across horizons within each hypothesis |
| Round 2 | R1P (+ prosody only); R3_resid (audio with format-predictable variance removed in-fold); ATT/UNI (attention vs uniform pooling over about 150 sentences per call, PCA 32, L-BFGS, 3 seeds); the ladder under naive CV |

Leakage is tested directly. Spy transformers record the rows every fit sees, including
inside the inner CV, and must never see an outer-test row. Planted-signal tests check
that each model finds a signal when one exists, and doesn't when none does.

## Files

| Path | Description |
|---|---|
| `data.py` | Joins the earlier stages into `CallData` (one row per call, row-aligned arrays) |
| `features.py` | Targets, ladder rungs (`RUNGS`, `ROUND2_RUNGS`), block assembly |
| `cv.py` | Firm-grouped repeated splits, inner grouped CV, `ungrouped` |
| `pipelines.py` | Block-wise sklearn pipelines, `LayerSelect`, `FormatResidualizer`, grids |
| `evaluate.py` | `run_rung`, metrics, firm-clustered `compare`, `holm` |
| `attention.py` | Sentence loading, `AttentionPool`, `AttentionRegressor`, `run_attention` |
| `run_ladder.py`, `run_round2.py` | CLIs for rounds 1 and 2 |
| `notebooks/` | `results.ipynb` and the script that builds it |
| `tests/` | pytest suite: synthetic data, leakage spies, planted signals, end-to-end runs |

### `results/` (round 1) and `results/round2/`

| File | Content |
|---|---|
| `oof.parquet` | Every out-of-fold prediction: `event_id, target, rung, repeat, fold, y, y_hat, params` (+ `cv` in round 2) |
| `metrics.csv` | Per target × rung: mean and SD over repeats of MSE, or AUC, accuracy and log loss |
| `comparisons.csv` | Rung comparisons with CI, p and Holm p (round 2 adds a `status` column) |
| `sanity.csv` | Round 1: permutation null and grouping effect |
| `attention_weights.parquet` | Round 2: each call's top-3 sentences per modality under attention |
| `run.json` | Settings, grid, versions, runtime |
| `figures/` | Dot plots of MSE by rung; permutation null; grouped vs naive |

## Caveats

- **Small sample, one year:** 488 calls from 2017 against 768-dimension representations.
- **Date clustering:** the firm-clustered bootstrap doesn't model correlation across firms
  on the same dates.
- **Trading days:** volatility windows count trading days; Qin & Yang describe calendar
  days.
- **One speaker per call:** the audio and text are one executive's remarks, not the
  analyst Q&A.
- **Recording format** is controlled as a covariate, not removed. It is encoded in the
  audio embeddings and weakly related to `vol_30`.
- **Round 2's attention model** chose the largest pre-registered weight decay in 189 of
  200 folds. H5 stands as run; a wider grid would be an exploratory follow-up.

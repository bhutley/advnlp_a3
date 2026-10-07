# Modelling Round 2 Design (pre-registered)

Date: 2026-10-07
Round 1 design: `docs/plans/2026-10-07-modelling-design.md`. Round 1 results: `results/`.

## Status

This document was written **after** round 1's results and **before** any round-2 model
was run. Round 2's hypotheses, comparisons, success criteria and the D runtime rule
below are fixed. Any change made after round-2 results are seen will be logged in
**Deviations** at the end, and the comparisons it affects become exploratory.

## Why a second round

Round 1, firm-grouped and tuned in-fold with format controlled, found:

- **H1** (text helps volatility): not supported. Only a 3-day MSE gain of 6%, Holm p = 0.34.
- **H2** (audio adds to text): failed in the opposite direction. Adding audio
  significantly increased MSE at every horizon.
- Past volatility, recalibrated (R1), dominates.

Round 2 tests specific explanations for H2's failure, each motivated by facts known
before round 1 ran (the embeddings' dimensionality, the format confound, Qin & Yang's
sentence-level design), not by searching round 1's results:

| | Follow-up | Question |
|---|---|---|
| A | Prosody-only audio | Does low-dimensional, interpretable audio help where 768-dimension embeddings don't? |
| B | Format-residualised embeddings | Is the recording-format confound what makes audio hurt? |
| C | Naive-split replication | Would a typical ungrouped split have made audio look helpful? (descriptive) |
| D | Sentence-level attention pooling | Does averaging ~150 sentences hide the informative ones? |

## Hypotheses

All are on the volatility targets, tested at each of the four horizons (`vol_3`, `vol_7`,
`vol_15`, `vol_30`), Holm-corrected across horizons within each hypothesis. This is the
same rule as round 1. The test is the firm-clustered paired bootstrap of per-call MSE,
as in round 1.

| | Hypothesis | a | b |
|---|---|---|---|
| **H3** | Low-dimensional prosody helps | `R1P` | `R1` |
| **H4** | The format confound explains audio's harm | `R3_resid` | `R3` |
| **H5** | Attention finds informative sentences | `ATT` | `UNI` |

A hypothesis is **supported at a horizon** if its Holm-adjusted one-sided p < 0.05.

**Exploratory** (reported, no confirmatory claims):

- R3_resid vs R1
- ATT vs R1
- UNI vs R4 (a sanity check: the two should be similar)
- direction targets for R1P, R3_resid, ATT and UNI

**Descriptive (C):** grouped vs naive MSE for every rung, and the naive-split H1/H2
comparisons.

## A: prosody-only audio (`R1P`)

`past` + `format` + `prosody` (the four format-robust Praat features `f0_sd_st`,
`speech_rate`, `intensity_sd_db`, `voiced_frac`). The existing pipeline with the
13-value α grid.

## B: format-residualised audio (`R3_resid`)

R3's blocks, with the audio branch changed:

- The branch receives the audio columns **and** the format columns.
- Steps: `LayerSelect` (keeps the format columns) → **`FormatResidualizer`** → `StandardScaler` → `PCA`.
- `FormatResidualizer.fit` runs an ordinary least-squares regression of each embedding
  dimension on the format one-hots (with an intercept) on the **training rows only**.
  `transform` returns the embedding minus its predicted value, using those training
  coefficients.
- Same grid as R3: α × k ∈ {8, 16, 32, 64, none} × layer ∈ {3, 6, 9, 12}.

## C: naive-split replication

The volatility ladder (B0, R1, R2, R3, R4) re-run with **ungrouped** CV: outer and inner
splits treat every call as its own group. It uses 10 repeats, round 1's seeds and the
full grid. Output rows are tagged `cv = "naive"`.

## D: attention pooling over sentences (`ATT`, control `UNI`)

**Inputs:** for each call, its sentences with `audio_ok` true, matched on `event_id` +
`sent_idx`:

- text: `text_embeddings/data/finbert_tone/sentence_emb.npy` (768)
- audio: `audio_embeddings/data/hubert/sentence_emb.npy`, layer 6 (768)

These arrays are local and git-ignored, so D runs only where they exist.

**In-fold reduction:** per modality, `StandardScaler → PCA(32)`, fitted on the
training calls' sentences. The size 32 is fixed, not tuned.

**Model** (torch, CPU, full batch, sentences padded with a mask):

- For each modality m, an attention vector w_m (32): scores s_i = w_m · x_i, weights
  a = softmax over the call's real sentences (padding gets weight 0), pooled vector
  p_m = Σ a_i x_i.
- Head: linear on [p_text, p_audio, scaled `past` and `format`] → ŷ.
- About 150 parameters. MSE loss with weight decay on all weights, optimised by L-BFGS.

**Control `UNI`:** the identical model with w_text = w_audio = 0 frozen, so the
weights are uniform and p_m is the masked mean. Its features, head, optimiser and tuning
are the same as ATT's.

**Tuning:** inner grouped CV over weight decay ∈ {10⁻³, 10⁻², 10⁻¹, 1}. Each fit is
the average of **3 random initialisations**. The outer splits are round 1's.

**Interpretability:** for every call and modality, the three highest-weighted
sentences (`sent_idx`, weight) go to `attention_weights.parquet`, from the refit on
each outer training fold, applied to that fold's test calls.

**Runtime rule (fixed now):** a `--quick` trial measures the time per fit. If the full
D run (10 repeats) is projected to exceed **3 hours**, D uses the **first 5 of round 1's
10 repeats**: identical splits, 2,440 out-of-fold predictions per model. This choice
is made from timing alone and recorded below before the full run.

## Code

All additions; round-1 code paths are unchanged, so round 1 stays reproducible.

| File | Change |
|---|---|
| `pipelines.py` | `FormatResidualizer`; `build_pipeline(..., residualize=True)` routes the format columns into the audio branch |
| `features.py` | Rungs `R1P` and `R3_resid` (`R3_resid` uses R3's blocks plus the residualise flag) |
| `cv.py` | `ungrouped(groups)`: each row its own group |
| `attention.py` | Sentence loading, `AttentionPool` (torch module), `AttentionRegressor` (sklearn estimator), weight extraction |
| `run_round2.py` | CLI: A, B, C, D → `results/round2/` |

`pyproject.toml` gains `torch`.

## Outputs (`results/round2/`)

- `oof.parquet`: same columns as round 1, plus `cv` (`grouped` / `naive`)
- `metrics.csv`
- `comparisons.csv`: H3–H5 (Holm), exploratory, and descriptive naive-split rows, with
  a `status` column (`confirmatory` / `exploratory` / `descriptive`)
- `attention_weights.parquet`
- `run.json`: settings, versions, the D repeat count, runtime
- `figures/mse_round2.png`: dot plot of MSE by rung, both rounds
- `figures/naive_vs_grouped.png`

## Testing

- **`FormatResidualizer`:** on the training rows, residuals are uncorrelated with
  format; transform uses the training coefficients. The leakage spy test is extended to
  `R3_resid`.
- **`ungrouped`:** each call is its own group; each call is tested once per repeat.
- **Attention:**
  - frozen-zero attention gives exactly the masked mean
  - padding doesn't change outputs
  - the estimator works inside `GridSearchCV` with the inner grouped CV
  - fixed seeds are deterministic
- **Planted signal:** when one sentence per call carries the signal, ATT clearly beats
  UNI. When the signal is spread across all sentences, ATT doesn't beat UNI.
- **Sentence loading:** alignment on `event_id` + `sent_idx`, and the `audio_ok` filter.
- **End to end:** `run_round2.py --quick` on the synthetic project, with synthetic
  sentence arrays added to `conftest`.

## Reporting

The notebook shows round 1 and round 2 side by side, labelled by status. Round 2's
confirmatory claims are limited to H3–H5 as specified here.

## Runtime decision (Task R7, timing only)

A `--quick` trial on the real data (2 repeats; 1 D repeat) took 27 min. Its outputs were
deleted unread. D took about 188 s per horizon per repeat (ATT about 156 s, UNI about
32 s), so the full D run is projected at about 2.1 h. That is under the 3-hour threshold,
so **D uses all 10 repeats**, as pre-specified. The projected total for round 2 is
about 3.3 h.

## Deviations

*None yet.*

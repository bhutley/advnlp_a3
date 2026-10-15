# Round 3 Design: Prepared Remarks vs Answers (pre-registered)

Date: 2026-10-13
Round 1 and 2: `modelling/docs/plans/` and `modelling/results/`. The split this round
uses: `remarks_vs_answers/` (commit f590d63).

## Status

Registered by the commit that adds this document, before any round-3 feature is
computed, any round-3 model is run, or any section-level quantity is related to any
target. Round 3's hypotheses, comparisons, sample and runtime rule below are fixed. Any
later change is logged in **Deviations** at the end, and the comparisons it affects
become exploratory.

What is known at the time of writing:

- Rounds 1–2: recalibrated past volatility (R1) dominates. Call-level text helps at most
  at 3 days (6%, Holm p = 0.34). Audio, in four forms, makes predictions worse.
- The split: each sentence is `remarks`, `answers`, `closing` or `heading`. A hand audit
  of 50 random modelling calls found the remarks/answers boundary correct in 50/50
  (95% CI 0.93–1.00) and the answers/closing boundary in 49/50.
- Sample facts below (sizes, firms, window availability). No section-level feature has
  been computed, and none has been related to a target.

## Question

Do the executive's answers to analysts' questions carry more information about post-call
volatility than their prepared remarks, in text and in voice? Prior work motivates the
direction: Q&A is unscripted, and prior studies find Q&A tone more informative for
markets than the presentation (Price et al., 2012; Matsumoto, Pronk & Roelofsen, 2011
[to verify]); Todd et al. (2025) found audio helped sentiment classification mainly in
Q&A sentences.

## Sample

The 488 modelling calls of rounds 1–2, restricted to calls with **at least 10
`audio_ok` sentences in both the remarks and the answers**: **457 calls, 232 firms**.
(At least 5: 467 calls; at least 20: 428.) Every round-3 model uses these 457 calls, so
all comparisons are paired. `closing` and `heading` sentences are not used.

Rules in the sample: 397 `para1`, 49 `handover`, 11 `handback`. Median sentences per
call: 67 remarks, 80 answers.

## Section features

For each call and each section s ∈ {rem, ans}, computed within the call (no fitting
across calls, so no leakage):

| Block | Contents | Sentences used |
|---|---|---|
| `text_emb_s` | mean FinBERT-tone sentence vector (768), the round-1 pooling | all sentences of the section |
| `text_sent_s` | the six round-1 sentiment summaries | all sentences of the section |
| `audio_emb_s` | mean HuBERT vector, all 13 hidden states (13 × 768) | `audio_ok` sentences |
| `prosody_s` | the four round-1 prosody features (F0 SD in semitones, speech rate, intensity SD, voiced fraction), averaged | `audio_ok` sentences |
| `size` | log number of sentences in the remarks and in the answers (2) | — |
| `shift` | `prosody_ans − prosody_rem` (4): how the voice changes under questioning | — |

"text_s" means `text_emb_s` + `text_sent_s`.

## Rungs

All rungs include **`past` + `format` + `size`**, so no content block is credited with
section length or recording format. Direction targets drop `past`, as in round 1.

| Rung | Blocks |
|---|---|
| **S1** | past + format + size (baseline) |
| **TR** | S1 + text_rem |
| **TA** | S1 + text_ans |
| **TB** | S1 + text_rem + text_ans |
| **PR** | S1 + prosody_rem |
| **PA** | S1 + prosody_ans |
| **PD** | S1 + shift |
| AR | S1 + audio_emb_rem (exploratory) |
| AA | S1 + audio_emb_ans (exploratory) |
| R1, R2 | round 1's call-level rungs, refit on the 457 calls (reference) |

## Hypotheses (confirmatory)

On the four volatility targets (`vol_3`, `vol_7`, `vol_15`, `vol_30`), Holm-corrected
across horizons within each hypothesis, as in rounds 1–2. A hypothesis is **supported at
a horizon** if its Holm-adjusted one-sided p < 0.05.

| | Hypothesis | a | b |
|---|---|---|---|
| **H6** | Answers' text helps | TA | S1 |
| **H7** | Answers' text beats remarks' text | TA | TR |
| **H8** | Answers' prosody beats remarks' prosody | PA | PR |
| **H9** | The prosody shift under questioning helps | PD | S1 |

Prosody, not embeddings, carries the audio hypotheses: round 2 showed the embeddings hurt,
while four features cannot overfit much.

**Interpretation, fixed now.** Failing to reject H7 or H8 does not show that the two
sections are equally informative; the report gives the 95% interval of each difference
and says what size of difference it rules out. Given rounds 1–2, the expected outcome
is that H8 and H9 fail and H6 is at most borderline at 3 days.

## Exploratory

Reported, no confirmatory claims:

- TR vs S1, PR vs S1, PA vs S1, TB vs TA (do remarks add to answers?)
- AR vs S1, AA vs S1, AA vs AR
- TB vs R2 on the 457 calls (does splitting beat pooling?)
- **Equal length:** TA vs TR and PA vs PR with each section's features computed from
  the same number of sentences per call: a random subset (seed 0) of size
  min(n_rem, n_ans) from each section.
- **ProsusAI FinBERT** in place of FinBERT-tone for TA vs S1 and TA vs TR.
- **Direction** (AUC, accuracy, log loss), all rungs without `past`, for the sign of the
  market-model CAR over [0, +1], [0, +5] and a new post-reaction window **[+2, +5]**,
  which excludes the reaction day for every timing class and so the press release's
  first impact. [+2, +5] is available for all 457 calls.

## Evaluation (as rounds 1–2)

- Outer: 5 folds × 10 repeats, grouped by firm (ticker), seed 0, drawn on the 457 calls
  (new splits; round 1's were drawn on 488).
- Inner: grouped 4-fold CV within each outer training set.
- Models: ridge (logistic for direction). Small blocks (past, format, size, prosody,
  shift, sentiment) standardised; embedding blocks standardised then PCA in-fold.
- Grid: α (13 values, 10⁻² to 10⁴); PCA k ∈ {8, 16, 32, 64, none}, shared across a
  rung's embedding blocks; audio layer ∈ {3, 6, 9, 12} for AR and AA.
- Comparison: per-call loss averaged over repeats, paired difference, firm-clustered
  bootstrap (10,000), one-sided p with +1 correction, Holm across horizons.

## Features build

- **Text:** regenerate FinBERT-tone and ProsusAI sentence vectors with the
  `text_embeddings` code at its pinned revisions (about 2–3 min each), then average per
  section.
- **Audio:** rerun HuBERT with the `audio_embeddings` code, accumulating per-section means
  of all 13 layers (about 53 min).
- **Prosody:** from `audio_embeddings/data/praat/sentences.csv` (already saved).

Acceptance (must pass): for each call, the sentence-count-weighted mean of its section
means over all four sections reproduces the committed call vector
(`text_embeddings/.../call_emb.npy`, `audio_embeddings/.../call_emb.npy`) within the
MPS/CPU tolerance, and the same for the Praat call features. This proves the section
features are the round-1 features, split.

## Testing

- Planted signal: with the signal in the answers block, TA beats TR and S1; with it in
  the remarks block, TR beats TA; with no signal, neither beats S1.
- Section aggregation: weighted recombination equals the call mean (synthetic data).
- Equal-length subsampling uses min(n_rem, n_ans) sentences per section, is seeded, and
  is deterministic.
- The leakage spy test from round 1 extended to the new blocks.
- End to end: `--quick` on the synthetic project.

## Runtime rule (fixed now)

A `--quick` trial on the real data (2 repeats) measures time per fit; its outputs are
deleted unread. If the full run is projected to exceed **8 hours**, AR and AA
(exploratory) use the first 5 of the 10 repeats. Decided from timing alone and recorded
below before the full run.

## Outputs (`remarks_vs_answers/results/`)

`oof.parquet`, `metrics.csv`, `comparisons.csv` (status: confirmatory / exploratory),
`run.json`, figures. Section features in `remarks_vs_answers/data/features/`.

## Known limitations

- The answers contain a few analysts' words that the dataset assigned to the executive
  (e.g. Waste Management, 16 Feb 2017, sentences 31 and 33); this adds noise to
  `text_ans`.
- One speaker per call, not always the CEO; other executives' remarks and answers are
  absent.
- Same sample caveats as rounds 1–2: one year, survivorship, date clustering.

## Runtime decision (timing only)

The `--quick` trial (`run_round3.py --quick`: 2 repeats, all rungs and targets) ran on
the real data on 16 October 2026 on an Apple M5 MacBook Air and took 35 minutes. By
construction it kept no predictions, metrics or comparisons, only
`results/round3_trial/timing.json`. It projects the full run at **2.95 hours** with all
10 repeats (2.04 hours with AR and AA on 5). That is under the 8-hour threshold, so
**AR and AA use all 10 repeats**, as pre-specified (`--audio-repeats 10`).

## Deviations

*None yet.*

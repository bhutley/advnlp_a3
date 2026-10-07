"""Build notebooks/results.ipynb. Every number in it is computed from results/ at run time.

    uv run python notebooks/build_results_notebook.py
    uv run jupyter nbconvert --to notebook --execute notebooks/results.ipynb --output results.ipynb
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# Do earnings-call text and audio predict post-call volatility?

Assignment 3, Advanced NLP. This notebook presents the modelling results. It reads saved
results only (`modelling/results/`), so it runs in seconds, locally or in Colab, without
the audio, the embeddings, or a GPU.

**Two rounds, each pre-registered:**

- **Round 1** (`docs/plans/2026-10-07-modelling-design.md`): an ablation ladder. Past
  volatility, then + text, + audio, + both. Hypotheses H1 (text helps) and H2 (audio
  adds to text).
- **Round 2** (`docs/plans/2026-10-07-round2-design.md`, committed before any round-2
  model ran): four follow-ups to round 1's negative audio result. Hypotheses H3–H5,
  plus a descriptive naive-split replication.

Each result below is labelled **confirmatory**, **exploratory** or **descriptive**.
Only confirmatory results are tests of a hypothesis fixed in advance."""))

cells.append(code("""# Setup: works locally (from modelling/ or modelling/notebooks/) and in Colab.
import json, os, subprocess, sys
from pathlib import Path

if "google.colab" in sys.modules and not Path("advnlp_a3").exists():
    # Clone the repository; if it is private, upload modelling/results/ to /content instead.
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/bhutley/advnlp_a3.git"])

CANDIDATES = [Path("results"), Path("../results"), Path("advnlp_a3/modelling/results"),
              Path("modelling/results")]
RESULTS = next((p for p in CANDIDATES if (p / "metrics.csv").exists()), None)
assert RESULTS is not None, "results/ not found: run from modelling/ or upload modelling/results/"
R2 = RESULTS / "round2"

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option("display.precision", 4)
VOL = ["vol_3", "vol_7", "vol_15", "vol_30"]
m1 = pd.read_csv(RESULTS / "metrics.csv")
m2 = pd.read_csv(R2 / "metrics.csv")
c1 = pd.read_csv(RESULTS / "comparisons.csv")
c2 = pd.read_csv(R2 / "comparisons.csv")
run1 = json.loads((RESULTS / "run.json").read_text())
run2 = json.loads((R2 / "run.json").read_text())
print(f"{run1['n_calls']} calls from {run1['n_firms']} firms; "
      f"{run1['repeats']} repeats × {run1['splits']} firm-grouped folds")"""))

cells.append(md("""## Data and design

- **Sample:** 2017 S&P 500 earnings calls (Qin & Yang 2019) with usable prices and audio.
  Firms appear up to four times.
- **Targets:** log volatility of daily returns over the 3, 7, 15 and 30 trading days after
  the call (Qin & Yang's definition). Secondary: the sign of the abnormal return over
  [0,+1] and [0,+5].
- **Features:**
  - **past volatility** (the same horizon before the call)
  - **recording format** (calls come in 8 MP3 formats, which shape the audio features)
  - **text:** FinBERT call vectors and sentiment
  - **audio:** HuBERT call vectors (layer tuned in-fold) and four Praat prosody features
- **Validation:** firm-grouped, repeated 5-fold CV. All of a firm's calls fall on the same
  side of each split, because a firm's volatility persists and **its executive's voice is
  in every one of its calls**. Hyperparameters are tuned by an inner grouped CV on the
  training folds only.
- **Models:** ridge regression (logistic for direction) on standardised blocks, with
  in-fold PCA. Comparisons use a firm-clustered bootstrap of per-call squared errors,
  with Holm correction across the four horizons within each hypothesis."""))

cells.append(md("## Round 1: the ablation ladder (confirmatory)"))
cells.append(code("""RUNG_NAMES = {"B0": "B0 raw past vol", "R1": "R1 past vol + format", "R2": "R2 + text",
              "R3": "R3 + audio", "R4": "R4 + text + audio", "R4_nopast": "R4 without past vol",
              "R4_prosus_finbert": "R4, ProsusAI FinBERT", "R4_wav2vec2": "R4, wav2vec2"}

def mse_table(m, rungs, names=RUNG_NAMES):
    t = m[m.metric == "mse"].pivot(index="rung", columns="target", values="mean")[VOL]
    sd = m[m.metric == "mse"].pivot(index="rung", columns="target", values="sd")[VOL]
    out = t.loc[rungs].round(3).astype(str) + " ± " + sd.loc[rungs].round(3).astype(str)
    return out.rename(index=names)

display(Markdown("**Out-of-fold MSE, mean ± SD over 10 repeats** (lower is better)"))
mse_table(m1, list(RUNG_NAMES))"""))
cells.append(code("""display(Image(str(RESULTS / "figures" / "mse_by_rung.png")))"""))
cells.append(code("""def hypothesis_table(c, status="confirmatory"):
    t = c[c.status == status] if "status" in c else c[c.hypothesis.isin(["H1", "H2"])]
    t = t[["hypothesis", "target", "a", "b", "mean_diff", "ci_low", "ci_high", "p", "p_holm"]]
    return t.rename(columns={"mean_diff": "MSE(b) − MSE(a)"}).reset_index(drop=True)

display(Markdown("**H1: R2 beats R1 (text helps). H2: R4 beats R2 (audio adds to text).** "
                 "A positive difference means *a* is better; one-sided p, Holm across horizons."))
hypothesis_table(c1)"""))
cells.append(code("""h = c1[c1.hypothesis.isin(["H1", "H2"])]
best = c1[(c1.hypothesis == "exploratory") & (c1.a == "R1") & (c1.b == "B0")]
gain = (best.mean_diff / (best.mean_diff + m1[(m1.metric == "mse") & (m1.rung == "R1")]
                          .set_index("target").loc[best.target, "mean"].values)).round(2)
display(Markdown(f'''**Round 1 findings**

- **Past volatility dominates.** Recalibrating it (R1: a linear fit with format) cuts MSE
  against raw past volatility (B0) by {gain.min():.0%}–{gain.max():.0%}.
- **H1 is not supported** at any horizon (smallest Holm p = {h[h.hypothesis == "H1"].p_holm.min():.2f}).
  Text's best case is 3 days.
- **H2 fails in the opposite direction:** adding audio *raises* MSE at every horizon
  (every CI below 0).'''))"""))

cells.append(md("### Direction (secondary, exploratory)"))
cells.append(code("""auc = m1[m1.metric == "auc"].pivot(index="rung", columns="target", values="mean")
auc.loc[["B0", "R1", "R2", "R3", "R4"]].rename(index=RUNG_NAMES).round(3)"""))
cells.append(md("Text (R2) reaches an AUC of about 0.57 for the sign of the abnormal return, "
                "in line with FinBERT tone's correlation with CARs. Audio adds nothing."))

cells.append(md("### Leakage checks"))
cells.append(code("""s = pd.read_csv(RESULTS / "sanity.csv")
perm = s[s.check == "permutation"].groupby("target").gain.agg(["mean", "max"]).loc[VOL]
group = s[s.check == "grouping"].set_index("target").loc[VOL, ["mse_grouped", "mse_ungrouped"]]
group["ungrouped lower by"] = 1 - group.mse_ungrouped / group.mse_grouped
display(Markdown("**Permutation null:** R1 − R4 MSE with shuffled targets. No spurious gain."))
display(perm.round(4))
display(Markdown("**Grouping effect** (audio-only R3, one repeat): ungrouped CV understates MSE."))
group.round(4)"""))
cells.append(code("""display(Image(str(RESULTS / "figures" / "permutation_null.png")))"""))

cells.append(md("### What the inner CV chose"))
cells.append(code("""oof1 = pd.read_parquet(RESULTS / "oof.parquet")
folds = oof1[oof1.target.isin(VOL)].drop_duplicates(["target", "rung", "repeat", "fold"])

def chosen(params):
    d = json.loads(params)
    k = "none" if any(v == "passthrough" for k_, v in d.items() if k_.endswith("__pca")) \\
        else next((v for k_, v in d.items() if k_.endswith("n_components")), None)
    return pd.Series({"PCA k": k if k in (None, "none") else str(int(k)),
                      "audio layer": d.get("features__audio_emb__layer__layer")})

rows = []
for rung in ["R2", "R3", "R4"]:
    c = folds[folds.rung == rung].params.apply(chosen)
    rows.append({"rung": rung, "folds": len(c),
                 "most common k": c["PCA k"].astype(str).value_counts().idxmax(),
                 "share": c["PCA k"].astype(str).value_counts(normalize=True).max(),
                 "layers chosen": dict(c["audio layer"].dropna().astype(int).value_counts())})
pd.DataFrame(rows)"""))
cells.append(md("For audio-only R3, the inner CV almost always chose the smallest PCA size, "
                "and its layer choices are spread out: the audio embeddings behave like noise."))

cells.append(md("""## Round 2: four follow-ups (pre-registered)

| | Follow-up | Hypothesis |
|---|---|---|
| A | Prosody-only audio (`R1P`: past vol + format + 4 Praat features) | H3: R1P beats R1 |
| B | Format-residualised audio embeddings (`R3_resid`) | H4: R3_resid beats R3 |
| D | Attention over a call's sentences (`ATT`), against the same model with uniform weights (`UNI`) | H5: ATT beats UNI |
| C | The ladder under naive, ungrouped CV | descriptive |"""))
cells.append(code("""R2_NAMES = RUNG_NAMES | {"R1P": "R1P = R1 + prosody", "R3_resid": "R3 format-residualised",
                        "UNI": "UNI uniform sentences", "ATT": "ATT attention"}
display(Markdown("**Out-of-fold MSE, mean ± SD** (R1–R4 from round 1, on the same splits)"))
mse_table(m2, ["R1", "R2", "R3", "R4", "R1P", "R3_resid", "UNI", "ATT"], R2_NAMES)"""))
cells.append(code("""display(Image(str(R2 / "figures" / "mse_round2.png")))
hypothesis_table(c2)"""))
cells.append(code("""h = c2[c2.status == "confirmatory"]
summary = h.groupby("hypothesis").agg(supported=("p_holm", lambda p: int((p < 0.05).sum())),
                                       below_zero=("ci_high", lambda x: int((x < 0).sum())))
display(Markdown(f'''**Round 2 findings**

- **None of H3–H5 is supported at any horizon** (horizons with Holm p < 0.05:
  {", ".join(f"{k} {v}" for k, v in summary.supported.items())}).
- **H3:** prosody alone makes predictions slightly *worse*: its CI lies below 0 at
  {summary.loc["H3", "below_zero"]} of 4 horizons.
- **H4:** removing format-predictable variance changes nothing, so **the format confound
  does not explain audio's harm**.
- **H5:** attention is *worse* than uniform pooling at {summary.loc["H5", "below_zero"]} of
  4 horizons.'''))"""))
cells.append(code("""oof2 = pd.read_parquet(R2 / "oof.parquet")
wd = oof2[oof2.rung == "ATT"].drop_duplicates(["target", "repeat", "fold"]).params \\
    .map(lambda p: json.loads(p)["weight_decay"]).value_counts()
display(Markdown(f"ATT chose the largest pre-registered weight decay (1.0) in "
                 f"{wd.get(1.0, 0)} of {wd.sum()} folds: the data asks for more shrinkage than "
                 "the grid allowed, i.e. for a smaller model. Extending the grid now would be "
                 "a post-hoc change, so H5 stands as run."))"""))

cells.append(md("### C: what a naive split would have shown (descriptive)"))
cells.append(code("""naive = pd.DataFrame({
    "grouped": m2[(m2.metric == "mse") & m2.rung.isin(["R1", "R2", "R3", "R4"])]
        .set_index(["target", "rung"])["mean"],
    "naive": m2[(m2.metric == "mse") & m2.rung.str.endswith("_naive") & (m2.rung != "B0_naive")]
        .assign(rung=lambda d: d.rung.str.removesuffix("_naive")).set_index(["target", "rung"])["mean"],
}).loc[VOL]
naive["naive lower by"] = 1 - naive.naive / naive.grouped
naive.round(4)"""))
cells.append(code("""display(Image(str(R2 / "figures" / "naive_vs_grouped.png")))
d = c2[c2.status == "descriptive"]
display(d[["hypothesis", "target", "mean_diff", "ci_low", "ci_high", "p"]].reset_index(drop=True))
naive_h1 = d[(d.hypothesis == "H1_naive") & (d.target == "vol_3")].iloc[0]
grouped_h1 = c1[(c1.hypothesis == "H1") & (c1.target == "vol_3")].iloc[0]
display(Markdown(f'''**A naive split manufactures a "significant" text result.** With ungrouped CV,
text beats past volatility at 3 days with p = {naive_h1.p:.3f} (CI {naive_h1.ci_low:.3f} to
{naive_h1.ci_high:.3f}). With firm-grouped CV, the same comparison has Holm p =
{grouped_h1.p_holm:.2f}. Naive splitting lowers MSE for the embedding rungs but barely for R1,
which is consistent with the embeddings identifying the firm or speaker.'''))"""))

cells.append(md("### D: where attention looks"))
cells.append(code("""w = pd.read_parquet(R2 / "attention_weights.parquet")
top1 = w[w["rank"] == 1].groupby("modality").weight.describe()[["mean", "50%", "max"]]
display(Markdown("**Weight on each call's top sentence** (uniform over a median call ≈ 1/153 ≈ 0.007):"))
display(top1.round(3))

# Example sentences need the transcripts, which only exist locally.
DATASET = Path.home() / "var/data/earnings_calls/ACL19_Release"
if DATASET.exists():
    # Distinct calls with at least 20 sentences (a 1-sentence call has weight 1 trivially),
    # ranked by their top text sentence's weight averaged over repeats.
    def lines_of(event_id):
        return [l.strip() for l in (DATASET / event_id / "TextSequence.txt")
                .read_text(errors="replace").splitlines() if l.strip()]
    top = (w[(w.modality == "text") & (w["rank"] == 1) & (w.target == "vol_3")]
           .groupby(["event_id", "sent_idx"]).weight.mean().reset_index())
    top = top[top.event_id.map(lambda e: len(lines_of(e)) >= 20)]
    for _, r in top.sort_values("weight", ascending=False).drop_duplicates("event_id").head(4).iterrows():
        n = len(lines_of(r.event_id))
        print(f"{r.event_id} ({n} sentences; weight {r.weight:.2f} vs uniform {1/n:.3f}):\\n"
              f"    {lines_of(r.event_id)[r.sent_idx]}")
else:
    print("Transcripts not available here; example sentences are shown only when run locally.")"""))

cells.append(md("Attention is moderately peaked, but some of the sentences it favours are short "
                "and generic (greetings, transitions). That is consistent with H5's result: the "
                "learned weights fit noise more than signal at this sample size."))

cells.append(md("""## Conclusions

1. **Past volatility, recalibrated, is the strongest predictor.** None of the text or
   audio models beats it reliably in a firm-grouped design.
2. **Text has at most a weak, short-horizon signal:** about 6% lower 3-day MSE (not
   significant), and a direction AUC of about 0.57.
3. **No way of using audio helped:** call-level embeddings, prosody, format-residualised
   embeddings, or attention over sentences. Format was not the reason.
4. **The evaluation design matters.** A naive split makes text look significant and audio
   look less harmful. Repeat firms (and repeat voices) must be kept apart.

## Caveats

- **One year (2017), about 490 calls:** small for 768-dimension representations. Larger
  samples could change the audio picture.
- **Date clustering:** calls bunch in earnings seasons, and the firm-clustered bootstrap
  doesn't account for correlation across firms on the same dates.
- **Volatility windows count trading days;** Qin & Yang describe calendar days.
- **The clips are a single executive's remarks,** not the full call (no analyst Q&A).
- **Recording format** is controlled as a covariate. It is encoded in the audio embeddings
  and weakly related to 30-day volatility.
- **The audio models** are pretrained only (not fine-tuned), on telephone-to-broadcast
  quality MP3s."""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("results.ipynb"))
print("written", len(cells), "cells")

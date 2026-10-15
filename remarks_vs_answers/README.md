# remarks_vs_answers — prepared remarks vs answers to questions

Splits each call into the executive's **prepared remarks** and their **answers to
analysts' questions**, so later work can compare the information content of the two for
post-call volatility and direction. This folder holds the split and its hand audit;
modelling on the split comes next, after a pre-registered design.

## What the data contain

Each call in the ACL19 release keeps one speaker, the "most spoken executive" (usually
the CEO, sometimes the CFO or head of investor relations). Their prepared remarks and
their Q&A answers are both included. The analysts' questions, the operator and other
executives are not. Clips are named `<Speaker>_<paragraph>_<sentence>.mp3`, one
paragraph per uninterrupted turn. The earlier stages sort by these numbers but discard
the paragraph; this stage keeps it.

## Rules

Each sentence gets one of four sections, in this order within a call: `remarks`, then
`answers`, then `closing`, with `heading` lines at either boundary. The prepared remarks
end at a boundary chosen by the first rule that applies:

| Rule | When | Remarks end at | Modelling calls |
|---|---|---|---|
| `handover` | Paragraph 1 or 2 contains a handover to questions ("happy to take your questions", "open the line for questions") or the heading "Question-and-Answer Session", after the end of paragraph 1 | that sentence | 52 |
| `handback` | Paragraph 1 ends handing off to a named colleague ("turn it over to Heath") and paragraph 2 opens by thanking them ("Thanks, Heath") | end of paragraph 2 | 11 |
| `para1` | otherwise | end of paragraph 1 | 425 |

Then:

- `heading`: the line "Question-and-Answer Session", a transcript heading that has its
  own clip (131 lines).
- `closing`: a final paragraph of at most 12 sentences after Q&A that thanks listeners
  for joining or looks forward to the next call (84 calls, 435 sentences).

The rules come from reading the transcripts, not from any target. Checks behind them:
no answer phrase ("thanks for the question", "Good morning, <Name>") appears in any
paragraph 1; long later paragraphs read as long answers; and of the 45 calls showing the
handback pattern, 34 also contain a handover phrase.

Known weakness, confirmed by the audit (its one error): a `closing` paragraph can begin
with the end of the last answer (Adobe, 16 March 2017: sentences 143–148 answer the last
question, and the closing starts at 149).

## Current state

| | |
|---|---|
| Release checked against saved outputs | 572 calls, 89,722 sentences; every sentence and word count matches |
| Modelling calls | 488, each with prepared remarks |
| Sentences in the 488 (remarks / answers / closing / heading) | 34,664 / 43,364 / 435 / 131 |
| Calls with at least 1 / 10 / 20 sentences in both sections | 474 / 457 / 428 |
| Calls with no answers | 11 (one paragraph only) |
| Hand audit (50 random modelling calls) | remarks/answers boundary 50/50 correct (95% CI 0.93–1.00); answers/closing boundary 49/50 (0.90–1.00) |

Details in `data/acceptance.md`. There are 31 tests, all passing.

## Environment

Everything in this folder runs in one conda environment, `remarks_vs_answers`, defined in
`environment.yml`. Its Python packages are pinned to the versions in the earlier stages'
`uv.lock` files (Python 3.13, torch 2.14.1, transformers 5.18.0, scikit-learn 1.9.1,
pandas 3.0.6, Parselmouth 0.4.7, ...), so round 3 reuses their code under the versions
that produced rounds 1–2. It also provides `ffmpeg`, for decoding the audio clips.

```sh
conda env create -f environment.yml
conda activate remarks_vs_answers
```

## Usage

Run from this directory, with the environment active:

```sh
# labels → data/sentences.csv, data/calls.csv, data/meta.json (about 1 s)
python build_sections.py --dataset /path/to/ACL19_Release

# audit sheet → data/audit/audit_sheet.csv and .md (refuses to overwrite marks)
python make_audit.py --dataset /path/to/ACL19_Release

# acceptance report → data/acceptance.md (exits 1 if a must-pass check fails)
python check_sections.py

python -m pytest
```

`--dataset` defaults to `~/var/data/earnings_calls/ACL19_Release`. `build_sections.py`
refuses to write anything unless the release matches the saved outputs of
`audio_embeddings`, so `sent_idx` means the same sentence everywhere.

## Outputs

- `data/sentences.csv`: `event_id`, `sent_idx`, `para`, `section`, for all 572 calls.
  Joins to `text_embeddings` and `audio_embeddings` on `event_id` + `sent_idx`.
- `data/calls.csv`: per call, `in_sample` (the 488), `rule`, `boundary` (sent_idx of the
  last remarks sentence) and sentence counts per section.
- `data/meta.json`: settings, the release check and counts.
- `data/audit/`: the hand-audit sheet. Unlike the other outputs it contains transcript
  excerpts (about 50 calls × 10–20 sentences).

No other output contains transcript text: the dataset itself is not redistributed.

## Hand audit

`make_audit.py` draws a simple random sample of 50 of the 488 modelling calls (seed 0),
so the share marked correct estimates the rule's accuracy on the modelling sample.
Instructions are at the top of `data/audit/audit_sheet.md`. Read the `.md`, mark the
`.csv`:

- `boundary_remarks_answers_ok`: `Y` / `N`; `correct_boundary_remarks_answers`: the true
  last remarks sentence if `N`
- `boundary_answers_finalpara_ok`: `Y` / `N` for where `closing` starts (or that there is
  none); `correct_boundary_answers_finalpara`: the true first closing sentence, or `none`
- `notes`

`check_sections.py` then reports the shares with 95% Wilson intervals, overall and by
rule. The random sample contains few `handover` (5) and no `handback` calls.

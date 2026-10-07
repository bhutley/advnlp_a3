# Advanced Natural Language Processing - Spring 2026

## Assignment 3

### Overview


The aim is to predict abnormal stock return direction or volatility after an earnings call, using:

- ⁠Transcript text
- Audio speech embeddings
- Combined multimodal signal

### Data

- Use Qin & Yang 2017 S&P500 ECC dataset.
- ⁠Keep transcript text and audio.
- Precompute text embeddings with FinBERT.
- Precompute audio embeddings with Wav2Vec2 or HuBERT.
- ⁠Use a small subset for Colab.

### Directories

The `acl19_proc` directory contains processing scripts and data to download the daily price history for companies that we have call transcripts for.

The `event_study` directory computes cumulative abnormal returns and post-call volatility for each call from that price data. Its `data/events.csv` holds the direction and volatility targets, one row per call, keyed by the dataset folder name.

The `text_embeddings` directory holds FinBERT sentence and call embeddings plus sentiment scores for every transcript, from two models (`finbert-tone` and `ProsusAI/finbert`). They join to `event_study/data/events.csv` on the same key.

The `audio_embeddings` directory holds wav2vec2 and HuBERT embeddings (sentence level at layer 6, call level for all 13 layers) and interpretable Praat acoustic features for every sentence clip. Sentences align with `text_embeddings` on `event_id` + `sent_idx`. Two calls with broken audio segmentation are flagged by `audio_call_ok` and should be excluded from audio models. Clips come in several recording formats (`audio_embeddings/data/formats.csv`); format shapes the audio features and is weakly related to volatility, so audio models must control for it.

The `modelling` directory tests whether text and audio predict post-call volatility beyond past volatility: an ablation ladder evaluated with firm-grouped cross-validation, in two pre-registered rounds. Recalibrated past volatility dominates. Text has at most a weak 3-day signal, and none of the four ways of using audio helps. A naive, ungrouped split would have made text look significant. Start with `modelling/notebooks/results.ipynb`.

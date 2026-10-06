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

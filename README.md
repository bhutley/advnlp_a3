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

The `acl19_proc1` directory contains processing scripts and data to download the daily price history for companies that we have call transcripts for.

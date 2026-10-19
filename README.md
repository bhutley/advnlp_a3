# Advanced Natural Language Processing - Spring 2026

## Assignment 3

### Overview


The aim is to predict abnormal stock return direction or volatility after an earnings call, using:

- ⁠Transcript text
- Audio speech embeddings
- Combined multimodal signal

### Data

- Use the Qin & Yang (ACL 2019) S&P 500 earnings-call dataset (calls from 2017).
- ⁠Keep transcript text and audio.
- Precompute text embeddings with FinBERT.
- Precompute audio embeddings with Wav2Vec2 or HuBERT.
- ⁠Use a small subset for Colab.

### Directories

The `acl19_proc` directory contains processing scripts and data to download the daily price history for companies that we have call transcripts for.

The `experiments` directory holds a volatility study testing whether call audio improves forecasts beyond text and past volatility. Start with `experiments/README.md`, which summarises the changes, the experiments run and the observations; `experiments/research.md` has the design and `experiments/RESULTS.md` the full results. The dataset and generated files live in `data/` (gitignored).

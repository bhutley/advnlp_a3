# Earnings-call volatility: research and implementation

Prepared 4 October 2026. This is an assignment experiment, using modern frozen
representations on historical calls. It is not a historically deployable trading
backtest or an exact replication of a published result.

## Assessment

The idea is well supported for an Advanced NLP assignment. Focus the research
question: **Does speech information improve post-call volatility forecasts beyond
financial text and pre-call volatility?** Return direction can be a secondary
experiment. Comparing modalities and documenting evaluation choices makes a useful
contribution even if audio does not improve results.

## Peer-reviewed evidence

1. Qin and Yang (ACL 2019), [What You Say and How You Say It Matters](https://aclanthology.org/P19-1038/).
   This is the source study for your dataset. It uses aligned sentence text/audio
   from the most-spoken executive and predicts log volatility. Its multimodal model
   outperforms its unimodal alternatives. It uses dividend-adjusted CRSP prices and
   3/7/15/30 **calendar-day** horizons. Your Yahoo data and the trading-day target
   below are an adaptation, so their published MSE is not directly comparable.
   The README's attribution should be Qin & Yang **2019**; 2017 is the call year.
2. Ye, Qin and Xu (IJCAI 2020), [Financial Risk Prediction with Multi-Round Q&A Attention Network](https://www.ijcai.org/proceedings/2020/631).
   Models question-answer structure in transcripts for volatility prediction.
   Evaluates 6,494 calls from 2015–2018 with chronological training, validation and
   testing. This motivates section-aware pooling if your release retains Q&A;
   check the actual files before assuming it does.
3. Huang, Wang and Yang (Contemporary Accounting Research 2023),
   [FinBERT: A Large Language Model for Extracting Information from Financial Text](https://doi.org/10.1111/1911-3846.12832).
   Supports a financial-domain text encoder. Its sentence-level tone measure is
   associated with earnings-call market reactions. This is evidence for useful
   textual information, not a guarantee of out-of-sample direction prediction.
4. Price et al. (Journal of Banking & Finance 2012),
   [Earnings conference calls and stock returns](https://doi.org/10.1016/j.jbankfin.2011.10.013).
   Finds incremental information in call tone and Q&A for abnormal returns. Provides
   motivation for a secondary return experiment, distinct from volatility.
5. Baevski et al. (NeurIPS 2020),
   [wav2vec 2.0](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html).
   Supports self-supervised speech representations. Its reported task is speech
   recognition, not financial prediction: audio's benefit here must be tested.

## Repository findings

`acl19_proc/data/company.csv` currently contains 240 `ok` companies covering 490
calls, and 40 `no_data` companies covering 82 calls. These counts were checked from
the CSV. A company-level table is not a call-level manifest: first/last dates cannot
recover all individual call dates. The configured dataset location
`~/var/data/earnings_calls/ACL19_Release` was absent during this review.

Excluding the 82 calls causes coverage/selection bias. Report this and compare
every modality on exactly the same available calls. Never replace a delisted
company with an unrelated successor's price history. Preserve the seed file.

## Target and timing

Use log sample standard deviation of adjusted daily simple returns over seven
trading days as the main target:

`r[t] = AdjClose[t] / AdjClose[t-1] - 1`

`target = log(max(std(future_returns, ddof=1), 1e-8))`

Because your mapping has dates without call completion times, the starter anchors
at the first trading close strictly after the call date. It predicts the next seven
close-to-close returns starting after that anchor. This is a delayed post-call risk
forecast: it intentionally excludes the immediate reaction. State this convention
in the report. If reliable call-end timestamps become available, define the first
post-completion tradable anchor explicitly and regenerate labels.

Use `Adj Close` for this total-return-style proxy. Do not multiply it by `Split
Factor` for return calculation: reconstructing raw price levels would introduce
split discontinuities. Yahoo and CRSP are different data sources; corporate actions
and data revisions can differ. Examine extreme returns before modeling, and report
any exclusions instead of silently clipping test labels.

The historical control is log volatility of 20 returns ending strictly before the
call date. The existing one-month pre-call price window may not contain 20 usable
returns for some first calls. Exclusions are saved; fetch a longer pre-call window
if this removes too many calls. The starter does not fill missing prices or infer
unavailable call times.

## Model and evaluation

Freeze FinBERT and Wav2Vec2. Precompute features once, then train regularized ridge
regressions. With roughly 490 available calls, this is a defensible first model.
A small MLP or aligned-segment attention model is a later comparison, not a
prerequisite. Frozen speech representations can encode speaker identity, channel
quality and linguistic content; they do not isolate emotion or tone of voice.

Compare training-mean and past-volatility persistence baselines, learned past
volatility, text, audio, text+audio, and each modality combination with past
volatility. The main incremental comparison is **text+past versus text+audio+past**.
Also include TF-IDF+ridge as a cheap text benchmark in the final study.

The starter splits distinct call dates at 60%/80%, keeping same-day calls together.
It purges training and validation calls whose label windows cross the next split
boundary. Ridge strength and scaling use training/validation only. The model is
not refit on validation in this starter, so all baselines use the same training
observations. Date fractions are not guaranteed to produce equal call-count ratios.

Report test MSE/MAE of log volatility and Spearman correlation. The starter outputs
MSE/MAE; add Spearman and paired bootstrap confidence intervals for the final study,
resampling by company and checking date-block uncertainty because calls share market
shocks. Rolling chronological evaluations strengthen the result beyond one split.
Report a separate company-disjoint sensitivity test to detect identity memorization;
do not confuse it with forecasting future calls for known companies.

Modern FinBERT pretraining may include evaluation-period financial text: the model
card lists earnings-call transcripts among its corpora. Pin checkpoint revisions,
record the pretraining period/possible overlap, and describe this as retrospective
representation evaluation. A chronological downstream split alone does not establish
that the entire system was available in 2017.

For optional return direction, first obtain benchmark price history. One simple
label is the sign of the summed stock-minus-market daily returns over the chosen
future window (a market-adjusted CAR proxy). A market-model abnormal return target
requires estimating alpha/beta strictly before the event and a longer price history.
Use balanced accuracy, macro-F1 and ROC-AUC with majority/logistic baselines. Returns
around the call date describe market reaction and may contain outcomes that occurred
before the transcript was available; avoid describing that as tradable forecasting.

## Runnable starter

Run from the repository root. Baseline requirements: numpy and pandas. Frozen
extraction additionally requires torch, transformers, soundfile and scipy; install
those in the Colab environment. These scripts do not modify `acl19_proc` dependencies.

Create `calls.csv`, one row per actual call folder:

```csv
call_id,name,call_date
3M Company_20170425,3M Company,2017-04-25
```

Create `segments.csv`, one row per matched text/audio segment. Paths are relative
to this manifest. Use the executive segments available in your release; do not pair
executive-only text with an entire multi-speaker recording. A UTF-8 text file and a
soundfile-readable audio file are required. Inspect the release format and adapt
its JSON/text metadata to this manifest; that dataset-specific adapter is not
implemented because the dataset is not locally available.

```csv
call_id,text_path,audio_path
3M Company_20170425,segments/mmm_001.txt,segments/mmm_001.wav
3M Company_20170425,segments/mmm_002.txt,segments/mmm_002.wav
```

The extractor uses
[FinBERT-tone](https://huggingface.co/yiyanghkust/finbert-tone)'s underlying encoder
hidden states, not its sentiment classification scores. It processes all tokens in
chunks of up to 510 content tokens, excludes special tokens, and pools by token
count. This avoids discarding all but the first 512 tokens. A sentiment-score-only
baseline would be a useful additional comparison.

Audio uses [Wav2Vec2-base](https://huggingface.co/facebook/wav2vec2-base) at 16 kHz,
mono, in approximately ten-second blocks. Hidden states are mean pooled within each
block and then weighted by sample count across the call. Start with 10 calls to
verify decoding and GPU memory; use the full available set for final experiments.
This simple pooling loses sequence/alignment information. It is a baseline, not
a reproduction of MDRM or MR-QA. Extraction downloads the model checkpoints and
currently writes the cache at completion; split manifests into batches if needed.

```sh
python experiments/volatility_baseline.py labels --calls calls.csv --out labels.csv
python experiments/embed_calls.py --segments segments.csv --out features.npz
python experiments/volatility_baseline.py train --labels labels.csv --features features.npz
```

For a first price-only experiment, omit `--features` from the training command.
`features.npz` stores string `call_id` values and 2D `text` and `audio` matrices.
Labels and embeddings are joined by ID rather than assumed to be in matching order.
Missing embeddings fail explicitly instead of silently changing the cohort.

Offline verification:

```sh
python -m unittest discover -s experiments -p "test_*.py"
```

Only the numerical baseline, label timing and split logic have local execution
checks. Real text/audio extraction and predictive performance require the dataset,
model dependencies and checkpoint downloads. No financial performance is claimed.

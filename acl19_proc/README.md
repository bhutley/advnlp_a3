# acl19_proc — price history for the ACL19 earnings-call dataset

Resolves the companies in the ACL19 earnings-call audio dataset to stock tickers and
downloads their daily price history from Yahoo Finance, from one month before each
company's first call to one year after its last.

The audio dataset itself lives outside this repository, by default at
`~/var/data/earnings_calls/ACL19_Release`. It holds one folder per call, named
`<Company Name>_<YYYYMMDD>`, e.g. `3M Company_20170425`. The current data covers
**572 calls from 280 companies**, dated 2017-01-17 to 2017-12-21.

## Current state

| | |
|---|---|
| Companies with price files | 240 (490 of the 572 calls) |
| Companies with no price data | 40 (82 calls), all delisted |
| Tests | 71, all passing |

## Requirements

Python 3.13 or newer, and [uv](https://docs.astral.sh/uv/). `uv sync` installs
pandas, yfinance and pytest.

## Usage

```sh
uv sync

# Step 1: dataset + seed → data/company.csv (about 10 minutes, one Yahoo request per company)
uv run python resolve_tickers.py

# Review data/company.csv, then fix tickers_seed.csv and re-run if needed.

# Step 2: company.csv → data/<TICKER>.csv (about 10 minutes)
uv run python fetch_prices.py

uv run pytest
```

Both scripts accept `--tickers MMM,META,AET` to limit a run to a few companies, which
is useful for a quick trial. `resolve_tickers.py` also takes `--dataset`, `--seed` and
`--out`; `fetch_prices.py` takes `--company-csv`, `--data-dir` and `--force`, which
overwrites existing price files instead of skipping them.

Both scripts wait 1 second between Yahoo requests, retry a failed request 3 times
(waiting 2s then 4s), and print per-company progress plus a summary. `fetch_prices.py`
skips tickers that already have a file, so an interrupted run can simply be repeated.

## Files

| Path | Description |
|---|---|
| `tickers_seed.csv` | **Hand-maintained input.** The only file to edit by hand. |
| `common.py` | Date windows, retries, and the two Yahoo Finance calls. |
| `resolve_tickers.py` | Step 1: scans the dataset, checks the seed against Yahoo, writes `data/company.csv`. |
| `fetch_prices.py` | Step 2: downloads one price file per `ok` company. |
| `tests/` | pytest suite. No network access; Yahoo calls are replaced by fakes. |
| `data/company.csv` | Generated: the name-to-ticker mapping with a status per company. |
| `data/<TICKER>.csv` | Generated: daily price history, one file per company. |
| `docs/plans/2026-09-17-earnings-call-prices-design.md` | The design, including decisions and the Yahoo quirks found along the way. |
| `docs/plans/2026-09-17-earnings-call-prices-plan.md` | The implementation plan, in 8 tasks. |
| `pyproject.toml`, `uv.lock` | Dependencies and pinned versions. |

Everything under `data/` can be regenerated. `tickers_seed.csv` cannot, so it is the
file worth protecting.

### `tickers_seed.csv`

| Column | Description |
|---|---|
| `name` | The company name, exactly as the dataset folder spells it. |
| `ticker_2017` | The ticker in use at the time of the call, e.g. `FB`, `COH`, `HCN`. |
| `yahoo_ticker` | Today's ticker on Yahoo, e.g. `META`, `TPR`, `WELL`. Blank when the company is delisted and Yahoo has no history for it. |
| `name_confirmed` | `yes` or blank. `yes` records that a name mismatch has been reviewed and accepted, which is usual after a rename. |

### `data/company.csv`

Columns: `name, ticker_2017, yahoo_ticker, yahoo_name, status, n_calls, first_call,
last_call`. `yahoo_name` is Yahoo's own name for the ticker, kept so mismatches can be
checked by eye.

| Status | Meaning |
|---|---|
| `ok` | Yahoo has data for the window. A price file was written. |
| `no_data` | The seed has no Yahoo ticker, or Yahoo returned nothing. |
| `check_name` | Data exists, but Yahoo's name doesn't match and hasn't been confirmed. Needs review. |
| `error` | The request still failed after 3 attempts. Re-running rechecks these rows. |

A network failure is never recorded as `no_data`, so a delisting and a dropped
connection can't be confused.

### `data/<TICKER>.csv`

One row per trading day, named after the **current** Yahoo ticker, so Facebook's file
is `META.csv`. Columns: `Date, Open, High, Low, Close, Adj Close, Volume, Dividends,
Stock Splits, Split Factor`. Yahoo's values are written unchanged and unrounded.

The window runs from one month before the company's first call to one year after its
last, so a company with several calls gets a single continuous file.

## Prices are not as-traded — use `Split Factor`

Yahoo adjusts `Open`, `High`, `Low`, `Close`, `Dividends` **and `Volume`** for every
split after a given date, including spin-offs, which it records as fractional splits.
`Adj Close` is additionally adjusted for dividends.

`Split Factor` undoes that. It is the product of the split ratios that fall after each
row:

- as-traded price = price × `Split Factor`
- as-traded dividend = dividend × `Split Factor`
- as-traded volume = volume ÷ `Split Factor`

Worked examples:

| Ticker | Date | `Close` | Factor | As-traded | Why |
|---|---|---|---|---|---|
| MMM | 2017-04-25 | 163.15 | 1.196 | 195.13 | Solventum spin-off, 2024 |
| AMZN | 2017-10-27 | 55.05 | 20 | 1100.95 | 20:1 split, 2022 |
| NVDA | 2017-05-10 | 3.03 | 40 | 121.29 | 4:1 in 2021 and 10:1 in 2024 |
| XRX | 2017-05-02 | 28.16 | 0.25 | 7.04 | 1-for-4 reverse split, 2018 |

Two consequences:

- **The as-traded values are the stable ones.** Re-download after a future split and
  every `Close` shifts, while `Close × Split Factor` stays put.
- **Spin-off ratios are Yahoo's rounded figures.** NKTR's 1-for-15 reverse split is
  stored as 0.0667, so its as-traded prices are about 0.05% off. Plain splits are exact.

## Editing the seed

Re-run `resolve_tickers.py` after any change, and then `fetch_prices.py`.

- **A company was renamed** (e.g. Marsh & McLennan changed its ticker from `MMC` to
  `MRSH`): put the new ticker in `yahoo_ticker`. The row will come back as
  `check_name`; once the `yahoo_name` looks right, set `name_confirmed` to `yes`.
- **A company was acquired, delisted or taken private:** blank the `yahoo_ticker`. The
  row becomes `no_data` and no request is made.
- **A `check_name` row you've reviewed:** set `name_confirmed` to `yes`.
- **An `error` row:** usually a network problem. Just re-run the script.

When a status looks wrong, check whether Yahoo has a new ticker for the company before
assuming it's gone. `MMC` → `MRSH` and `BK` → `BNY` both looked like delistings at
first.

Verifying a rename is worth the effort, because Yahoo sometimes keeps history under a
successor that belongs to a *different* company. Two examples from this dataset:

- Ingersoll-Rand's history lives under `TT` (Trane Technologies). Today's `IR` is the
  former Gardner Denver.
- Raytheon Co. (`RTN`) has no history at all, while `RTX` holds United Technologies'.

The quickest check is to compare an as-traded 2017 price against what the original
company traded at then.

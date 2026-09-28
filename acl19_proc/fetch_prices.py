"""Download daily price history for every 'ok' company in data/company.csv.

Yahoo adjusts Open/High/Low/Close, Dividends and Volume for every later split, including
spin-offs it records as fractional splits. Each row's 'Split Factor' undoes that:
as-traded price = price * factor, as-traded volume = volume / factor.
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import common
from common import TICKER_RE, compute_window

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
OUTPUT_COLUMNS = PRICE_COLUMNS + ["Split Factor"]
REQUEST_DELAY = 1.0  # seconds between Yahoo requests


def split_factors(splits: pd.Series) -> pd.Series:
    """For each row, the product of split ratios on later rows (1.0 if none).

    A split's own row is already post-split, so its ratio applies only to earlier rows.
    """
    ratios = splits.replace(0.0, 1.0)
    return ratios.iloc[::-1].cumprod().iloc[::-1].shift(-1, fill_value=1.0)


def to_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Price history → Date (YYYY-MM-DD) followed by OUTPUT_COLUMNS, values unchanged."""
    out = df[OUTPUT_COLUMNS].reset_index(drop=True)
    out.insert(0, "Date", df.index.strftime("%Y-%m-%d"))
    return out


def coverage_gap(df: pd.DataFrame, start: date, end: date, tolerance_days: int = 7) -> bool:
    """True if the data starts or ends more than tolerance_days inside [start, end)."""
    tolerance = timedelta(days=tolerance_days)
    first, last = df.index.min().date(), df.index.max().date()
    return first - start > tolerance or (end - timedelta(days=1)) - last > tolerance


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-csv", type=Path, default=Path("data/company.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--tickers", help="comma-separated tickers (2017 or Yahoo) to limit the run to"
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing price files")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    companies = pd.read_csv(args.company_csv, dtype=object, keep_default_na=False)
    companies = companies[companies["status"] == "ok"]
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        mask = companies["ticker_2017"].isin(wanted) | companies["yahoo_ticker"].isin(wanted)
        companies = companies[mask]
        if companies.empty:
            print(f"error: no ok companies match --tickers {args.tickers}", file=sys.stderr)
            return 1

    args.data_dir.mkdir(parents=True, exist_ok=True)
    results = {"written": [], "skipped": [], "failed": [], "partial": []}
    hit_yahoo = False
    for i, c in enumerate(companies.to_dict("records"), start=1):
        ticker = c["yahoo_ticker"]
        prefix = f"[{i}/{len(companies)}] {ticker}:"
        if not TICKER_RE.match(ticker):
            print(f"{prefix} failed (invalid ticker)")
            results["failed"].append(ticker)
            continue
        path = args.data_dir / f"{ticker}.csv"
        if path.exists() and not args.force:
            print(f"{prefix} skipped (exists)")
            results["skipped"].append(ticker)
            continue

        if hit_yahoo:
            time.sleep(REQUEST_DELAY)
        hit_yahoo = True
        start, end = compute_window(
            date.fromisoformat(c["first_call"]), date.fromisoformat(c["last_call"])
        )
        try:
            # Fetch through today so splits after the window are seen.
            full = common.fetch_history(ticker, start, date.today() + timedelta(days=1))
        except Exception as e:
            print(f"{prefix} failed ({type(e).__name__}: {e})")
            results["failed"].append(ticker)
            continue
        if full.empty:
            print(f"{prefix} failed (no data returned)")
            results["failed"].append(ticker)
            continue
        full["Split Factor"] = split_factors(full["Stock Splits"])
        prices = full[full.index.tz_localize(None) < pd.Timestamp(end)]
        if prices.empty:
            print(f"{prefix} failed (no data in window)")
            results["failed"].append(ticker)
            continue

        to_output_frame(prices).to_csv(path, index=False)
        results["written"].append(ticker)
        note = ""
        if coverage_gap(prices, start, end):
            results["partial"].append(ticker)
            first, last = prices.index.min().date(), prices.index.max().date()
            note = f" (partial: {first} to {last}, wanted {start} to {end - timedelta(days=1)})"
        print(f"{prefix} written, {len(prices)} rows{note}")

    print()
    for key, tickers in results.items():
        print(f"{key}: {len(tickers)}")
        if key in ("failed", "partial"):
            for t in sorted(tickers):
                print(f"  {t}")
    return 1 if results["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

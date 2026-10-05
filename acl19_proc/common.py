"""Shared helpers: date windows, retries and Yahoo Finance access."""

import re
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFPricesMissingError, YFTzMissingError

TICKER_RE = re.compile(r"^[A-Z0-9.\-]+$")

# Errors yfinance raises when Yahoo has no data for a symbol (e.g. delisted).
NO_DATA_ERRORS = (YFPricesMissingError, YFTzMissingError)


def compute_window(
    first_call: date, last_call: date, lookback_months: int = 12
) -> tuple[date, date]:
    """Return (start, end) covering lookback_months before first_call to 1 year after last_call.

    The 12-month default leaves room for an event-study estimation window before each call.
    end is one day past the last wanted day, because yfinance treats end as exclusive.
    """
    start = (pd.Timestamp(first_call) - pd.DateOffset(months=lookback_months)).date()
    end = (pd.Timestamp(last_call) + pd.DateOffset(years=1)).date() + timedelta(days=1)
    return start, end


def with_retries(fn, attempts=3, base_delay=2.0, sleep=None):
    """Call fn(), retrying on any exception with exponential backoff; re-raise after the last attempt."""
    sleep = sleep or time.sleep
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                raise
            sleep(base_delay * 2**i)


def fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Daily unadjusted history with dividends and splits; empty frame if Yahoo has no data."""

    def fetch():
        try:
            return yf.Ticker(ticker).history(
                start=start, end=end, auto_adjust=False, actions=True, raise_errors=True
            )
        except NO_DATA_ERRORS:
            return pd.DataFrame()

    return with_retries(fetch)


def fetch_name(ticker: str) -> str | None:
    """Yahoo's long company name for ticker, or None if it has none."""
    return with_retries(lambda: yf.Ticker(ticker).info.get("longName"))

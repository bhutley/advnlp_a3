"""Pure event-study calculations: day 0, market model, abnormal returns, CARs, volatility.

Relative days count trading days on the market (SPY) calendar, with day 0 the first day
whose return can reflect the call. A stock's prices are aligned to that calendar before
returns are computed, so a missing day gives NaN rather than a return spanning two days.
"""

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

EST_WINDOW = (-150, -30)
EVENT_WINDOW = (-5, 60)
CAR_WINDOWS = {"m5_m1": (-5, -1), "0_1": (0, 1), "0_5": (0, 5), "0_20": (0, 20), "0_60": (0, 60)}
VOL_HORIZONS = (3, 7, 15, 30)
MIN_EST_OBS = 100


def load_prices(path: Path) -> pd.Series:
    """Adj Close from an acl19_proc price file, indexed by date."""
    df = pd.read_csv(path, usecols=["Date", "Adj Close"], parse_dates=["Date"])
    return df.set_index("Date")["Adj Close"]


def day0_index(dates: pd.DatetimeIndex, call_date: date, timing: str) -> int | None:
    """Position of day 0 in the sorted trading dates, or None if it lies beyond them.

    'pre' and 'unknown': the first trading day on or after the call date.
    'post': the first trading day strictly after it.
    """
    if timing not in ("pre", "post", "unknown"):
        raise ValueError(f"unknown timing: {timing!r}")
    side = "right" if timing == "post" else "left"
    pos = int(dates.searchsorted(pd.Timestamp(call_date), side=side))
    return pos if pos < len(dates) else None


def relative_frame(
    stock: pd.Series, market: pd.Series, call_date: date, timing: str
) -> pd.DataFrame | None:
    """Daily returns indexed by relative trading day (rel_day; 0 = day 0).

    Columns: date, r_i (stock), r_m (market). None if day 0 lies beyond the market data.
    """
    stock = stock.reindex(market.index)
    d0 = day0_index(market.index, call_date, timing)
    if d0 is None:
        return None
    rel = pd.DataFrame(
        {
            "date": market.index,
            "r_i": (stock / stock.shift(1) - 1).to_numpy(),
            "r_m": (market / market.shift(1) - 1).to_numpy(),
        },
        index=pd.RangeIndex(-d0, len(market) - d0, name="rel_day"),
    )
    return rel


def fit_market_model(est: pd.DataFrame) -> tuple[float, float, float, int]:
    """OLS of r_i on r_m over the rows with both returns → (alpha, beta, sigma, n).

    sigma is the residual standard deviation with n − 2 degrees of freedom.
    All three are NaN if n < 3.
    """
    est = est[["r_i", "r_m"]].dropna()
    n = len(est)
    if n < 3:
        return math.nan, math.nan, math.nan, n
    x, y = est["r_m"].to_numpy(), est["r_i"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    sigma = math.sqrt((resid @ resid) / (n - 2))
    return float(alpha), float(beta), sigma, n


def abnormal_returns(rel: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    """ar_mm (market model) and ar_ma (market-adjusted) for every day in EVENT_WINDOW.

    Days missing from rel are NaN. A NaN alpha or beta gives NaN ar_mm.
    """
    lo, hi = EVENT_WINDOW
    window = rel.reindex(pd.RangeIndex(lo, hi + 1, name="rel_day"))
    return pd.DataFrame(
        {
            "ar_mm": window["r_i"] - (alpha + beta * window["r_m"]),
            "ar_ma": window["r_i"] - window["r_m"],
        }
    )


def car(ar: pd.Series, window: tuple[int, int]) -> float:
    """Sum of ar over relative days [start, end]; NaN if any day is missing."""
    lo, hi = window
    values = ar.reindex(range(lo, hi + 1))
    return math.nan if values.isna().any() else float(values.sum())


def log_vol(r: pd.Series, start: int, n: int) -> float:
    """Log return volatility over relative days [start, start + n], as in Qin & Yang (2019).

    Their equation (1), with t = start + n:
        v[t−n, t] = ln( sqrt( sum_{i=0..n} (r_{t−i} − r̄)² / n ) )
    i.e. the sample standard deviation (ddof=1) of n + 1 daily returns. The paper calls
    n "calendar days", but its formula sums daily returns, so n counts trading days here.
    NaN if any return in the window is missing.
    """
    values = r.reindex(range(start, start + n + 1))
    if values.isna().any():
        return math.nan
    return math.log(values.std(ddof=1))

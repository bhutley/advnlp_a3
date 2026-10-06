import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from returns import (
    EVENT_WINDOW,
    abnormal_returns,
    car,
    day0_index,
    fit_market_model,
    load_prices,
    log_vol,
    relative_frame,
)

# Mon 2017-04-24 .. Mon 2017-05-01, no weekend rows.
WEEK = pd.DatetimeIndex(
    ["2017-04-24", "2017-04-25", "2017-04-26", "2017-04-27", "2017-04-28", "2017-05-01"]
)
# 2017-07-04 (Tuesday) was a market holiday.
JULY = pd.DatetimeIndex(["2017-06-30", "2017-07-03", "2017-07-05", "2017-07-06"])


# --- load_prices ---


def test_load_prices(tmp_path):
    path = tmp_path / "X.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Adj Close,Volume,Dividends,Stock Splits,Split Factor\n"
        "2017-04-24,1,1,1,10.0,9.0,100,0,0,1\n"
        "2017-04-25,1,1,1,11.0,9.9,100,0,0,1\n"
    )
    px = load_prices(path)
    assert list(px.index) == [pd.Timestamp("2017-04-24"), pd.Timestamp("2017-04-25")]
    assert px.tolist() == [9.0, 9.9]  # Adj Close, not Close


# --- day0_index ---


@pytest.mark.parametrize(
    "call, timing, expected",
    [
        (date(2017, 4, 25), "pre", 1),
        (date(2017, 4, 25), "unknown", 1),
        (date(2017, 4, 25), "post", 2),
        (date(2017, 4, 29), "pre", 5),  # Saturday → Monday
        (date(2017, 4, 29), "post", 5),
        (date(2017, 5, 1), "post", None),  # next trading day is beyond the data
        (date(2017, 5, 2), "pre", None),
    ],
)
def test_day0_index(call, timing, expected):
    assert day0_index(WEEK, call, timing) == expected


@pytest.mark.parametrize(
    "call, timing, expected",
    [
        (date(2017, 7, 3), "post", 2),  # after-close call before a holiday → 07-05
        (date(2017, 7, 4), "pre", 2),  # call on the holiday → 07-05
        (date(2017, 7, 4), "post", 2),
    ],
)
def test_day0_index_holiday(call, timing, expected):
    assert day0_index(JULY, call, timing) == expected


def test_day0_index_rejects_bad_timing():
    with pytest.raises(ValueError):
        day0_index(WEEK, date(2017, 4, 25), "noon")


# --- relative_frame ---


def test_relative_frame_aligns_to_market_calendar():
    market = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=WEEK)
    # Stock halted on 04-26: no row that day.
    stock = pd.Series([10.0, 11.0, 12.1, 12.1, 13.31], index=WEEK.delete(2))

    rel = relative_frame(stock, market, date(2017, 4, 26), "pre")

    assert list(rel.index) == [-2, -1, 0, 1, 2, 3]
    assert rel.index.name == "rel_day"
    assert rel.loc[0, "date"] == pd.Timestamp("2017-04-26")
    assert rel.loc[-1, "r_i"] == pytest.approx(0.1)
    # The halt day and the day after it have no stock return; nothing spans two days.
    assert math.isnan(rel.loc[0, "r_i"]) and math.isnan(rel.loc[1, "r_i"])
    assert rel.loc[2, "r_i"] == pytest.approx(0.0)
    assert rel.loc[0, "r_m"] == pytest.approx(102 / 101 - 1)
    assert math.isnan(rel.loc[-2, "r_i"])  # first day has no previous price


def test_relative_frame_day0_beyond_data():
    market = pd.Series(np.arange(6.0) + 100, index=WEEK)
    assert relative_frame(market, market, date(2017, 5, 2), "pre") is None


# --- fit_market_model ---


def make_estimation(n=500, seed=0):
    rng = np.random.default_rng(seed)
    r_m = rng.normal(0, 0.01, n)
    r_i = 0.001 + 1.3 * r_m + rng.normal(0, 0.005, n)
    return pd.DataFrame({"r_i": r_i, "r_m": r_m})


def test_fit_market_model_recovers_parameters():
    alpha, beta, sigma, n = fit_market_model(make_estimation())
    assert alpha == pytest.approx(0.001, abs=5e-4)
    assert beta == pytest.approx(1.3, abs=0.05)
    assert sigma == pytest.approx(0.005, rel=0.1)
    assert n == 500


def test_fit_market_model_drops_nan_rows():
    est = make_estimation()
    est.loc[:9, "r_i"] = np.nan
    *_, n = fit_market_model(est)
    assert n == 490


def test_fit_market_model_too_few_rows():
    alpha, beta, sigma, n = fit_market_model(make_estimation(n=2))
    assert n == 2 and all(math.isnan(x) for x in (alpha, beta, sigma))


# --- abnormal_returns ---


def test_abnormal_returns_covers_event_window():
    rel = pd.DataFrame({"r_i": [0.02, 0.03], "r_m": [0.01, 0.01]}, index=[0, 1])
    ar = abnormal_returns(rel, alpha=0.0, beta=1.0)

    lo, hi = EVENT_WINDOW
    assert list(ar.index) == list(range(lo, hi + 1))
    assert ar.loc[0, "ar_mm"] == pytest.approx(0.01)
    assert ar.loc[1, "ar_mm"] == pytest.approx(0.02)
    assert ar["ar_mm"].equals(ar["ar_ma"])  # alpha=0, beta=1
    assert ar.loc[-5:-1].isna().all().all()


def test_abnormal_returns_market_model():
    rel = pd.DataFrame({"r_i": [0.05], "r_m": [0.02]}, index=[0])
    ar = abnormal_returns(rel, alpha=0.001, beta=1.5)
    assert ar.loc[0, "ar_mm"] == pytest.approx(0.05 - (0.001 + 1.5 * 0.02))
    assert ar.loc[0, "ar_ma"] == pytest.approx(0.03)


def test_abnormal_returns_without_model():
    rel = pd.DataFrame({"r_i": [0.05], "r_m": [0.02]}, index=[0])
    ar = abnormal_returns(rel, alpha=math.nan, beta=math.nan)
    assert math.isnan(ar.loc[0, "ar_mm"])
    assert ar.loc[0, "ar_ma"] == pytest.approx(0.03)


# --- car ---


def test_car_sums_window():
    ar = pd.Series([0.01, 0.02, -0.005, 0.03], index=[-1, 0, 1, 2])
    assert car(ar, (0, 1)) == pytest.approx(0.015)
    assert car(ar, (-1, 2)) == pytest.approx(0.055)


def test_car_nan_if_day_missing():
    ar = pd.Series([0.01, np.nan, 0.03], index=[0, 1, 2])
    assert math.isnan(car(ar, (0, 2)))


def test_car_nan_if_window_beyond_data():
    ar = pd.Series([0.01, 0.02], index=[0, 1])
    assert math.isnan(car(ar, (0, 5)))


# --- log_vol ---


def test_log_vol_uses_n_plus_one_returns():
    # Qin & Yang (2019): days 0..n inclusive, divisor n (sample std).
    r = pd.Series([0.5, 0.01, -0.02, 0.03, 0.0, 0.5], index=[-1, 0, 1, 2, 3, 4])
    window = [0.01, -0.02, 0.03, 0.0]
    mean = sum(window) / 4
    expected = math.log(math.sqrt(sum((x - mean) ** 2 for x in window) / 3))
    assert log_vol(r, start=0, n=3) == pytest.approx(expected)


def test_log_vol_nan_if_incomplete():
    r = pd.Series([0.01, np.nan, 0.03, 0.0], index=[0, 1, 2, 3])
    assert math.isnan(log_vol(r, start=0, n=3))
    assert math.isnan(log_vol(r, start=0, n=7))

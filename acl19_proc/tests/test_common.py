from datetime import date

import pytest
from yfinance.exceptions import YFPricesMissingError

import common
from common import TICKER_RE, compute_window, fetch_history, with_retries


def test_compute_window_single_call():
    assert compute_window(date(2017, 4, 25), date(2017, 4, 25)) == (
        date(2017, 3, 25),
        date(2018, 4, 26),
    )


def test_compute_window_union_of_calls():
    assert compute_window(date(2017, 4, 27), date(2017, 10, 27)) == (
        date(2017, 3, 27),
        date(2018, 10, 28),
    )


def test_compute_window_month_end():
    start, _ = compute_window(date(2017, 3, 31), date(2017, 3, 31))
    assert start == date(2017, 2, 28)


def test_with_retries_succeeds_on_third_attempt():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("boom")
        return "ok"

    assert with_retries(flaky, sleep=sleeps.append) == "ok"
    assert len(calls) == 3
    assert sleeps == [2, 4]


def test_with_retries_reraises_after_last_attempt():
    sleeps = []

    def always_fails():
        raise ConnectionError("boom")

    with pytest.raises(ConnectionError):
        with_retries(always_fails, sleep=sleeps.append)
    assert sleeps == [2, 4]


class FakeTicker:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def history(self, **kwargs):
        self.calls += 1
        raise self.exc


def test_fetch_history_missing_prices_returns_empty_without_retry(monkeypatch):
    fake = FakeTicker(YFPricesMissingError("AET", "possibly delisted"))
    monkeypatch.setattr(common.yf, "Ticker", lambda t: fake)
    monkeypatch.setattr(common.time, "sleep", lambda s: None)

    df = fetch_history("AET", date(2017, 3, 25), date(2018, 4, 26))

    assert df.empty
    assert fake.calls == 1


def test_fetch_history_network_error_retries_then_raises(monkeypatch):
    fake = FakeTicker(ConnectionError("boom"))
    monkeypatch.setattr(common.yf, "Ticker", lambda t: fake)
    monkeypatch.setattr(common.time, "sleep", lambda s: None)

    with pytest.raises(ConnectionError):
        fetch_history("MMM", date(2017, 3, 25), date(2018, 4, 26))
    assert fake.calls == 3


@pytest.mark.parametrize("ticker", ["MMM", "BRK-B", "BF.B"])
def test_ticker_re_accepts(ticker):
    assert TICKER_RE.match(ticker)


@pytest.mark.parametrize("ticker", ["../x", "meta", "", "A/B"])
def test_ticker_re_rejects(ticker):
    assert not TICKER_RE.match(ticker)

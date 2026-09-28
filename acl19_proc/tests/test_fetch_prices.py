from datetime import date, timedelta

import pandas as pd
import pytest

import common
import fetch_prices
from fetch_prices import OUTPUT_COLUMNS, coverage_gap, main, split_factors, to_output_frame


def make_prices(days, splits=None):
    index = pd.DatetimeIndex(
        [pd.Timestamp(d, tz="America/New_York") for d in days], name="Date"
    )
    n = len(days)
    splits = splits or [0.0] * n
    return pd.DataFrame(
        {
            "Volume": [100] * n,
            "Open": [1.5] * n,
            "High": [2.25] * n,
            "Low": [1.0] * n,
            "Close": [2.0] * n,
            "Adj Close": [1.875] * n,
            "Dividends": [0.0] * n,
            "Stock Splits": splits,
            "Capital Gains": [0.0] * n,
        },
        index=index,
    )


# --- to_output_frame ---


def test_to_output_frame():
    prices = make_prices([date(2017, 4, 25), date(2017, 4, 26)])
    prices["Split Factor"] = [2.0, 1.0]

    out = to_output_frame(prices)

    assert list(out.columns) == ["Date"] + OUTPUT_COLUMNS
    assert out["Date"].tolist() == ["2017-04-25", "2017-04-26"]
    assert out.iloc[0].tolist() == ["2017-04-25", 1.5, 2.25, 1.0, 2.0, 1.875, 100, 0.0, 0.0, 2.0]


def test_to_output_frame_missing_column():
    with pytest.raises(KeyError):
        to_output_frame(make_prices([date(2017, 4, 25)]).drop(columns="Adj Close"))


# --- split_factors ---


def test_split_factors_multiplies_later_splits_only():
    days = [date(2017, 1, d) for d in (2, 3, 4, 5, 6)]
    prices = make_prices(days, splits=[0.0, 0.0, 10.0, 0.0, 2.0])

    assert split_factors(prices["Stock Splits"]).tolist() == [20.0, 20.0, 2.0, 2.0, 1.0]


def test_split_factors_no_splits():
    prices = make_prices([date(2017, 1, 2), date(2017, 1, 3)])
    assert split_factors(prices["Stock Splits"]).tolist() == [1.0, 1.0]


# --- coverage_gap ---

START, END = date(2017, 3, 25), date(2018, 4, 26)  # END is exclusive


def test_coverage_gap_full():
    assert not coverage_gap(make_prices([date(2017, 3, 27), date(2018, 4, 25)]), START, END)


def test_coverage_gap_starts_late():
    assert coverage_gap(make_prices([date(2017, 4, 4), date(2018, 4, 25)]), START, END)


def test_coverage_gap_ends_early():
    assert coverage_gap(make_prices([date(2017, 3, 27), date(2018, 4, 15)]), START, END)


# --- main ---

# Wanted [start, end) windows for the companies in the fixture's company.csv.
WINDOWS = {
    "MMM": (date(2017, 3, 25), date(2018, 7, 26)),
    "META": (date(2017, 10, 1), date(2018, 11, 2)),
}
LATER_SPLIT_DAY = date(2020, 6, 1)


def full_window(ticker, start, end):
    """Rows on the first and last wanted day, plus a 2:1 split after the window."""
    w_start, w_end = WINDOWS[ticker]
    return make_prices(
        [w_start, w_end - timedelta(days=1), LATER_SPLIT_DAY], splits=[0.0, 0.0, 2.0]
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    company_csv = tmp_path / "company.csv"
    company_csv.write_text(
        "name,ticker_2017,yahoo_ticker,yahoo_name,status,n_calls,first_call,last_call\n"
        "3M Company,MMM,MMM,3M Company,ok,2,2017-04-25,2017-07-25\n"
        "Aetna Inc,AET,,,no_data,3,2017-01-31,2017-10-31\n"
        '"Facebook, Inc.",FB,META,"Meta Platforms, Inc.",ok,1,2017-11-01,2017-11-01\n'
    )
    data_dir = tmp_path / "data"
    calls = []

    def fake_fetch(ticker, start, end):
        calls.append((ticker, start, end))
        return full_window(ticker, start, end)

    monkeypatch.setattr(common, "fetch_history", fake_fetch)
    monkeypatch.setattr(fetch_prices.time, "sleep", lambda s: None)
    argv = ["--company-csv", str(company_csv), "--data-dir", str(data_dir)]
    return argv, data_dir, calls


def test_main_writes_ok_tickers(project):
    argv, data_dir, calls = project

    assert main(argv) == 0
    assert sorted(p.name for p in data_dir.iterdir()) == ["META.csv", "MMM.csv"]
    mmm_call = next(c for c in calls if c[0] == "MMM")
    assert mmm_call[1] == date(2017, 3, 25)
    assert mmm_call[2] > date.today()  # fetches to today to see later splits
    mmm = pd.read_csv(data_dir / "MMM.csv")
    assert list(mmm.columns) == ["Date"] + OUTPUT_COLUMNS
    assert mmm["Date"].tolist() == ["2017-03-25", "2018-07-25"]
    assert mmm["Split Factor"].tolist() == [2.0, 2.0]


def test_main_skips_existing_file(project):
    argv, data_dir, calls = project
    data_dir.mkdir()
    (data_dir / "MMM.csv").write_text("old")

    assert main(argv) == 0
    assert (data_dir / "MMM.csv").read_text() == "old"
    assert [c[0] for c in calls] == ["META"]


def test_main_force_overwrites(project):
    argv, data_dir, calls = project
    data_dir.mkdir()
    (data_dir / "MMM.csv").write_text("old")

    assert main(argv + ["--force"]) == 0
    assert (data_dir / "MMM.csv").read_text().startswith("Date,")


def test_main_tickers_filter(project):
    argv, data_dir, _ = project

    assert main(argv + ["--tickers", "fb"]) == 0
    assert [p.name for p in data_dir.iterdir()] == ["META.csv"]


def test_main_failed_fetch(project, monkeypatch, capsys):
    argv, data_dir, _ = project

    def fetch(ticker, start, end):
        if ticker == "MMM":
            raise ConnectionError("boom")
        return pd.DataFrame() if ticker == "META" else full_window(ticker, start, end)

    monkeypatch.setattr(common, "fetch_history", fetch)

    assert main(argv) == 1
    assert not data_dir.exists() or not any(data_dir.iterdir())
    assert "failed: 2\n  META\n  MMM\n" in capsys.readouterr().out


def test_main_partial_is_written_and_reported(project, monkeypatch, capsys):
    argv, data_dir, _ = project
    monkeypatch.setattr(
        common,
        "fetch_history",
        lambda t, start, end: make_prices(
            [WINDOWS[t][0] + timedelta(days=10), WINDOWS[t][1] - timedelta(days=1)]
        ),
    )

    assert main(argv + ["--tickers", "MMM"]) == 0
    assert (data_dir / "MMM.csv").exists()
    assert "partial: 1\n  MMM\n" in capsys.readouterr().out


def test_main_no_rows_in_window_is_failed(project, monkeypatch, capsys):
    argv, data_dir, _ = project
    monkeypatch.setattr(
        common, "fetch_history", lambda t, start, end: make_prices([LATER_SPLIT_DAY])
    )

    assert main(argv + ["--tickers", "MMM"]) == 1
    assert not (data_dir / "MMM.csv").exists()
    assert "failed: 1\n  MMM\n" in capsys.readouterr().out

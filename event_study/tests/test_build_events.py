import math

import numpy as np
import pandas as pd
import pytest

from build_events import EVENT_COLUMNS, large_returns, main

# Market calendar for the synthetic data: business days, long enough for [-150, +60].
DATES = pd.bdate_range("2016-06-01", "2017-12-29")
JUMP = 0.10  # planted abnormal return on day 0
COMPANY_HEADER = "name,ticker_2017,yahoo_ticker,yahoo_name,status,n_calls,first_call,last_call\n"


def write_prices(path, dates, returns):
    prices = 100 * np.cumprod(1 + np.asarray(returns))
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Adj Close": prices}).to_csv(
        path, index=False
    )


def make_call(root, folder, text):
    d = root / folder
    d.mkdir(parents=True)
    (d / "TextSequence.txt").write_text(text)


@pytest.fixture
def project(tmp_path):
    """Dataset, company.csv and price files for four calls.

    Alpha (pre) and Beta (post) both jump by JUMP on their day 0. Gamma has no prices.
    Delta's price file starts only 60 trading days before its call.
    """
    rng = np.random.default_rng(0)
    r_m = rng.normal(0.0003, 0.008, len(DATES))
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    write_prices(prices_dir / "SPY.csv", DATES, r_m)

    def stock_returns(jump_date=None):
        r = 0.0002 + 1.2 * r_m + rng.normal(0, 0.002, len(DATES))
        if jump_date is not None:
            r[DATES.get_loc(pd.Timestamp(jump_date))] += JUMP
        return r

    write_prices(prices_dir / "AAA.csv", DATES, stock_returns("2017-04-25"))
    write_prices(prices_dir / "BBB.csv", DATES, stock_returns("2017-04-26"))
    d0 = DATES.get_loc(pd.Timestamp("2017-06-01"))
    write_prices(prices_dir / "DDD.csv", DATES[d0 - 60 :], stock_returns()[d0 - 60 :])

    dataset = tmp_path / "dataset"
    make_call(dataset, "Alpha Corp_20170425", "Thanks, and good morning everyone\n")
    make_call(dataset, "Beta Inc_20170425", "Good afternoon, everyone\n")
    make_call(dataset, "Gamma Co_20170426", "Hello and welcome\n")
    make_call(dataset, "Delta Ltd_20170601", "Hello and welcome\n")

    company_csv = tmp_path / "company.csv"
    company_csv.write_text(
        COMPANY_HEADER
        + "Alpha Corp,AAA,AAA,Alpha Corp,ok,1,2017-04-25,2017-04-25\n"
        + "Beta Inc,BBB,BBB,Beta Inc,ok,1,2017-04-25,2017-04-25\n"
        + "Gamma Co,GGG,,,no_data,1,2017-04-26,2017-04-26\n"
        + "Delta Ltd,DDD,DDD,Delta Ltd,ok,1,2017-06-01,2017-06-01\n"
    )
    out_dir = tmp_path / "out"
    argv = [
        "--dataset", str(dataset),
        "--company-csv", str(company_csv),
        "--prices-dir", str(prices_dir),
        "--out-dir", str(out_dir),
    ]  # fmt: skip
    return argv, dataset, out_dir


def read_events(out_dir):
    return pd.read_csv(out_dir / "events.csv", keep_default_na=False, na_values=[""]).set_index(
        "event_id"
    )


def test_main_end_to_end(project, capsys):
    argv, _, out_dir = project

    assert main(argv) == 0

    events = read_events(out_dir)
    assert list(events.reset_index().columns) == EVENT_COLUMNS
    assert events["status"].to_dict() == {
        "Alpha Corp_20170425": "ok",
        "Beta Inc_20170425": "ok",
        "Delta Ltd_20170601": "short_estimation",
        "Gamma Co_20170426": "no_price",
    }

    alpha = events.loc["Alpha Corp_20170425"]
    assert (alpha["timing"], alpha["day0_date"]) == ("pre", "2017-04-25")
    assert alpha["beta"] == pytest.approx(1.2, abs=0.05)
    assert alpha["est_n"] == 121
    assert alpha["car_mm_0_1"] == pytest.approx(JUMP, abs=0.01)
    assert alpha["car_mm_m5_m1"] == pytest.approx(0.0, abs=0.01)

    beta = events.loc["Beta Inc_20170425"]
    assert (beta["timing"], beta["day0_date"]) == ("post", "2017-04-26")
    assert beta["car_mm_0_1"] == pytest.approx(JUMP, abs=0.01)
    # Market-adjusted ignores beta, so it is noisier but still sees the jump.
    assert beta["car_ma_0_1"] == pytest.approx(JUMP, abs=0.05)
    assert beta["vol_3"] > beta["vol_pre_3"]  # the jump raises post-call volatility

    gamma = events.loc["Gamma Co_20170426"]
    assert gamma["timing"] == "unknown" and math.isnan(gamma["car_ma_0_1"])

    out = capsys.readouterr().out
    assert "ok: 2" in out and "no_price: 1" in out and "short_estimation: 1" in out


def test_main_short_estimation_keeps_market_adjusted(project):
    argv, _, out_dir = project
    main(argv)

    delta = read_events(out_dir).loc["Delta Ltd_20170601"]
    assert delta["est_n"] < 100
    assert math.isnan(delta["alpha"]) and math.isnan(delta["car_mm_0_1"])
    assert not math.isnan(delta["car_ma_0_1"])
    assert not math.isnan(delta["vol_30"]) and not math.isnan(delta["vol_pre_30"])


def test_main_writes_ar_daily(project):
    argv, _, out_dir = project
    main(argv)

    ar = pd.read_csv(out_dir / "ar_daily.csv")
    assert list(ar.columns) == ["event_id", "rel_day", "ar_mm", "ar_ma"]
    # One row per day in [-5, +60] for every event with prices.
    assert ar.groupby("event_id").size().to_dict() == {
        "Alpha Corp_20170425": 66,
        "Beta Inc_20170425": 66,
        "Delta Ltd_20170601": 66,
    }
    alpha = ar[ar["event_id"] == "Alpha Corp_20170425"].set_index("rel_day")
    assert alpha.loc[0, "ar_mm"] == pytest.approx(JUMP, abs=0.01)


def test_main_unmatched_company_fails(project, capsys):
    argv, dataset, out_dir = project
    make_call(dataset, "Zeta Holdings_20170501", "Good morning\n")

    assert main(argv) == 1
    assert read_events(out_dir).loc["Zeta Holdings_20170501", "status"] == "unmatched"
    assert "unmatched: 1" in capsys.readouterr().out


def test_main_no_day0(project):
    argv, dataset, out_dir = project
    make_call(dataset, "Alpha Corp_20180601", "Good morning\n")  # beyond the price data

    main(argv)
    assert read_events(out_dir).loc["Alpha Corp_20180601", "status"] == "no_day0"


def test_large_returns():
    rel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-03", "2017-01-04", "2017-01-05", "2017-01-06"]),
            "r_i": [0.01, -0.6, np.nan, 0.7],
            "r_m": [0.0, 0.0, 0.0, 0.0],
        },
        index=pd.Index([-200, -1, 0, 61], name="rel_day"),
    )
    # -200 and +61 are outside [-150, +60]; only -1 is reported.
    assert large_returns(rel) == [(pd.Timestamp("2017-01-04"), -0.6)]

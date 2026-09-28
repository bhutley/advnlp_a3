from datetime import date

import pandas as pd
import pytest

import common
import resolve_tickers
from resolve_tickers import (
    load_seed,
    main,
    names_match,
    normalise_name,
    parse_call_dir,
    resolve_company,
    scan_dataset,
    validate_seed,
)


# --- parse_call_dir ---


def test_parse_call_dir():
    assert parse_call_dir("3M Company_20170425") == ("3M Company", date(2017, 4, 25))


def test_parse_call_dir_splits_on_last_underscore():
    assert parse_call_dir("Foo_Bar Inc_20170101") == ("Foo_Bar Inc", date(2017, 1, 1))


@pytest.mark.parametrize("dirname", ["3M Company_20171345", "NoDate", "_20170101", "3M_2017"])
def test_parse_call_dir_rejects_bad_names(dirname):
    with pytest.raises(ValueError):
        parse_call_dir(dirname)


# --- scan_dataset ---


def test_scan_dataset(tmp_path):
    for d in ["AbbVie Inc._20170728", "3M Company_20170425", "AbbVie Inc._20170427"]:
        (tmp_path / d).mkdir()
    (tmp_path / "README.txt").write_text("not a call")

    df = scan_dataset(tmp_path)

    assert list(df.columns) == ["name", "call_date"]
    assert df.values.tolist() == [
        ["3M Company", date(2017, 4, 25)],
        ["AbbVie Inc.", date(2017, 4, 27)],
        ["AbbVie Inc.", date(2017, 7, 28)],
    ]


def test_scan_dataset_reports_all_bad_names(tmp_path):
    for d in ["3M Company_20170425", "Bad One", "Bad Two_20171399"]:
        (tmp_path / d).mkdir()

    with pytest.raises(ValueError) as exc:
        scan_dataset(tmp_path)
    assert "Bad One" in str(exc.value)
    assert "Bad Two_20171399" in str(exc.value)


# --- load_seed ---


def test_load_seed(tmp_path):
    path = tmp_path / "seed.csv"
    path.write_text(
        "name,ticker_2017,yahoo_ticker\n"
        "3M Company,MMM,MMM\n"
        "Aetna Inc,AET,\n"
        '"Facebook, Inc.",FB,META\n'
    )

    seed = load_seed(path)

    assert seed.to_dict("records") == [
        {"name": "3M Company", "ticker_2017": "MMM", "yahoo_ticker": "MMM", "name_confirmed": False},
        {"name": "Aetna Inc", "ticker_2017": "AET", "yahoo_ticker": None, "name_confirmed": False},
        {"name": "Facebook, Inc.", "ticker_2017": "FB", "yahoo_ticker": "META", "name_confirmed": False},
    ]  # fmt: skip


def test_load_seed_name_confirmed(tmp_path):
    path = tmp_path / "seed.csv"
    path.write_text(
        "name,ticker_2017,yahoo_ticker,name_confirmed\n"
        '"Facebook, Inc.",FB,META,Yes\n'
        "3M Company,MMM,MMM,\n"
    )

    assert load_seed(path)["name_confirmed"].tolist() == [True, False]


def test_load_seed_name_confirmed_bad_value(tmp_path):
    path = tmp_path / "seed.csv"
    path.write_text("name,ticker_2017,yahoo_ticker,name_confirmed\n3M Company,MMM,MMM,true\n")

    with pytest.raises(ValueError, match="3M Company"):
        load_seed(path)


def test_load_seed_missing_column(tmp_path):
    path = tmp_path / "seed.csv"
    path.write_text("name,ticker_2017\n3M Company,MMM\n")

    with pytest.raises(ValueError, match="yahoo_ticker"):
        load_seed(path)


# --- validate_seed ---


def make_seed(rows):
    return pd.DataFrame(rows, columns=["name", "ticker_2017", "yahoo_ticker"])


def test_validate_seed_ok():
    seed = make_seed([["3M Company", "MMM", "MMM"], ["Aetna Inc", "AET", None]])
    assert validate_seed(seed, {"3M Company", "Aetna Inc"}) == ([], [])


def test_validate_seed_missing_dataset_name():
    seed = make_seed([["3M Company", "MMM", "MMM"]])
    errors, _ = validate_seed(seed, {"3M Company", "Aetna Inc"})
    assert len(errors) == 1 and "Aetna Inc" in errors[0]


def test_validate_seed_duplicate_name():
    seed = make_seed([["3M Company", "MMM", "MMM"], ["3M Company", "MMM", "MMM2"]])
    errors, _ = validate_seed(seed, {"3M Company"})
    assert any("duplicate name" in e and "3M Company" in e for e in errors)


def test_validate_seed_duplicate_yahoo_ticker():
    seed = make_seed([["Raytheon Co.", "RTN", "RTX"], ["United Technologies", "UTX", "RTX"]])
    errors, _ = validate_seed(seed, {"Raytheon Co.", "United Technologies"})
    assert any("duplicate yahoo_ticker" in e and "RTX" in e for e in errors)


def test_validate_seed_blank_yahoo_tickers_are_not_duplicates():
    seed = make_seed([["Aetna Inc", "AET", None], ["Celgene Corp.", "CELG", None]])
    assert validate_seed(seed, {"Aetna Inc", "Celgene Corp."}) == ([], [])


def test_validate_seed_bad_ticker_format():
    seed = make_seed([["3M Company", "MMM", "../mmm"]])
    errors, _ = validate_seed(seed, {"3M Company"})
    assert any("../mmm" in e for e in errors)


def test_validate_seed_warns_on_extra_seed_name():
    seed = make_seed([["3M Company", "MMM", "MMM"], ["Extra Corp", "XTRA", "XTRA"]])
    errors, warnings = validate_seed(seed, {"3M Company"})
    assert errors == []
    assert len(warnings) == 1 and "Extra Corp" in warnings[0]


# --- name matching ---


def test_normalise_name():
    assert normalise_name("The Estée Lauder Companies Inc.") == {"estee", "lauder"}
    assert normalise_name("Mylan N.V.") == {"mylan"}


@pytest.mark.parametrize(
    "dataset_name, yahoo_name",
    [
        ("Lilly (Eli) & Co.", "Eli Lilly and Company"),
        ("3M Company", "3M Company"),
        ("Estee Lauder Cos.", "The Estée Lauder Companies Inc."),
        ("Grainger (W.W.) Inc.", "W.W. Grainger, Inc."),
    ],
)
def test_names_match(dataset_name, yahoo_name):
    assert names_match(dataset_name, yahoo_name)


@pytest.mark.parametrize(
    "dataset_name, yahoo_name",
    [
        ("Raytheon Co.", "RTX Corporation"),
        ("Mylan N.V.", "Viatris Inc."),
        ("Alaska Air Group Inc", "Other Group Inc"),
        ("X Corp", None),
        ("X Corp", ""),
    ],
)
def test_names_do_not_match(dataset_name, yahoo_name):
    assert not names_match(dataset_name, yahoo_name)


# --- resolve_company ---

FIRST, LAST = date(2017, 4, 27), date(2017, 10, 27)
PRICES = pd.DataFrame({"Close": [1.0]})


def no_call(*args):
    raise AssertionError("Yahoo should not be called")


def raises(*args):
    raise ConnectionError("boom")


def test_resolve_company_no_yahoo_ticker():
    result = resolve_company("Aetna Inc", None, FIRST, LAST, no_call, no_call)
    assert result == {"yahoo_name": None, "status": "no_data"}


def test_resolve_company_fetch_error():
    result = resolve_company("AbbVie Inc.", "ABBV", FIRST, LAST, raises, no_call)
    assert result == {"yahoo_name": None, "status": "error"}


def test_resolve_company_empty_history():
    result = resolve_company("AbbVie Inc.", "ABBV", FIRST, LAST, lambda *a: pd.DataFrame(), no_call)
    assert result == {"yahoo_name": None, "status": "no_data"}


def test_resolve_company_requests_union_window():
    seen = []
    resolve_company(
        "AbbVie Inc.", "ABBV", FIRST, LAST,
        lambda *a: seen.append(a) or PRICES, lambda t: "AbbVie Inc.",
    )  # fmt: skip
    assert seen == [("ABBV", date(2017, 3, 27), date(2018, 10, 28))]


@pytest.mark.parametrize(
    "fetch_name, yahoo_name",
    [(raises, None), (lambda t: None, None), (lambda t: "RTX Corporation", "RTX Corporation")],
)
def test_resolve_company_check_name(fetch_name, yahoo_name):
    result = resolve_company("Raytheon Co.", "RTX", FIRST, LAST, lambda *a: PRICES, fetch_name)
    assert result == {"yahoo_name": yahoo_name, "status": "check_name"}


def test_resolve_company_name_confirmed_accepts_mismatch():
    result = resolve_company(
        "Facebook, Inc.", "META", FIRST, LAST,
        lambda *a: PRICES, lambda t: "Meta Platforms, Inc.", name_confirmed=True,
    )  # fmt: skip
    assert result == {"yahoo_name": "Meta Platforms, Inc.", "status": "ok"}


def test_resolve_company_name_confirmed_still_needs_data():
    result = resolve_company(
        "Facebook, Inc.", "META", FIRST, LAST,
        lambda *a: pd.DataFrame(), no_call, name_confirmed=True,
    )  # fmt: skip
    assert result == {"yahoo_name": None, "status": "no_data"}


def test_resolve_company_ok():
    result = resolve_company(
        "AbbVie Inc.", "ABBV", FIRST, LAST, lambda *a: PRICES, lambda t: "AbbVie Inc."
    )
    assert result == {"yahoo_name": "AbbVie Inc.", "status": "ok"}


# --- main ---


@pytest.fixture
def project(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    for d in ["3M Company_20170425", "3M Company_20170725", "Aetna Inc_20170131"]:
        (dataset / d).mkdir(parents=True)
    seed = tmp_path / "seed.csv"
    seed.write_text("name,ticker_2017,yahoo_ticker\n3M Company,MMM,MMM\nAetna Inc,AET,\n")

    monkeypatch.setattr(common, "fetch_history", lambda *a: PRICES)
    monkeypatch.setattr(common, "fetch_name", lambda t: "3M Company")
    monkeypatch.setattr(resolve_tickers.time, "sleep", lambda s: None)

    out = tmp_path / "data" / "company.csv"
    return ["--dataset", str(dataset), "--seed", str(seed), "--out", str(out)], seed, out


def read_rows(path):
    return pd.read_csv(path, dtype=object, keep_default_na=False).to_dict("records")


def test_main_writes_company_csv(project):
    argv, _, out = project

    assert main(argv) == 0
    assert read_rows(out) == [
        {"name": "3M Company", "ticker_2017": "MMM", "yahoo_ticker": "MMM",
         "yahoo_name": "3M Company", "status": "ok", "n_calls": "2",
         "first_call": "2017-04-25", "last_call": "2017-07-25"},
        {"name": "Aetna Inc", "ticker_2017": "AET", "yahoo_ticker": "",
         "yahoo_name": "", "status": "no_data", "n_calls": "1",
         "first_call": "2017-01-31", "last_call": "2017-01-31"},
    ]  # fmt: skip


def test_main_tickers_filter_matches_2017_ticker(project):
    argv, _, out = project

    assert main(argv + ["--tickers", "aet"]) == 0
    assert [r["name"] for r in read_rows(out)] == ["Aetna Inc"]


def test_main_seed_error_exits_before_yahoo(project, monkeypatch):
    argv, seed, out = project
    seed.write_text("name,ticker_2017,yahoo_ticker\n3M Company,MMM,MMM\n")
    monkeypatch.setattr(common, "fetch_history", no_call)

    assert main(argv) == 1
    assert not out.exists()


def test_main_summary_labels_blank_yahoo_ticker_with_2017_ticker(project, capsys):
    argv, _, _ = project

    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "no_data: 1\n  AET (Aetna Inc)\n" in out

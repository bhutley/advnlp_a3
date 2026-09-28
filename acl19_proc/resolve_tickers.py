"""Resolve dataset company names to tickers and verify them against Yahoo Finance."""

import argparse
import re
import sys
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import common
from common import TICKER_RE, compute_window

CALL_DIR_RE = re.compile(r"^(.+)_(\d{8})$")
SEED_COLUMNS = ["name", "ticker_2017", "yahoo_ticker"]
OUTPUT_COLUMNS = SEED_COLUMNS + ["yahoo_name", "status", "n_calls", "first_call", "last_call"]
STATUSES = ["ok", "no_data", "check_name", "error"]
DEFAULT_DATASET = Path("~/var/data/earnings_calls/ACL19_Release").expanduser()
REQUEST_DELAY = 1.0  # seconds between companies that hit Yahoo

# Words too generic to show that two company names refer to the same company.
FILLER_WORDS = {
    "a", "and", "class", "co", "companies", "company", "corp", "corporation", "cos",
    "group", "holding", "holdings", "inc", "incorporated", "international", "limited",
    "ltd", "plc", "the",
}  # fmt: skip


def parse_call_dir(dirname: str) -> tuple[str, date]:
    """Split '<Company Name>_<YYYYMMDD>' into (name, date); raise ValueError if malformed."""
    m = CALL_DIR_RE.match(dirname)
    if not m:
        raise ValueError(f"not '<name>_<YYYYMMDD>': {dirname!r}")
    return m.group(1), datetime.strptime(m.group(2), "%Y%m%d").date()


def scan_dataset(root: Path) -> pd.DataFrame:
    """One row per call folder under root, sorted; raise ValueError listing every bad folder name."""
    rows, bad = [], []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        try:
            rows.append(parse_call_dir(entry.name))
        except ValueError:
            bad.append(entry.name)
    if bad:
        raise ValueError("bad call folder names: " + ", ".join(sorted(bad)))
    return pd.DataFrame(sorted(rows), columns=["name", "call_date"])


def load_seed(path: Path) -> pd.DataFrame:
    """Read the seed CSV as strings; blank yahoo_ticker becomes None.

    The optional name_confirmed column ("yes" or blank) becomes a bool.
    """
    seed = pd.read_csv(path, dtype=object, keep_default_na=False)
    missing = [c for c in SEED_COLUMNS if c not in seed.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if "name_confirmed" not in seed.columns:
        seed["name_confirmed"] = ""
    seed = seed[SEED_COLUMNS + ["name_confirmed"]].apply(lambda col: col.str.strip())

    confirmed = seed["name_confirmed"].str.lower()
    bad = seed.loc[~confirmed.isin(["yes", ""]), "name"]
    if not bad.empty:
        raise ValueError(f"{path}: name_confirmed must be 'yes' or blank for {list(bad)}")
    seed["name_confirmed"] = confirmed == "yes"

    # object dtype so blanks stay None; pandas 3's str dtype would turn them into NaN
    seed["yahoo_ticker"] = pd.Series(
        [t or None for t in seed["yahoo_ticker"]], index=seed.index, dtype=object
    )
    return seed


def validate_seed(seed: pd.DataFrame, dataset_names: set[str]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) describing problems with the seed."""
    errors, warnings = [], []
    seed_names = set(seed["name"])

    for name in sorted(dataset_names - seed_names):
        errors.append(f"dataset name missing from seed: {name!r}")
    for name in sorted(set(seed.loc[seed["name"].duplicated(), "name"])):
        errors.append(f"duplicate name in seed: {name!r}")

    tickers = seed["yahoo_ticker"].dropna()
    for ticker in sorted(set(tickers[tickers.duplicated()])):
        errors.append(f"duplicate yahoo_ticker in seed: {ticker!r}")
    for ticker in sorted(t for t in set(tickers) if not TICKER_RE.match(t)):
        errors.append(f"invalid yahoo_ticker in seed: {ticker!r}")

    for name in sorted(seed_names - dataset_names):
        warnings.append(f"seed name not in dataset: {name!r}")
    return errors, warnings


def normalise_name(name: str) -> set[str]:
    """Distinctive lowercase words of a company name, without accents, punctuation or filler."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    words = re.sub(r"[^a-z0-9]+", " ", ascii_name.lower()).split()
    return {w for w in words if len(w) > 1 and w not in FILLER_WORDS}


def names_match(dataset_name: str, yahoo_name: str | None) -> bool:
    """True if the two names share at least one distinctive word."""
    if not yahoo_name:
        return False
    return bool(normalise_name(dataset_name) & normalise_name(yahoo_name))


def resolve_company(
    name: str,
    yahoo_ticker: str | None,
    first_call: date,
    last_call: date,
    fetch_history=common.fetch_history,
    fetch_name=common.fetch_name,
    *,
    name_confirmed: bool = False,
) -> dict:
    """Check yahoo_ticker against Yahoo; return {"yahoo_name", "status"}.

    name_confirmed marks a name mismatch Brett has reviewed and accepted.
    """
    if yahoo_ticker is None:
        return {"yahoo_name": None, "status": "no_data"}
    try:
        prices = fetch_history(yahoo_ticker, *compute_window(first_call, last_call))
    except Exception:
        return {"yahoo_name": None, "status": "error"}
    if prices.empty:
        return {"yahoo_name": None, "status": "no_data"}
    try:
        yahoo_name = fetch_name(yahoo_ticker)
    except Exception:
        yahoo_name = None
    status = "ok" if name_confirmed or names_match(name, yahoo_name) else "check_name"
    return {"yahoo_name": yahoo_name, "status": status}


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=Path, default=Path("tickers_seed.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/company.csv"))
    parser.add_argument(
        "--tickers", help="comma-separated tickers (2017 or Yahoo) to limit the run to"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        calls = scan_dataset(args.dataset)
        seed = load_seed(args.seed)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    errors, warnings = validate_seed(seed, set(calls["name"]))
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if errors:
        return 1

    summary = (
        calls.groupby("name")["call_date"]
        .agg(n_calls="count", first_call="min", last_call="max")
        .reset_index()
    )
    companies = seed.merge(summary, on="name").sort_values("name")
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
        mask = companies["ticker_2017"].isin(wanted) | companies["yahoo_ticker"].isin(wanted)
        companies = companies[mask]
        if companies.empty:
            print(f"error: no companies match --tickers {args.tickers}", file=sys.stderr)
            return 1

    rows, hit_yahoo = [], False
    for i, c in enumerate(companies.to_dict("records"), start=1):
        if c["yahoo_ticker"] is not None:
            if hit_yahoo:
                time.sleep(REQUEST_DELAY)
            hit_yahoo = True
        result = resolve_company(
            c["name"], c["yahoo_ticker"], c["first_call"], c["last_call"],
            common.fetch_history, common.fetch_name, name_confirmed=c["name_confirmed"],
        )  # fmt: skip
        rows.append(c | result)
        label = c["yahoo_ticker"] or c["ticker_2017"]
        print(f"[{i}/{len(companies)}] {label}: {result['status']}")

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"\nwrote {len(out)} companies to {args.out}")
    for status in STATUSES:
        group = [r for r in rows if r["status"] == status]
        print(f"{status}: {len(group)}")
        if status != "ok":
            for r in group:
                print(f"  {r['yahoo_ticker'] or r['ticker_2017']} ({r['name']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

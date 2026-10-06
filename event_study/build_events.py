"""Build the event-study table: one row per earnings call, with CARs and volatility.

Reads the ACL19 dataset folders, acl19_proc's company.csv and price files, and writes
data/events.csv (one row per call) and data/ar_daily.csv (daily abnormal returns over
[-5, +60] for every call with prices).
"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from events_input import scan_dataset
from returns import (
    CAR_WINDOWS,
    EST_WINDOW,
    EVENT_WINDOW,
    MIN_EST_OBS,
    VOL_HORIZONS,
    abnormal_returns,
    car,
    fit_market_model,
    load_prices,
    log_vol,
    relative_frame,
)

LARGE_RETURN = 0.5  # |daily return| above this is printed as a likely data error

EVENT_COLUMNS = (
    ["event_id", "company", "ticker", "call_date", "timing", "day0_date", "status"]
    + ["alpha", "beta", "sigma", "est_n"]
    + [f"car_{m}_{w}" for m in ("mm", "ma") for w in CAR_WINDOWS]
    + [f"vol_{n}" for n in VOL_HORIZONS]
    + [f"vol_pre_{n}" for n in VOL_HORIZONS]
)
AR_COLUMNS = ["event_id", "rel_day", "ar_mm", "ar_ma"]


def large_returns(rel: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    """(date, return) for each stock return beyond ±LARGE_RETURN within [-150, +60]."""
    window = rel.loc[(rel.index >= EST_WINDOW[0]) & (rel.index <= EVENT_WINDOW[1])]
    big = window[window["r_i"].abs() > LARGE_RETURN]
    return list(zip(big["date"], big["r_i"]))


def build_event(
    call: dict, company: dict | None, prices_dir: Path, market: pd.Series
) -> tuple[dict, pd.DataFrame | None]:
    """One call → (events.csv row, ar_daily rows or None).

    Status, in order of precedence: unmatched, no_price, no_day0, short_estimation, ok.
    """
    row: dict[str, object] = dict.fromkeys(EVENT_COLUMNS, math.nan)
    row.update(
        event_id=call["event_id"],
        company=call["company"],
        call_date=call["call_date"].isoformat(),
        timing=call["timing"],
        ticker="",
        day0_date="",
    )
    if company is None:
        return {**row, "status": "unmatched"}, None
    if company["status"] != "ok":
        return {**row, "status": "no_price"}, None
    row["ticker"] = company["yahoo_ticker"]

    stock = load_prices(prices_dir / f"{row['ticker']}.csv")
    rel = relative_frame(stock, market, call["call_date"], call["timing"])
    if rel is None:
        return {**row, "status": "no_day0"}, None
    row["day0_date"] = rel.loc[0, "date"].date().isoformat()
    for when, r in large_returns(rel):
        print(f"warning: {row['event_id']} {when.date()} return={r:.3f}")

    alpha, beta, sigma, n = fit_market_model(rel.loc[EST_WINDOW[0] : EST_WINDOW[1]])
    row["est_n"] = n
    if n >= MIN_EST_OBS:
        row.update(alpha=alpha, beta=beta, sigma=sigma, status="ok")
    else:
        alpha = beta = math.nan
        row["status"] = "short_estimation"

    ar = abnormal_returns(rel, alpha, beta)
    for model in ("mm", "ma"):
        for name, window in CAR_WINDOWS.items():
            row[f"car_{model}_{name}"] = car(ar[f"ar_{model}"], window)
    for n in VOL_HORIZONS:
        row[f"vol_{n}"] = log_vol(rel["r_i"], 0, n)
        row[f"vol_pre_{n}"] = log_vol(rel["r_i"], -n - 1, n)

    ar_rows = ar.reset_index()
    ar_rows.insert(0, "event_id", row["event_id"])
    return row, ar_rows[AR_COLUMNS]


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path.home() / "var/data/earnings_calls/ACL19_Release"
    )
    parser.add_argument(
        "--company-csv", type=Path, default=Path("../acl19_proc/data/company.csv")
    )
    parser.add_argument("--prices-dir", type=Path, default=Path("../acl19_proc/data"))
    parser.add_argument("--market", default="SPY", help="market index ticker (default SPY)")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    calls = scan_dataset(args.dataset)
    companies = pd.read_csv(args.company_csv, dtype=object, keep_default_na=False)
    by_name = {c["name"]: c for c in companies.to_dict("records")}
    market = load_prices(args.prices_dir / f"{args.market}.csv")

    rows, ar_frames = [], []
    for call in calls.to_dict("records"):
        row, ar = build_event(call, by_name.get(call["company"]), args.prices_dir, market)
        rows.append(row)
        if ar is not None:
            ar_frames.append(ar)
        if row["status"] == "unmatched":
            print(f"error: {row['event_id']}: company not in {args.company_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    events.to_csv(args.out_dir / "events.csv", index=False)
    ar_daily = pd.concat(ar_frames) if ar_frames else pd.DataFrame(columns=AR_COLUMNS)
    ar_daily.to_csv(args.out_dir / "ar_daily.csv", index=False)

    print()
    print(f"events: {len(events)}")
    for status, n in Counter(events["status"]).most_common():
        print(f"{status}: {n}")
    with_prices = events[events["status"].isin(["ok", "short_estimation"])]
    print("timing (events with prices): " + ", ".join(
        f"{t} {n}" for t, n in Counter(with_prices["timing"]).most_common()
    ))  # fmt: skip
    return 1 if (events["status"] == "unmatched").any() else 0


if __name__ == "__main__":
    sys.exit(main())

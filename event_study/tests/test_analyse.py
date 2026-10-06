import math

import numpy as np
import pandas as pd
import pytest

from analyse import (
    _spread,
    abs_ar_by_day,
    caar_paths,
    caar_table,
    markdown_table,
    plot_abs_ar,
    plot_caar,
    vol_table,
    winsorize,
)
from returns import CAR_WINDOWS, EVENT_WINDOW, VOL_HORIZONS

DAYS = range(EVENT_WINDOW[0], EVENT_WINDOW[1] + 1)


def make_events(cars, status=None, timing=None):
    """events.csv-shaped frame with every CAR window set to the same values."""
    n = len(cars)
    df = pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(n)],
            "status": status or ["ok"] * n,
            "timing": timing or ["pre"] * n,
        }
    )
    for model in ("mm", "ma"):
        for w in CAR_WINDOWS:
            df[f"car_{model}_{w}"] = cars
    return df


def make_ar_daily(per_event):
    """{event_id: value or list of 66 values} → ar_daily.csv-shaped frame."""
    frames = []
    for event_id, values in per_event.items():
        values = values if isinstance(values, list) else [values] * len(DAYS)
        frames.append(
            pd.DataFrame(
                {"event_id": event_id, "rel_day": list(DAYS), "ar_mm": values, "ar_ma": values}
            )
        )
    return pd.concat(frames, ignore_index=True)


# --- winsorize ---


def test_winsorize_clips_tails():
    s = pd.Series(list(range(1, 100)) + [1000.0])
    w = winsorize(s, 0.01)
    assert w.max() == pytest.approx(s.quantile(0.99))
    assert w.min() == pytest.approx(s.quantile(0.01))
    assert w.iloc[50] == s.iloc[50]  # the middle is untouched


def test_winsorize_keeps_nan():
    w = winsorize(pd.Series([1.0, np.nan, 3.0]), 0.01)
    assert math.isnan(w.iloc[1])


# --- caar_table ---


def test_caar_table():
    cars = [0.01, 0.02, 0.03, -0.01]
    events = pd.concat(
        [make_events(cars), make_events([np.nan], status=["no_price"])], ignore_index=True
    )

    table = caar_table(events, "mm", winsor=0.0)

    assert list(table.index) == list(CAR_WINDOWS)
    row = table.loc["0_1"]
    sd = np.std(cars, ddof=1)
    assert row["n"] == 4  # the no_price row is excluded
    assert row["mean"] == pytest.approx(0.0125)
    assert row["t"] == pytest.approx(0.0125 / (sd / 2))
    assert row["pct_pos"] == pytest.approx(75.0)
    # winsor=0 leaves the data unchanged
    assert row["mean_w"] == pytest.approx(row["mean"])


def test_caar_table_winsorized_columns_resist_outlier():
    events = make_events([0.01] * 99 + [5.0])
    row = caar_table(events, "mm", winsor=0.01).loc["0_60"]
    assert row["mean"] > 0.05
    assert row["mean_w"] < 0.07


def test_caar_table_drops_nan_per_window():
    events = make_events([0.01, 0.02, 0.03])
    events.loc[0, "car_mm_0_60"] = np.nan
    table = caar_table(events, "mm")
    assert table.loc["0_60", "n"] == 2 and table.loc["0_1", "n"] == 3


# --- vol_table ---


def test_vol_table_paired_difference():
    events = make_events([0.0, 0.0, 0.0])
    for n in VOL_HORIZONS:
        events[f"vol_pre_{n}"] = [-5.0, -4.0, -6.0]
        events[f"vol_{n}"] = [-4.6, -3.4, -5.5]  # differences 0.4, 0.6, 0.5

    row = vol_table(events).loc[3]
    assert row["n"] == 3
    assert row["post"] == pytest.approx(-4.5)
    assert row["pre"] == pytest.approx(-5.0)
    assert row["diff"] == pytest.approx(0.5)
    assert row["t"] == pytest.approx(0.5 / (0.1 / math.sqrt(3)))


# --- abs_ar_by_day ---


def test_abs_ar_by_day():
    events = make_events([0.0, 0.0, 0.0], timing=["pre", "pre", "post"])
    ar = make_ar_daily({"E0": 0.02, "E1": -0.04, "E2": 0.01})

    table = abs_ar_by_day(ar, events, days=(-1, 1))

    assert list(table.columns) == [-1, 0, 1]
    assert table.loc["pre", 0] == pytest.approx(0.03)
    assert table.loc["post", 0] == pytest.approx(0.01)


# --- caar_paths ---


def test_caar_paths_cumulates_complete_events_only():
    events = make_events([0.0, 0.0, 0.0], timing=["pre", "pre", "post"])
    incomplete = [0.01] * (len(DAYS) - 1) + [np.nan]
    ar = make_ar_daily({"E0": 0.01, "E1": 0.03, "E2": incomplete})

    paths = caar_paths(ar, events)

    assert set(paths["group"]) == {"All"}
    last = paths.set_index("rel_day").loc[EVENT_WINDOW[1]]
    assert last["n"] == 2  # E2 has a missing day, so it is dropped
    assert last["caar"] == pytest.approx(0.02 * len(DAYS))
    first = paths.set_index("rel_day").loc[EVENT_WINDOW[0]]
    assert first["caar"] == pytest.approx(0.02)
    assert first["se"] == pytest.approx(np.std([0.01, 0.03], ddof=1) / math.sqrt(2))


def test_caar_paths_by_group():
    events = make_events([0.0, 0.0, 0.0], timing=["pre", "pre", "post"])
    ar = make_ar_daily({"E0": 0.01, "E1": 0.03, "E2": -0.01})

    paths = caar_paths(ar, events, group_col="timing")

    day0 = paths[paths["rel_day"] == 0].set_index("group")
    assert day0.loc["pre", "caar"] == pytest.approx(0.02 * 6)  # days -5..0
    assert day0.loc["post", "caar"] == pytest.approx(-0.01 * 6)
    assert day0.loc["post", "n"] == 1


# --- output helpers ---


def test_markdown_table():
    df = pd.DataFrame({"a": [1, 2], "b": [0.12345, np.nan]}, index=pd.Index(["x", "y"], name="k"))
    assert markdown_table(df, "{:.2f}") == (
        "| k | a | b |\n|---|---|---|\n| x | 1 | 0.12 |\n| y | 2 |  |"
    )


def test_plots_write_files(tmp_path):
    events = make_events([0.0, 0.0], timing=["pre", "post"])
    ar = make_ar_daily({"E0": 0.01, "E1": -0.01})

    plot_caar(caar_paths(ar, events, "timing"), "CAAR by timing", tmp_path / "caar.png")
    plot_abs_ar(abs_ar_by_day(ar, events), tmp_path / "abs.png")

    assert (tmp_path / "caar.png").stat().st_size > 0
    assert (tmp_path / "abs.png").stat().st_size > 0


def test_spread_separates_close_labels():
    assert _spread([0.0, 0.01, 0.5], 0.05) == [0.0, 0.05, 0.5]
    assert _spread([0.3, 0.0], 0.05) == [0.3, 0.0]  # already apart, order preserved

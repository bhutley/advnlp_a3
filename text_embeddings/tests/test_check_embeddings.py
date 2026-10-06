import numpy as np
import pandas as pd
import pytest

from check_embeddings import correlations, direction_failures, target_table


def test_direction_failures_positive():
    probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])  # second scores negative
    assert direction_failures(["good", "oops"], probs, positive=True) == ["oops"]


def test_direction_failures_negative():
    probs = np.array([[0.1, 0.8, 0.1], [0.3, 0.3, 0.4]])  # a tie is a failure
    assert direction_failures(["bad", "tie"], probs, positive=False) == ["tie"]


def test_direction_failures_all_correct():
    probs = np.array([[0.1, 0.8, 0.1]])
    assert direction_failures(["bad"], probs, positive=False) == []


def test_correlations_drops_missing_rows():
    x = pd.Series([1.0, 2.0, 3.0, np.nan], index=list("abcd"))
    y = pd.Series([2.0, 4.0, 6.0, 1.0], index=list("abcd"))
    c = correlations(x, y)
    assert c["n"] == 3
    assert c["pearson"] == pytest.approx(1.0) and c["spearman"] == pytest.approx(1.0)


def test_target_table_uses_ok_events_only():
    calls = pd.DataFrame({"event_id": list("abcd"), "tone": [0.1, 0.2, 0.3, 0.4],
                          "frac_neg": [0.4, 0.3, 0.2, 0.1]})  # fmt: skip
    events = pd.DataFrame({"event_id": list("abcd"), "status": ["ok", "ok", "ok", "no_price"]})
    for t in ("car_mm_0_1", "car_mm_0_5", "vol_3", "vol_30"):
        events[t] = [1.0, 2.0, 3.0, np.nan]

    table = target_table(calls, events)

    assert set(table["n"]) == {3}
    row = table[(table.feature == "tone") & (table.target == "vol_3")].iloc[0]
    assert row["spearman"] == pytest.approx(1.0)
    row = table[(table.feature == "frac_neg") & (table.target == "vol_3")].iloc[0]
    assert row["spearman"] == pytest.approx(-1.0)

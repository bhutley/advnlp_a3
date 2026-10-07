import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from evaluate import compare, holm, metrics, per_call_loss


def oof_frame(rows):
    df = pd.DataFrame(rows, columns=["event_id", "target", "rung", "repeat", "y", "y_hat"])
    df["fold"], df["params"] = 0, "{}"
    return df


def test_metrics_regression():
    oof = oof_frame([
        ("a", "vol_3", "R1", 0, 1.0, 1.5), ("b", "vol_3", "R1", 0, 2.0, 2.0),  # repeat 0 MSE 0.125
        ("a", "vol_3", "R1", 1, 1.0, 2.0), ("b", "vol_3", "R1", 1, 2.0, 2.0),  # repeat 1 MSE 0.5
    ])  # fmt: skip
    m = metrics(oof).set_index(["target", "rung", "metric"])
    row = m.loc[("vol_3", "R1", "mse")]
    assert row["mean"] == pytest.approx(0.3125)
    assert row["sd"] == pytest.approx(np.std([0.125, 0.5], ddof=1))
    assert row["n_repeats"] == 2


def test_metrics_classification():
    y, p = [0, 1, 1, 0, 1], [0.2, 0.7, 0.4, 0.6, 0.9]
    oof = oof_frame([(f"e{i}", "dir_0_1", "R2", 0, yi, pi) for i, (yi, pi) in enumerate(zip(y, p))])
    m = metrics(oof).set_index("metric")
    assert m.loc["auc", "mean"] == pytest.approx(roc_auc_score(y, p))
    assert m.loc["accuracy", "mean"] == pytest.approx(0.6)  # 0.5 threshold: 3 of 5 right
    assert m.loc["log_loss", "mean"] == pytest.approx(log_loss(y, p))


def test_per_call_loss_averages_repeats():
    oof = oof_frame([("a", "vol_3", "R1", 0, 1.0, 2.0), ("a", "vol_3", "R1", 1, 1.0, 1.0),
                     ("b", "vol_3", "R1", 0, 0.0, 0.0), ("b", "vol_3", "R1", 1, 0.0, 0.0)])  # fmt: skip
    loss = per_call_loss(oof, "regression")
    assert loss.to_dict() == {"a": 0.5, "b": 0.0}


def paired_oof(loss_a, loss_b):
    """Regression OOF where y = 0 and y_hat = sqrt(loss), so squared error = loss."""
    rows = []
    for rung, losses in (("A", loss_a), ("B", loss_b)):
        rows += [(f"e{i}", "vol_3", rung, 0, 0.0, float(np.sqrt(v))) for i, v in enumerate(losses)]
    return oof_frame(rows)


def test_compare_sign_and_antisymmetry():
    loss_a, loss_b = np.full(30, 1.0), np.full(30, 1.5)  # A is better by 0.5
    oof = paired_oof(loss_a, loss_b)
    groups = pd.Series([f"F{i // 3}" for i in range(30)], index=[f"e{i}" for i in range(30)])

    ab = compare(oof, "A", "B", "vol_3", groups, n_boot=2000)
    ba = compare(oof, "B", "A", "vol_3", groups, n_boot=2000)

    assert ab["mean_diff"] == pytest.approx(0.5) and ab["p"] < 0.01
    assert ba["mean_diff"] == pytest.approx(-0.5) and ba["p"] > 0.99
    assert (ab["n_calls"], ab["n_firms"]) == (30, 10)


def test_compare_clusters_by_firm():
    # One firm with 20 calls carries the whole difference; 19 firms have one call each.
    diff = np.r_[np.ones(20), np.zeros(19)]
    oof = paired_oof(np.zeros(39), diff)
    ids = [f"e{i}" for i in range(39)]
    by_firm = pd.Series(["BIG"] * 20 + [f"F{i}" for i in range(19)], index=ids)
    by_call = pd.Series(ids, index=ids)  # every call its own cluster

    firm = compare(oof, "A", "B", "vol_3", by_firm, n_boot=4000)
    call = compare(oof, "A", "B", "vol_3", by_call, n_boot=4000)

    assert firm["mean_diff"] == call["mean_diff"] == pytest.approx(20 / 39)
    assert firm["ci_high"] - firm["ci_low"] > 2 * (call["ci_high"] - call["ci_low"])
    assert firm["ci_low"] <= 0.0 < call["ci_low"]  # only the clustered CI admits no effect


def test_compare_no_difference_covers_zero():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 1.5, 200)
    noise = rng.normal(0, 0.05, 200)
    oof = paired_oof(np.clip(base + noise, 0, None), np.clip(base - noise, 0, None))
    groups = pd.Series([f"F{i // 2}" for i in range(200)], index=[f"e{i}" for i in range(200)])

    r = compare(oof, "A", "B", "vol_3", groups, n_boot=4000)
    assert r["ci_low"] < 0 < r["ci_high"]


def test_holm():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert adjusted == pytest.approx({"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06})
    assert holm({"x": 0.6, "y": 0.7}) == pytest.approx({"x": 1.0, "y": 1.0})  # capped at 1

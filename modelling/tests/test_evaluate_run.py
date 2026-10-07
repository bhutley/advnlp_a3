import json

import numpy as np
import pytest

from conftest import make_call_data
from cv import outer_splits
from evaluate import run_rung

FAST = {"ks": (4, 8), "layers": (3, 6), "n_jobs": 1}


@pytest.fixture(scope="module")
def data():
    return make_call_data()


@pytest.fixture(scope="module")
def splits(data):
    return outer_splits(data.groups, n_splits=5, n_repeats=1)


def mse(oof):
    return float(((oof["y"] - oof["y_hat"]) ** 2).mean())


def test_columns_and_coverage(data, splits):
    oof = run_rung(data, "R1", "vol_3", splits, **FAST)
    assert list(oof.columns) == ["event_id", "target", "rung", "repeat", "fold", "y", "y_hat", "params"]
    assert sorted(oof["event_id"]) == sorted(data.table["event_id"])  # each call once per repeat
    assert set(oof["target"]) == {"vol_3"} and set(oof["rung"]) == {"R1"}


def test_b0_is_past_volatility(data, splits):
    oof = run_rung(data, "B0", "vol_7", splits, **FAST).set_index("event_id")
    past = data.table.set_index("event_id")["vol_pre_7"]
    np.testing.assert_allclose(oof["y_hat"], past.loc[oof.index])
    assert set(oof["params"]) == {"{}"}


def test_planted_text_signal_is_found(data, splits):
    # PCA is unsupervised: with k far below the 20 equally noisy text dims, it can drop the
    # planted directions. A grid reaching k=16 keeps them, as the real grid's k=64 can.
    grid = FAST | {"ks": (8, 16)}
    r1 = mse(run_rung(data, "R1", "vol_3", splits, **grid))
    r2 = mse(run_rung(data, "R2", "vol_3", splits, **grid))
    assert r2 < 0.5 * r1


def test_noise_audio_adds_nothing(data, splits):
    r1 = mse(run_rung(data, "R1", "vol_3", splits, **FAST))
    r3 = mse(run_rung(data, "R3", "vol_3", splits, **FAST))
    assert r3 >= 0.95 * r1


def test_permutation_destroys_signal(data, splits):
    r1 = mse(run_rung(data, "R1", "vol_3", splits, permute=0, **FAST))
    r2 = mse(run_rung(data, "R2", "vol_3", splits, permute=0, **FAST))
    assert r2 >= 0.95 * r1


def test_params_recorded(data, splits):
    oof = run_rung(data, "R3", "vol_3", splits, **FAST)
    params = json.loads(oof["params"].iloc[0])
    assert set(params) == {"model__alpha", "features__audio_emb__pca__n_components",
                           "features__audio_emb__layer__layer"}  # fmt: skip
    assert params["features__audio_emb__layer__layer"] in (3, 6)
    assert oof.groupby("fold")["params"].nunique().eq(1).all()  # one choice per outer fold


def test_direction(data, splits):
    oof = run_rung(data, "R2", "dir_0_1", splits, **FAST)
    assert oof["y_hat"].between(0, 1).all()
    b0 = run_rung(data, "B0", "dir_0_1", splits, **FAST)
    # B0 predicts the training base rate, a constant within each fold
    assert b0.groupby("fold")["y_hat"].nunique().eq(1).all()
    assert b0["y_hat"].between(0.2, 0.8).all()


def test_r3_resid_uses_the_residualizer(data, splits, monkeypatch):
    import evaluate

    seen = []
    real = evaluate.build_pipeline
    monkeypatch.setattr(evaluate, "build_pipeline",
                        lambda *a, **k: seen.append(k.get("residualize", False)) or real(*a, **k))  # fmt: skip
    run_rung(data, "R3", "vol_3", splits[:1], **FAST)
    run_rung(data, "R3_resid", "vol_3", splits[:1], **FAST)
    assert seen == [False, True]  # round-1 rungs are built exactly as before

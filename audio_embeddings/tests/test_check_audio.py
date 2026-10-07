import numpy as np
import pandas as pd
import pytest

from check_audio import (
    count_check,
    format_predictability,
    kruskal_by_group,
    offset_check,
    plausibility,
    pulse_train,
    rsa,
    target_table,
)
from praat import praat_features

SR = 16000


# --- count_check ---


def test_count_check():
    meta = {"n_calls": 572, "n_clips": 89722, "n_failed": 1}
    sent = np.ones((5, 3)); sent[2] = np.nan  # the one failed clip
    call = np.ones((2, 13, 3))

    assert count_check(meta, 572, 89722, n_empty=1, sentence_emb=sent, call_emb=call)
    # a failure that isn't an empty dataset file
    assert not count_check(meta | {"n_failed": 2}, 572, 89722, 1, sent, call)
    assert not count_check(meta | {"n_clips": 89721}, 572, 89722, 1, sent, call)
    # NaN rows beyond the failed clips, or any NaN call vector
    sent2 = sent.copy(); sent2[3] = np.nan
    assert not count_check(meta, 572, 89722, 1, sent2, call)
    call2 = call.copy(); call2[0, 0, 0] = np.nan
    assert not count_check(meta, 572, 89722, 1, sent, call2)
    # praat has no arrays
    assert count_check(meta, 572, 89722, 1, None, None)


# --- offset_check: decode offsets must be constant within each call ---


def test_offset_check_constant_within_calls_passes():
    # Different calls have different encoder delays (by sample rate); each is constant.
    by_call = {"a": np.array([-0.048, -0.048, -0.0481]), "b": np.array([-0.012, -0.0115, -0.012])}
    worst_range, max_abs, ok = offset_check(by_call)
    assert worst_range == pytest.approx(0.0005, abs=1e-6) and max_abs == pytest.approx(0.0481)
    assert ok


def test_offset_check_varying_within_call_fails():
    *_, ok = offset_check({"a": np.array([-0.048, -0.020])})
    assert not ok


def test_offset_check_large_offset_fails():
    *_, ok = offset_check({"a": np.array([-0.3, -0.3])})
    assert not ok


# --- recording-format helpers ---


def test_kruskal_by_group():
    df = pd.DataFrame({"x": [1.0, 1.1, 0.9, 5.0, 5.1, 4.9, np.nan], "g": list("aaabbbb")})
    medians, p = kruskal_by_group(df, "x", "g")
    assert medians == {"a": pytest.approx(1.0), "b": pytest.approx(5.0)}
    assert p < 0.1


def test_format_predictability():
    rng = np.random.default_rng(0)
    labels = np.array(["a"] * 30 + ["b"] * 30)
    separated = np.vstack([rng.normal(0, 1, (30, 5)), rng.normal(4, 1, (30, 5))])
    mixed = rng.normal(0, 1, (60, 5))

    acc, majority = format_predictability(separated, labels)
    assert acc > 0.95 and majority == pytest.approx(0.5)
    assert format_predictability(mixed, labels)[0] < 0.75


# --- pulse_train: a voiced signal with known jitter ---


def test_pulse_train_jitter_is_measurable():
    # Uniform ±1% period perturbation gives expected local jitter 2/3 × 1% ≈ 0.0067.
    raw = praat_features(pulse_train(jitter=0.01, seed=0), SR, n_words=1)
    assert raw["f0_mean_hz"] == pytest.approx(120, abs=3)
    assert 0.004 < raw["jitter_local"] < 0.010


def test_pulse_train_without_jitter():
    steady = praat_features(pulse_train(jitter=0.0, seed=0), SR, n_words=1)
    assert steady["jitter_local"] < 0.002


# --- rsa ---


def test_rsa_identical_and_rotated():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))  # a random rotation

    assert rsa(a, a) == pytest.approx(1.0)
    assert rsa(a, a @ q) == pytest.approx(1.0)  # cosine similarity ignores rotation


def test_rsa_unrelated_is_near_zero():
    rng = np.random.default_rng(1)
    assert abs(rsa(rng.normal(size=(60, 8)), rng.normal(size=(60, 8)))) < 0.1


# --- plausibility ---


def test_plausibility_flags_out_of_range_pitch():
    calls = pd.DataFrame({
        "event_id": ["a", "b", "c"],
        "f0_mean_hz": [110.0, 300.0, 70.0],
        "speech_rate": [2.0, 2.5, 3.0],
        "voiced_frac": [0.5, 0.5, 0.6],
    })  # fmt: skip
    summary, flagged = plausibility(calls)
    assert flagged == ["b", "c"]
    assert summary["speech_rate_median"] == pytest.approx(2.5)


# --- target_table ---


def test_target_table_uses_ok_events():
    calls = pd.DataFrame({"event_id": list("abcd"), "f0_sd_st": [1.0, 2.0, 3.0, 4.0]})
    events = pd.DataFrame({"event_id": list("abcd"), "status": ["ok", "ok", "ok", "no_price"],
                           "vol_3": [1.0, 2.0, 3.0, np.nan]})  # fmt: skip

    table = target_table(calls, events, features=["f0_sd_st"], targets=["vol_3"])

    row = table.iloc[0]
    assert (row["feature"], row["target"], row["n"]) == ("f0_sd_st", "vol_3", 3)
    assert row["spearman"] == pytest.approx(1.0)

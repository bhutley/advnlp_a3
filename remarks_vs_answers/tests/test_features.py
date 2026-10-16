"""Section-feature logic on synthetic data: aggregation, recombination, equal-length
subsets, the round-3 sample rule and the audio section sums."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import build_section_features as bsf
from sections import SECTIONS

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "text_embeddings"))


def test_sentiment_matches_text_embeddings_aggregation():
    import embed  # text_embeddings/embed.py

    rng = np.random.default_rng(0)
    probs = rng.dirichlet([1, 1, 1], size=20).astype(np.float32)
    out = embed.SentenceOutput(emb=np.zeros((20, 4), np.float32), probs=probs, n_tokens=np.ones(20), n_truncated=0)
    calls, _ = embed.aggregate_calls(["c"], pd.DataFrame({"event_id": ["c"] * 20}), out)
    ours = bsf.sentiment(probs)
    for col in bsf.SENT_COLS:
        assert ours[col] == pytest.approx(float(calls.loc[0, col]), abs=1e-6)


def test_equal_subsets_keep_the_smaller_section_and_sample_the_larger():
    rem, ans = np.arange(0, 30), np.arange(30, 42)
    a, b = bsf.equal_subsets(rem, ans, np.random.default_rng(0))
    assert len(a) == len(b) == 12 and (b == ans).all() and set(a) <= set(rem) and (np.diff(a) > 0).all()
    a2, _ = bsf.equal_subsets(rem, ans, np.random.default_rng(0))
    assert (a == a2).all()  # seeded, deterministic


def test_section_means_recombine_to_the_call_mean_with_an_empty_section():
    rng = np.random.default_rng(1)
    labels = pd.DataFrame({"event_id": ["a"] * 7 + ["b"] * 5,
                           "section": ["remarks"] * 3 + ["answers"] * 3 + ["closing"] + ["remarks"] * 2 + ["answers"] * 3})  # fmt: skip
    values = rng.normal(size=(12, 4))
    means, counts = bsf.section_means(labels, values, ["a", "b"])
    assert counts.tolist() == [[3, 3, 1, 0], [2, 3, 0, 0]]
    assert np.isnan(means[1, SECTIONS.index("closing")]).all()
    calls = bsf.recombine(means, counts)
    assert np.allclose(calls, [values[:7].mean(axis=0), values[7:].mean(axis=0)])


def test_round3_sample_needs_ten_audio_ok_sentences_in_each_section():
    rows, praat = [], []
    for event_id, n_rem, n_ans, bad in [("a", 10, 10, 0), ("b", 12, 9, 0), ("c", 11, 11, 2), ("d", 20, 20, 0)]:
        for k in range(n_rem + n_ans):
            rows.append((event_id, k, "remarks" if k < n_rem else "answers"))
            praat.append((event_id, k, not (k < bad)))  # the first `bad` remarks clips fail
    sections = pd.DataFrame(rows, columns=["event_id", "sent_idx", "section"])
    praat = pd.DataFrame(praat, columns=["event_id", "sent_idx", "audio_ok"])
    calls = pd.DataFrame({"event_id": list("abcd"), "in_sample": [True, True, True, False]})
    # b: 9 answers; c: 11 - 2 = 9 usable remarks; d: not a modelling call
    assert bsf.round3_sample(sections, calls, praat) == ["a"]


def test_prosody_features_are_nan_aware_with_shift_and_counts():
    rows = []
    for k, (sec, f0) in enumerate([("remarks", 2.0), ("remarks", np.nan), ("remarks", 4.0),
                                   ("answers", 5.0), ("answers", 7.0), ("closing", 100.0)]):  # fmt: skip
        rows.append({"event_id": "a", "sent_idx": k, "section": sec, "audio_ok": True, "f0_sd_st": f0,
                     "speech_rate": 1.0, "intensity_sd_db": 1.0, "voiced_frac": 0.5})  # fmt: skip
    df = pd.DataFrame(rows)
    out = bsf.prosody_features(["a"], df.drop(columns="section"), df[["event_id", "sent_idx", "section"]])
    r = out.iloc[0]
    assert r["prosody_rem_f0_sd_st"] == 3.0 and r["prosody_ans_f0_sd_st"] == 6.0 and r["shift_f0_sd_st"] == 3.0
    assert r["n_rem_ok"] == 3 and r["n_ans_ok"] == 2  # closing is excluded


def test_text_features_shapes_and_values():
    labels = pd.DataFrame({"event_id": ["a"] * 5, "sent_idx": range(5),
                           "section": ["remarks", "remarks", "answers", "answers", "closing"]})  # fmt: skip
    emb = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    probs = np.tile([[0.7, 0.1, 0.2]], (5, 1)).astype(np.float32)
    table, means, eq = bsf.text_features(["a"], labels, probs, emb, "m")
    assert means.shape == eq.shape == (1, 2, 3)
    assert np.allclose(means[0, 0], emb[:2].mean(axis=0)) and np.allclose(means[0, 1], emb[2:4].mean(axis=0))
    assert table.loc[0, "m_rem_tone"] == pytest.approx(0.6) and table.loc[0, "m_ans_frac_pos"] == 1.0


def test_audio_section_sums_and_call_check():
    from build_audio_sections import SectionSums, compare_calls

    rng = np.random.default_rng(2)
    sums = SectionSums(["a", "b"], n_layers=2, hidden=3)
    clips = [("a", "remarks"), ("a", "answers"), ("a", "answers"), ("b", "remarks"), ("b", "heading")]
    vals = rng.normal(size=(5, 2, 3))
    for (e, s), v in zip(clips, vals):
        sums.add(e, s, v)
    assert sums.counts.tolist() == [[1, 2, 0, 0], [1, 0, 0, 1]]
    assert np.allclose(sums.section_means()[0, 1], vals[1:3].mean(axis=0))
    calls = sums.call_means()
    assert np.allclose(calls, [vals[:3].mean(axis=0), vals[3:].mean(axis=0)])
    assert np.allclose(bsf.recombine(sums.section_means(), sums.counts), calls)
    assert compare_calls(calls, calls.astype(np.float16))["ok"]
    assert not compare_calls(calls, calls + 1.0)["ok"]

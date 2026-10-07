from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import AUDIO_DIM, AUDIO_LAYERS, TEXT_DIM, make_project
from data import PAST_COLS, PROSODY_COLS, SENT_COLS, load_call_data

REPO = Path(__file__).resolve().parents[2]


def test_load_call_data(tmp_path):
    ids = make_project(tmp_path)

    data = load_call_data(tmp_path)

    expected = sorted(ids[:-2])  # drops the no_price call and the audio_call_ok=False call
    assert data.table["event_id"].tolist() == expected
    assert data.groups.tolist() == data.table["ticker"].tolist()
    assert data.text_emb.shape == (len(expected), TEXT_DIM)
    assert data.audio_emb.shape == (len(expected), AUDIO_LAYERS * AUDIO_DIM)
    assert (data.audio_layers, data.audio_dim) == (AUDIO_LAYERS, AUDIO_DIM)
    assert data.text_emb.dtype == data.audio_emb.dtype == np.float32
    for cols in (PAST_COLS, SENT_COLS, PROSODY_COLS):
        assert set(cols) <= set(data.table.columns)


def test_arrays_align_with_table(tmp_path):
    make_project(tmp_path)
    data = load_call_data(tmp_path)

    # Rows were shuffled in every source; the loaded arrays must follow the table's order.
    text_calls = pd.read_csv(tmp_path / "text_embeddings/data/finbert_tone/calls.csv")
    text_raw = np.load(tmp_path / "text_embeddings/data/finbert_tone/call_emb.npy")
    audio_calls = pd.read_csv(tmp_path / "audio_embeddings/data/hubert/calls.csv")
    audio_raw = np.load(tmp_path / "audio_embeddings/data/hubert/call_emb.npy")
    for i, event_id in enumerate(data.table["event_id"]):
        t = text_calls.index[text_calls["event_id"] == event_id][0]
        a = audio_calls.index[audio_calls["event_id"] == event_id][0]
        np.testing.assert_array_equal(data.text_emb[i], text_raw[t])
        np.testing.assert_array_equal(data.audio_emb[i], audio_raw[a].astype(np.float32).ravel())
        assert data.table.loc[i, "tone"] == text_calls.loc[t, "tone"]


def test_audio_flattened_layer_major(tmp_path):
    make_project(tmp_path)
    data = load_call_data(tmp_path)
    raw = np.load(tmp_path / "audio_embeddings/data/hubert/call_emb.npy")
    calls = pd.read_csv(tmp_path / "audio_embeddings/data/hubert/calls.csv")
    row = calls.index[calls["event_id"] == data.table.loc[0, "event_id"]][0]
    layer = 6
    np.testing.assert_array_equal(
        data.audio_emb[0, layer * AUDIO_DIM : (layer + 1) * AUDIO_DIM],
        raw[row, layer].astype(np.float32),
    )


def test_targets_and_format_columns(tmp_path):
    make_project(tmp_path)
    data = load_call_data(tmp_path)
    t = data.table

    assert set(t["dir_0_1"]) <= {0, 1}
    np.testing.assert_array_equal(t["dir_0_5"], (t["car_mm_0_5"] > 0).astype(int))
    assert data.format_cols == ["fmt_11025_16", "fmt_22050_32", "fmt_44100_128", "fmt_44100_64", "stereo"]
    assert (t[data.format_cols[:-1]].sum(axis=1) == 1).all()  # exactly one format per call
    assert (t["stereo"] == (t["fmt_44100_128"] == 1)).all()


def test_text_and_audio_models_selectable(tmp_path):
    make_project(tmp_path)
    a = load_call_data(tmp_path)
    b = load_call_data(tmp_path, text_model="prosus_finbert", audio_model="wav2vec2")
    assert not np.array_equal(a.text_emb, b.text_emb)
    assert not np.array_equal(a.audio_emb, b.audio_emb)


def test_missing_source_row_raises(tmp_path):
    make_project(tmp_path)
    path = tmp_path / "text_embeddings/data/finbert_tone/calls.csv"
    calls = pd.read_csv(path)
    calls.iloc[1:].to_csv(path, index=False)  # one call now missing (and arrays misaligned)

    with pytest.raises(ValueError):
        load_call_data(tmp_path)


def test_duplicate_event_raises(tmp_path):
    make_project(tmp_path)
    path = tmp_path / "audio_embeddings/data/formats.csv"
    formats = pd.read_csv(path)
    pd.concat([formats, formats.iloc[:1]]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        load_call_data(tmp_path)


def test_nan_feature_raises(tmp_path):
    make_project(tmp_path)
    path = tmp_path / "audio_embeddings/data/praat/calls.csv"
    praat = pd.read_csv(path)
    praat.loc[0, "speech_rate"] = np.nan
    praat.to_csv(path, index=False)

    with pytest.raises(ValueError, match="speech_rate"):
        load_call_data(tmp_path)


@pytest.mark.skipif(not (REPO / "audio_embeddings/data/formats.csv").exists(), reason="real data absent")
def test_real_data():
    data = load_call_data(REPO)
    assert len(data.table) == 488
    assert data.table["ticker"].nunique() == 240
    assert data.text_emb.shape == (488, 768)
    assert data.audio_emb.shape == (488, 13 * 768)
    assert np.isfinite(data.audio_emb).all() and np.isfinite(data.text_emb).all()

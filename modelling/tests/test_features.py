import numpy as np
import pytest

from conftest import AUDIO_DIM, AUDIO_LAYERS, TEXT_DIM, make_project
from data import PROSODY_COLS, SENT_COLS, load_call_data
from features import (
    RUNGS,
    TARGETS_DIR,
    TARGETS_VOL,
    assemble,
    past_column,
    rung_blocks,
    task_of,
)


@pytest.fixture
def data(tmp_path):
    make_project(tmp_path)
    return load_call_data(tmp_path)


def test_targets_and_tasks():
    assert TARGETS_VOL == ["vol_3", "vol_7", "vol_15", "vol_30"]
    assert TARGETS_DIR == ["dir_0_1", "dir_0_5"]
    assert task_of("vol_7") == "regression"
    assert task_of("dir_0_5") == "classification"
    with pytest.raises(KeyError):
        task_of("car_mm_0_1")


def test_past_column():
    assert past_column("vol_3") == "vol_pre_3"
    assert past_column("vol_30") == "vol_pre_30"
    with pytest.raises(ValueError):
        past_column("dir_0_1")


def test_rungs():
    assert list(RUNGS) == ["B0", "R1", "R2", "R3", "R4", "R4_nopast"]
    assert rung_blocks("B0", "vol_3") == []
    assert rung_blocks("R1", "vol_3") == ["past", "format"]
    assert rung_blocks("R4", "vol_3") == ["past", "format", "text_emb", "text_sent", "audio_emb", "prosody"]
    assert rung_blocks("R4_nopast", "vol_3") == ["format", "text_emb", "text_sent", "audio_emb", "prosody"]


def test_direction_rungs_drop_past():
    for rung in RUNGS:
        assert "past" not in rung_blocks(rung, "dir_0_1")
    assert rung_blocks("R1", "dir_0_1") == ["format"]


def test_unknown_rung_or_target():
    with pytest.raises(KeyError):
        rung_blocks("R9", "vol_3")
    with pytest.raises(KeyError):
        rung_blocks("R1", "vol_99")


def test_assemble_slices_and_values(data):
    blocks = rung_blocks("R4", "vol_7")
    X, slices = assemble(data, blocks, "vol_7")

    assert list(slices) == blocks
    widths = {"past": 1, "format": len(data.format_cols), "text_emb": TEXT_DIM,
              "text_sent": len(SENT_COLS), "audio_emb": AUDIO_LAYERS * AUDIO_DIM,
              "prosody": len(PROSODY_COLS)}  # fmt: skip
    assert {b: s.stop - s.start for b, s in slices.items()} == widths
    assert X.shape == (len(data.table), sum(widths.values()))
    assert slices["past"].start == 0 and slices["prosody"].stop == X.shape[1]  # contiguous

    np.testing.assert_array_equal(X[:, slices["past"]].ravel(), data.table["vol_pre_7"])
    np.testing.assert_array_equal(X[:, slices["text_emb"]], data.text_emb)
    np.testing.assert_array_equal(X[:, slices["audio_emb"]], data.audio_emb)
    np.testing.assert_array_equal(X[:, slices["text_sent"]], data.table[SENT_COLS])
    np.testing.assert_array_equal(X[:, slices["format"]], data.table[data.format_cols])


def test_assemble_past_follows_target(data):
    X3, _ = assemble(data, ["past"], "vol_3")
    X30, _ = assemble(data, ["past"], "vol_30")
    np.testing.assert_array_equal(X3.ravel(), data.table["vol_pre_3"])
    np.testing.assert_array_equal(X30.ravel(), data.table["vol_pre_30"])


# --- round 2 ---


def test_round2_rungs():
    from features import RESIDUALIZED, ROUND2_RUNGS

    assert ROUND2_RUNGS == {"R1P": ["past", "format", "prosody"], "R3_resid": RUNGS["R3"]}
    assert RESIDUALIZED == {"R3_resid"}
    assert rung_blocks("R1P", "vol_3") == ["past", "format", "prosody"]
    assert rung_blocks("R3_resid", "vol_3") == rung_blocks("R3", "vol_3")
    assert rung_blocks("R1P", "dir_0_1") == ["format", "prosody"]


def test_round1_rungs_unchanged():
    # run_ladder.py iterates over RUNGS; round 2 must not change what round 1 runs.
    assert list(RUNGS) == ["B0", "R1", "R2", "R3", "R4", "R4_nopast"]

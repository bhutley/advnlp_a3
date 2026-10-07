import math

import pytest

from clips import (
    MAX_BAD_SHARE,
    MAX_WORDS_PER_SEC,
    call_ok,
    list_call_clips,
    list_clips,
    plausible_rate,
)


def make_call(root, folder, lines, clip_ids, speaker="Jane Doe"):
    """A call folder with TextSequence.txt and empty placeholder clips <speaker>_<p>_<s>.mp3."""
    d = root / folder
    (d / "CEO").mkdir(parents=True)
    (d / "TextSequence.txt").write_text("\n".join(lines) + "\n")
    for para, sent in clip_ids:
        (d / "CEO" / f"{speaker}_{para}_{sent}.mp3").write_bytes(b"")
    return d


def names(paths):
    return [p.name for p in paths]


def test_list_call_clips_sorts_numerically(tmp_path):
    ids = [(1, 10), (2, 1), (1, 9), (1, 1)]
    d = make_call(tmp_path, "A_20170101", ["w"] * 4, ids)

    assert names(list_call_clips(d)) == [
        "Jane Doe_1_1.mp3", "Jane Doe_1_9.mp3", "Jane Doe_1_10.mp3", "Jane Doe_2_1.mp3",
    ]  # fmt: skip


def test_list_call_clips_speaker_with_underscore(tmp_path):
    d = make_call(tmp_path, "A_20170101", ["w"], [(1, 1)], speaker="J_R Smith")
    assert names(list_call_clips(d)) == ["J_R Smith_1_1.mp3"]


def test_list_call_clips_ignores_non_clips(tmp_path):
    d = make_call(tmp_path, "A_20170101", ["w"], [(1, 1)])
    (d / "CEO" / ".ipynb_checkpoints").mkdir()
    (d / "CEO" / "notes.txt").write_text("")

    assert names(list_call_clips(d)) == ["Jane Doe_1_1.mp3"]


def test_list_call_clips_rejects_bad_name(tmp_path):
    d = make_call(tmp_path, "A_20170101", ["w"], [(1, 1)])
    (d / "CEO" / "Jane Doe_intro.mp3").write_bytes(b"")

    with pytest.raises(ValueError, match="Jane Doe_intro.mp3"):
        list_call_clips(d)


def test_list_clips(tmp_path):
    make_call(tmp_path, "Beta_20170102", ["one two three", "", "four"], [(1, 1), (1, 2)])
    make_call(tmp_path, "Alpha_20170101", ["hello there"], [(1, 1)])
    (tmp_path / ".DS_Store").write_text("")

    df = list_clips(tmp_path)

    assert list(df.columns) == ["event_id", "sent_idx", "path", "n_words"]
    assert df[["event_id", "sent_idx", "n_words"]].values.tolist() == [
        ["Alpha_20170101", 0, 2],
        ["Beta_20170102", 0, 3],
        ["Beta_20170102", 1, 1],  # the blank line is skipped, as in text_embeddings
    ]
    assert df["path"].iloc[2].endswith("Beta_20170102/CEO/Jane Doe_1_2.mp3")


def test_list_clips_count_mismatch_names_call(tmp_path):
    make_call(tmp_path, "Alpha_20170101", ["a", "b"], [(1, 1)])

    with pytest.raises(ValueError, match="Alpha_20170101"):
        list_clips(tmp_path)


def test_list_clips_limit_calls(tmp_path):
    make_call(tmp_path, "Beta_20170102", ["b"], [(1, 1)])
    make_call(tmp_path, "Alpha_20170101", ["a"], [(1, 1)])

    assert list_clips(tmp_path, limit_calls=1)["event_id"].tolist() == ["Alpha_20170101"]


# --- quality rules ---


def test_plausible_rate():
    assert MAX_WORDS_PER_SEC == 8.0
    assert plausible_rate(10, 2.0)  # 5 w/s
    assert plausible_rate(16, 2.0)  # exactly 8 w/s
    assert not plausible_rate(58, 0.073)  # a truncated clip
    assert not plausible_rate(5, math.nan)  # failed to decode
    assert not plausible_rate(5, 0.0)
    assert plausible_rate(0, 1.0)  # an empty line is not evidence of truncation


def test_call_ok():
    assert MAX_BAD_SHARE == 0.10
    assert call_ok(n_bad=4, n_clips=40)  # exactly 10%
    assert not call_ok(n_bad=5, n_clips=40)
    assert call_ok(n_bad=0, n_clips=1)

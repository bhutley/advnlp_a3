import pandas as pd
import pytest
from conftest import write_call

from dataset import check_against_saved, modelling_sample, read_call, read_dataset


def test_paragraphs_sort_numerically_not_as_text(tmp_path):
    paragraphs = [[f"p{p} s{s}" for s in range(1, 3)] for p in range(1, 12)]
    call = write_call(tmp_path, "X_20170101", paragraphs)
    df = read_call(call)
    assert df["para"].tolist() == [p for p in range(1, 12) for _ in range(2)]
    assert df["text"].tolist() == [s for para in paragraphs for s in para]
    assert df["sent_idx"].tolist() == list(range(22))


def test_gaps_in_paragraph_numbers_are_kept(tmp_path):
    call = write_call(tmp_path, "X_20170101", [["a"], ["b"], ["c"]], para_numbers=[1, 3, 7])
    assert read_call(call)["para"].tolist() == [1, 3, 7]


def test_count_mismatch_raises(tmp_path):
    call = write_call(tmp_path, "X_20170101", [["a", "b"]])
    (call / "TextSequence.txt").write_text("a\n")
    with pytest.raises(ValueError, match="2 clips but 1 transcript lines"):
        read_call(call)


def test_bad_clip_name_raises(tmp_path):
    call = write_call(tmp_path, "X_20170101", [["a"]])
    (call / "CEO" / "oops.mp3").write_bytes(b"")
    with pytest.raises(ValueError, match="unexpected clip name"):
        read_call(call)


def test_checkpoint_folders_are_ignored(release):
    assert len(read_call(release / "Gamma Co_20170303")) == 2


def test_release_check_passes_on_a_match(release, repo):
    result = check_against_saved(read_dataset(release), repo)
    assert result["ok"] and result["sentences"] == 19 and result["calls_with_several_speakers"] == 0


def test_release_check_fails_on_word_count_or_missing_rows(release, repo):
    path = repo / "audio_embeddings/data/praat/sentences.csv"
    saved = pd.read_csv(path)
    saved.loc[0, "n_words"] += 1
    saved.iloc[1:].to_csv(path, index=False)  # drop row 0 entirely
    result = check_against_saved(read_dataset(release), repo)
    assert not result["ok"] and result["only_in_release"] == 1

    saved.to_csv(path, index=False)  # row 0 back, with the wrong count
    result = check_against_saved(read_dataset(release), repo)
    assert not result["ok"] and result["word_count_mismatches"] == 1


def test_modelling_sample_needs_ok_status_and_usable_audio(repo):
    assert modelling_sample(repo) == {"Alpha Corp_20170101", "Beta Inc_20170202"}

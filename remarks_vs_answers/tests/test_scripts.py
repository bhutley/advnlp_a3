"""End to end: build_sections → make_audit → check_sections on the synthetic release."""

import json

import pandas as pd

import build_sections
import check_sections
import make_audit


def build(release, repo, out):
    return build_sections.main(["--dataset", str(release), "--root", str(repo), "--out", str(out)])


def test_build_writes_labels_without_transcript_text(release, repo, tmp_path):
    out = tmp_path / "data"
    assert build(release, repo, out) == 0
    sentences = pd.read_csv(out / "sentences.csv")
    calls = pd.read_csv(out / "calls.csv")
    assert list(sentences.columns) == build_sections.SENTENCE_COLUMNS
    assert "text" not in calls.columns and len(sentences) == 19
    assert calls.set_index("event_id")["in_sample"].tolist() == [True, True, False]
    meta = json.loads((out / "meta.json").read_text())
    assert meta["release_check"]["ok"] and meta["n_in_sample"] == 2


def test_build_refuses_a_release_that_does_not_match(release, repo, tmp_path):
    path = repo / "audio_embeddings/data/praat/sentences.csv"
    saved = pd.read_csv(path)
    saved.loc[0, "n_words"] += 1
    saved.to_csv(path, index=False)
    out = tmp_path / "data"
    assert build(release, repo, out) == 1
    assert not out.exists()


def test_audit_sheet_samples_modelling_calls_with_blank_marks(release, repo, tmp_path):
    out = tmp_path / "data"
    build(release, repo, out)
    audit = out / "audit"
    args = ["--dataset", str(release), "--calls", str(out / "calls.csv"), "--out", str(audit), "--n", "2"]
    assert make_audit.main(args) == 0
    sheet = pd.read_csv(audit / "audit_sheet.csv", keep_default_na=False)
    assert set(sheet["event_id"]) == {"Alpha Corp_20170101", "Beta Inc_20170202"}
    assert (sheet[make_audit.MARK_COLUMNS] == "").all().all()
    beta = sheet.set_index("event_id").loc["Beta Inc_20170202"]
    assert beta["before_boundary"].splitlines()[-1] == "[5 p2 remarks] We'd be happy to take your questions"
    assert beta["after_boundary"].splitlines()[0] == "[6 p3 answers] Hi, Ben"
    assert beta["end_of_para1"]  # the rule moved the boundary past paragraph 1
    assert "[8 p4 closing] Thank you all for joining us today" in beta["final_paragraph"]
    assert "## 1. Alpha Corp_20170101" in (audit / "audit_sheet.md").read_text() or "## 1. Beta" in (
        audit / "audit_sheet.md").read_text()  # fmt: skip
    assert make_audit.main(args) == 1  # refuses to overwrite marks without --force


def test_acceptance_passes_and_reports_the_audit(release, repo, tmp_path):
    out = tmp_path / "data"
    build(release, repo, out)
    make_audit.main(["--dataset", str(release), "--calls", str(out / "calls.csv"), "--out", str(out / "audit"),
                     "--n", "2"])  # fmt: skip
    sheet = pd.read_csv(out / "audit" / "audit_sheet.csv", keep_default_na=False)
    sheet["boundary_remarks_answers_ok"] = ["Y", "N"]
    sheet.to_csv(out / "audit" / "audit_sheet.csv", index=False)
    assert check_sections.main(["--data", str(out), "--root", str(repo), "--expected-calls", "2"]) == 0
    report = (out / "acceptance.md").read_text()
    assert "| remarks/answers boundary | 2 | 1 | 0.50 |" in report


def test_acceptance_fails_when_sections_are_out_of_order(release, repo, tmp_path):
    out = tmp_path / "data"
    build(release, repo, out)
    sentences = pd.read_csv(out / "sentences.csv")
    sentences.loc[(sentences.event_id == "Alpha Corp_20170101") & (sentences.sent_idx == 7), "section"] = "remarks"
    sentences.loc[(sentences.event_id == "Alpha Corp_20170101") & (sentences.sent_idx == 2), "section"] = "answers"
    sentences.to_csv(out / "sentences.csv", index=False)
    assert check_sections.main(["--data", str(out), "--root", str(repo), "--expected-calls", "2"]) == 1


def test_wilson_interval():
    lo, hi = check_sections.wilson(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97


def test_refresh_rebuilds_the_text_and_keeps_the_marks(release, repo, tmp_path):
    out = tmp_path / "data"
    build(release, repo, out)
    args = ["--dataset", str(release), "--calls", str(out / "calls.csv"), "--out", str(out / "audit"), "--n", "2"]
    make_audit.main(args)
    path = out / "audit" / "audit_sheet.csv"
    sheet = pd.read_csv(path, keep_default_na=False)
    sheet["boundary_remarks_answers_ok"], sheet["notes"] = ["Y", "N"], ["fine", "check 7"]
    sheet = sheet.drop(columns="before_final")  # as if made by an older version
    sheet.to_csv(path, index=False)
    assert make_audit.main(args + ["--refresh"]) == 0
    new = pd.read_csv(path, keep_default_na=False)
    assert new["event_id"].tolist() == sheet["event_id"].tolist()
    assert new["boundary_remarks_answers_ok"].tolist() == ["Y", "N"] and new["notes"].tolist() == ["fine", "check 7"]
    beta = new.set_index("event_id").loc["Beta Inc_20170202"]
    assert beta["before_final"].splitlines() == ["[6 p3 answers] Hi, Ben", "[7 p3 answers] Yes, pricing held up"]
    assert "Before the final paragraph" in (out / "audit" / "audit_sheet.md").read_text()

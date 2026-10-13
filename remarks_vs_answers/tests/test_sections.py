import pandas as pd
import pytest

from sections import label_call, label_sections


def call(*paragraphs):
    text = [s for p in paragraphs for s in p]
    para = [k for k, p in enumerate(paragraphs, start=1) for _ in p]
    return label_call(text, para)


def test_default_rule_ends_remarks_with_paragraph_one():
    labels, summary = call(["Welcome", "Sales rose", "Over to Inge"], ["Good morning, Andy", "Demand held"])
    assert labels == ["remarks"] * 3 + ["answers"] * 2
    assert summary == {"rule": "para1", "boundary": 2}


def test_single_paragraph_call_is_all_remarks():
    labels, summary = call(["Welcome", "Sales rose"])
    assert labels == ["remarks", "remarks"] and summary["rule"] == "para1"


@pytest.mark.parametrize("cue", [
    "Now we'd be happy to take your questions",
    "With that, operator, please open the line for questions",
    "And with that, Jennifer, we're ready for questions",
])  # fmt: skip
def test_handover_in_paragraph_two_extends_the_remarks(cue):
    labels, summary = call(["Welcome", "Over to Tom"], ["Thanks, Tom", "In summary", cue], ["Hi, Ben", "Yes"])
    assert summary == {"rule": "handover", "boundary": 4}
    assert labels == ["remarks"] * 5 + ["answers"] * 2


def test_heading_extends_the_remarks_and_is_labelled_heading():
    labels, summary = call(["Welcome", "Over to Tom"], ["Thanks, Tom", "Question-and-Answer Session"], ["Hi, Ben"])
    assert summary == {"rule": "handover", "boundary": 3}
    assert labels == ["remarks", "remarks", "remarks", "heading", "answers"]


def test_heading_inside_paragraph_one_keeps_the_default():
    labels, summary = call(["Welcome", "Question-and-Answer Session"], ["Hi, Ben", "Yes"])
    assert summary == {"rule": "para1", "boundary": 1}
    assert labels == ["remarks", "heading", "answers", "answers"]


def test_handover_phrase_after_paragraph_two_is_ignored():
    _, summary = call(["Welcome"], ["Hi, Ben"], ["Happy to take your questions offline"])
    assert summary["rule"] == "para1"


def test_handback_to_the_named_colleague_extends_to_paragraph_two():
    labels, summary = call(["Welcome", "Let me turn it over to Heath"],
                           ["Thanks, Heath", "Our guidance is unchanged"], ["Wamsi, let me take that"])  # fmt: skip
    assert summary == {"rule": "handback", "boundary": 3}
    assert labels == ["remarks"] * 4 + ["answers"]


def test_thanking_someone_else_is_an_answer():
    _, summary = call(["Welcome", "Let me turn it over to Heath"], ["Thanks, Andy", "Good question"])
    assert summary["rule"] == "para1"


def test_thanking_the_operator_is_not_a_handback():
    _, summary = call(["Welcome", "Over to the operator"], ["Thanks, Operator", "Sure"])
    assert summary["rule"] == "para1"


def test_short_final_thank_you_paragraph_is_closing():
    labels, _ = call(["Welcome"], ["Hi, Ben", "Yes"], ["Thank you all for joining us today", "Take care"])
    assert labels == ["remarks", "answers", "answers", "closing", "closing"]


def test_long_final_paragraph_stays_an_answer():
    answer = ["Hi, Ben"] + ["more detail"] * 12 + ["Thanks for your questions"]
    labels, _ = call(["Welcome"], answer)
    assert set(labels[1:]) == {"answers"}


def test_final_paragraph_inside_the_remarks_is_not_closing():
    labels, _ = call(["Welcome", "Over to Tom"], ["Thanks, Tom", "Thank you for joining us"])
    assert labels == ["remarks"] * 4


def test_bad_input_raises():
    with pytest.raises(ValueError):
        label_call([], [])
    with pytest.raises(ValueError):
        label_call(["a", "b"], [1])


def test_label_sections_on_the_synthetic_release(release):
    from dataset import read_dataset

    labelled, calls = label_sections(read_dataset(release))
    c = calls.set_index("event_id")
    assert c.loc["Alpha Corp_20170101", ["rule", "boundary", "n_remarks", "n_answers"]].tolist() == ["para1", 2, 3, 5]
    assert c.loc["Beta Inc_20170202", ["rule", "boundary", "n_remarks", "n_answers", "n_closing"]].tolist() == [
        "handover", 5, 6, 2, 1]  # fmt: skip
    assert c.loc["Gamma Co_20170303", ["n_remarks", "n_answers"]].tolist() == [2, 0]
    assert len(labelled) == c["n_sentences"].sum() == 19
    pd.testing.assert_series_equal(
        labelled.groupby("event_id")["section"].size(), c["n_sentences"], check_names=False)  # fmt: skip

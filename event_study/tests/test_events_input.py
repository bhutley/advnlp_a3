from datetime import date

import pytest

from events_input import GREETING_CHARS, classify_timing, parse_event_id, scan_dataset

# --- classify_timing ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Thanks, Inge, and good morning, everyone", "pre"),
        ("Thank you, operator. Good afternoon and welcome", "post"),
        ("Good evening, everyone", "post"),
        ("Thank you, operator. Welcome to our call", "unknown"),
        ("GOOD MORNING", "pre"),
        ("", "unknown"),
    ],
)
def test_classify_timing(text, expected):
    assert classify_timing(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello. We released our results this morning", "pre"),
        ("Hello. Our press release went out this afternoon", "post"),
        ("Hello. As announced this evening", "post"),
    ],
)
def test_classify_timing_falls_back_to_release_phrase(text, expected):
    assert classify_timing(text) == expected


def test_classify_timing_greeting_wins_over_release_phrase():
    assert classify_timing("Good afternoon. We announced a deal this morning") == "post"


def test_classify_timing_morning_wins_over_afternoon():
    # e.g. a speaker greeting listeners in another time zone
    assert classify_timing("Good afternoon in Europe, and good morning here") == "pre"


def test_classify_timing_ignores_greeting_beyond_limit():
    text = "x" * GREETING_CHARS + " good morning"
    assert classify_timing(text) == "unknown"


def test_classify_timing_greeting_split_by_newline():
    # TextSequence.txt has one sentence per line; a greeting never spans lines,
    # but whitespace between the words should not break a match within a line.
    assert classify_timing("good  morning") == "pre"


# --- parse_event_id ---


def test_parse_event_id():
    assert parse_event_id("3M Company_20170425") == ("3M Company", date(2017, 4, 25))


def test_parse_event_id_underscore_in_name():
    assert parse_event_id("Foo_Bar Inc_20171026") == ("Foo_Bar Inc", date(2017, 10, 26))


@pytest.mark.parametrize("name", ["3M Company", "3M Company_2017", "3M Company_20171332"])
def test_parse_event_id_malformed(name):
    with pytest.raises(ValueError):
        parse_event_id(name)


# --- scan_dataset ---


def make_call(root, folder, text):
    d = root / folder
    d.mkdir()
    if text is not None:
        (d / "TextSequence.txt").write_bytes(text)


def test_scan_dataset(tmp_path):
    make_call(tmp_path, "3M Company_20170425", b"Thanks, and good morning, everyone\n")
    make_call(tmp_path, "Amazon.com Inc._20171026", b"Good afternoon\n")
    make_call(tmp_path, "Bad Bytes Co_20170501", b"\xff\xfe Hello and good morning\n")
    make_call(tmp_path, "No Text Inc_20170601", None)
    (tmp_path / ".DS_Store").write_text("")

    df = scan_dataset(tmp_path)

    assert list(df.columns) == ["event_id", "company", "call_date", "timing"]
    assert df.to_dict("records") == [
        {
            "event_id": "3M Company_20170425",
            "company": "3M Company",
            "call_date": date(2017, 4, 25),
            "timing": "pre",
        },
        {
            "event_id": "Amazon.com Inc._20171026",
            "company": "Amazon.com Inc.",
            "call_date": date(2017, 10, 26),
            "timing": "post",
        },
        {
            "event_id": "Bad Bytes Co_20170501",
            "company": "Bad Bytes Co",
            "call_date": date(2017, 5, 1),
            "timing": "pre",
        },
        {
            "event_id": "No Text Inc_20170601",
            "company": "No Text Inc",
            "call_date": date(2017, 6, 1),
            "timing": "unknown",
        },
    ]

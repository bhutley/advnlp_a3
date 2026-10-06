from transcripts import read_call, read_sentences


def make_call(root, folder, content: bytes | None):
    d = root / folder
    d.mkdir(parents=True)
    if content is not None:
        (d / "TextSequence.txt").write_bytes(content)


def test_read_call_skips_blank_lines(tmp_path):
    path = tmp_path / "TextSequence.txt"
    path.write_bytes(b"  First sentence \n\n   \nSecond sentence\n")
    assert read_call(path) == ["First sentence", "Second sentence"]


def test_read_call_invalid_utf8(tmp_path):
    path = tmp_path / "TextSequence.txt"
    path.write_bytes(b"Good \xff morning\n")
    [line] = read_call(path)
    assert line.startswith("Good") and line.endswith("morning")


def test_read_sentences(tmp_path):
    make_call(tmp_path, "Beta Inc_20170426", b"One\n\nTwo\nThree\n")
    make_call(tmp_path, "Alpha Corp_20170425", b"Hello\n")
    make_call(tmp_path, "Empty Co_20170501", b"\n  \n")
    make_call(tmp_path, "No Text Ltd_20170601", None)
    (tmp_path / ".DS_Store").write_text("")

    event_ids, sentences = read_sentences(tmp_path)

    assert event_ids == [
        "Alpha Corp_20170425",
        "Beta Inc_20170426",
        "Empty Co_20170501",
        "No Text Ltd_20170601",
    ]
    assert list(sentences.columns) == ["event_id", "sent_idx", "text"]
    assert sentences.values.tolist() == [
        ["Alpha Corp_20170425", 0, "Hello"],
        ["Beta Inc_20170426", 0, "One"],
        ["Beta Inc_20170426", 1, "Two"],  # the blank line does not use up an index
        ["Beta Inc_20170426", 2, "Three"],
    ]


def test_read_sentences_limit_calls(tmp_path):
    make_call(tmp_path, "Beta Inc_20170426", b"One\n")
    make_call(tmp_path, "Alpha Corp_20170425", b"Hello\n")

    event_ids, sentences = read_sentences(tmp_path, limit_calls=1)

    assert event_ids == ["Alpha Corp_20170425"]
    assert sentences["event_id"].unique().tolist() == ["Alpha Corp_20170425"]


def test_read_sentences_no_calls(tmp_path):
    event_ids, sentences = read_sentences(tmp_path)
    assert event_ids == [] and sentences.empty
    assert list(sentences.columns) == ["event_id", "sent_idx", "text"]

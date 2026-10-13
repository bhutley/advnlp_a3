"""A synthetic ACL19-style release and repository, so tests need no real data."""

import pandas as pd
import pytest


def write_call(root, event_id, paragraphs, speaker="Jane Doe", para_numbers=None):
    """A call folder: TextSequence.txt plus one empty CEO/<speaker>_<para>_<sent>.mp3 per
    line. paragraphs is a list of lists of sentences; para_numbers defaults to 1..n."""
    call = root / event_id
    (call / "CEO").mkdir(parents=True)
    numbers = para_numbers or list(range(1, len(paragraphs) + 1))
    lines = []
    for number, sentences in zip(numbers, paragraphs):
        for k, sentence in enumerate(sentences, start=1):
            (call / "CEO" / f"{speaker}_{number}_{k}.mp3").write_bytes(b"")
            lines.append(sentence)
    (call / "TextSequence.txt").write_text("\n\n".join(lines) + "\n")
    return call


CALLS = {
    "Alpha Corp_20170101": [
        ["Good morning, everyone", "Revenue grew 5%", "Inge will now cover the outlook"],
        ["Good morning, Andy", "Thanks for the question", "Demand was strong"],
        ["Hi, Joe", "Margins were flat"],
    ],
    "Beta Inc_20170202": [
        ["Thanks, operator", "We had a record quarter", "Now over to Tom"],
        ["Thanks, Tom", "In summary, a strong year", "We'd be happy to take your questions"],
        ["Hi, Ben", "Yes, pricing held up"],
        ["Thank you all for joining us today"],
    ],
    "Gamma Co_20170303": [
        ["Thanks, Steve", "Our results were solid"],
    ],
}


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "ACL19_Release"
    for event_id, paragraphs in CALLS.items():
        write_call(root, event_id, paragraphs)
    (root / "Gamma Co_20170303" / "CEO" / ".ipynb_checkpoints").mkdir()
    return root


@pytest.fixture
def repo(tmp_path, release):
    """A repository root with the saved tables dataset.py reads, matching `release`."""
    from dataset import read_dataset

    root = tmp_path / "repo"
    sentences = read_dataset(release)
    praat = root / "audio_embeddings/data/praat"
    praat.mkdir(parents=True)
    sentences[["event_id", "sent_idx", "n_words"]].to_csv(praat / "sentences.csv", index=False)
    hubert = root / "audio_embeddings/data/hubert"
    hubert.mkdir(parents=True)
    ids = list(CALLS)
    pd.DataFrame({"event_id": ids, "audio_call_ok": [True, True, False]}).to_csv(hubert / "calls.csv", index=False)
    events = root / "event_study/data"
    events.mkdir(parents=True)
    pd.DataFrame({"event_id": ids, "status": ["ok", "ok", "ok"]}).to_csv(events / "events.csv", index=False)
    return root

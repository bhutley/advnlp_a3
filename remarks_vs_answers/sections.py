"""Label each sentence of a call: prepared remarks, answers, closing remarks or heading.

The data keep one executive per call, and the clips are numbered by paragraph: one
paragraph per uninterrupted turn of that executive. Prepared remarks come first, so the
boundary between remarks and answers always falls between sentences in paragraph order.
The rules, each found by reading the transcripts (see README.md):

1. para1     Default: the first paragraph is prepared remarks, everything after it
             answers. No answer phrase ("thanks for the question", "Good morning,
             <Name>") occurs in any first paragraph of the 488 modelling calls.
2. handover  The executive often speaks again before Q&A, e.g. after the CFO. If the
             first two paragraphs contain a handover to questions ("we'll be happy to
             take your questions") or the transcript heading "Question-and-Answer
             Session", the remarks run to the last such sentence.
3. handback  With no handover phrase, a second scripted segment shows as paragraph 1
             ending with a hand-off to a named colleague ("turn it over to Heath") and
             paragraph 2 opening by thanking them ("Thanks, Heath"). The remarks then
             run to the end of paragraph 2.

Two kinds of sentence are set aside rather than counted as answers:
- heading: the "Question-and-Answer Session" line, a transcript heading with a clip.
- closing: a short final paragraph after Q&A that thanks listeners for joining.
"""

import re

import pandas as pd

SECTIONS = ("remarks", "answers", "closing", "heading")
RULES = ("para1", "handover", "handback")

HEADING_RE = re.compile(r"(?i)^\s*question[- ]and[- ]answer session\s*$")
HANDOVER_RE = re.compile(
    r"(?i)(take|taking|answer|address) (your|any|some) questions"
    r"|(open|opening) (it |the (call|line|lines|queue|floor) )?(up )?(for|to) (your )?questions"
    r"|ready (for|to take) (your )?questions"
    r"|(go|move) (on )?to (the )?(q&a|questions)"
    r"|begin (the )?(q&a|question)"
)
# "Thanks, Heath" / "Okay, thank you, Mike": group 1 is the name, which must be capitalised.
HANDBACK_RE = re.compile(r"^(?i:(?:okay,? |well,? |great,? |and )?(?:thanks?|thank you)(?: very much)?),? ([A-Z][a-z]+)\b")
HANDBACK_LOOKBACK = 2  # last sentences of paragraph 1 searched for the colleague's name
CLOSING_RE = re.compile(
    r"(?i)(thank you|thanks) (all |everyone |everybody )?(again )?"
    r"(for (joining|your (time|interest|participation|questions))|for being with us)"
    r"|in closing|closing (remarks|comments)"
    r"|look forward to (speaking|talking|seeing|updating)"
)
CLOSING_MAX_SENTENCES = 12  # a longer final paragraph is an answer that ends politely


def _matches(pattern: re.Pattern, text: str) -> bool:
    return bool(pattern.search(text))


def label_call(text: list[str], para: list[int]) -> tuple[list[str], dict]:
    """Sections for one call's sentences (in sent_idx order), and a summary.

    The summary has 'rule' (one of RULES) and 'boundary', the sent_idx of the last
    prepared-remarks sentence.
    """
    if not text or len(text) != len(para):
        raise ValueError("need one paragraph number per sentence, and at least one sentence")
    order = sorted(set(para))
    rank = [order.index(p) for p in para]
    n = len(text)
    p1_end = max(i for i in range(n) if rank[i] == 0)

    rule, boundary = "para1", p1_end
    cues = [i for i in range(n) if rank[i] <= 1
            and (_matches(HEADING_RE, text[i]) or _matches(HANDOVER_RE, text[i]))]  # fmt: skip
    if cues and max(cues) > p1_end:
        rule, boundary = "handover", max(cues)
    elif len(order) > 1:
        m = HANDBACK_RE.match(text[p1_end + 1])
        tail = " ".join(text[max(0, p1_end - HANDBACK_LOOKBACK + 1) : p1_end + 1])
        if m and re.search(rf"\b{re.escape(m[1])}\b", tail):
            rule, boundary = "handback", max(i for i in range(n) if rank[i] == 1)

    labels = ["remarks" if i <= boundary else "answers" for i in range(n)]
    last = [i for i in range(n) if para[i] == order[-1]]
    if (last[0] > boundary and len(last) <= CLOSING_MAX_SENTENCES
            and any(_matches(CLOSING_RE, text[i]) for i in last)):  # fmt: skip
        for i in last:
            labels[i] = "closing"
    for i in range(n):
        if _matches(HEADING_RE, text[i]):
            labels[i] = "heading"
    return labels, {"rule": rule, "boundary": boundary}


def label_sections(sentences: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sentences with a 'section' column, one summary row per call).

    sentences needs event_id, sent_idx, para and text, one row per sentence.
    """
    sentences = sentences.sort_values(["event_id", "sent_idx"]).reset_index(drop=True)
    labels, calls = [], []
    for event_id, g in sentences.groupby("event_id", sort=False):
        call_labels, summary = label_call(g["text"].tolist(), g["para"].tolist())
        labels += call_labels
        counts = pd.Series(call_labels).value_counts()
        calls.append({"event_id": event_id, "n_sentences": len(g), "n_paragraphs": g["para"].nunique(),
                      **summary} | {f"n_{s}": int(counts.get(s, 0)) for s in SECTIONS})  # fmt: skip
    return sentences.assign(section=labels), pd.DataFrame(calls)

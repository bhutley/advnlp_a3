"""Round-3 section features, and the must-pass checks that they are round 1's, split.

Inputs (all produced before this step):
- data/sentences.csv, data/calls.csv            the remarks/answers split
- data/features/text/<model>/                    text_embeddings rerun: sentences.csv
                                                 (probabilities) + sentence_emb.npy
- data/features/hubert/                          build_audio_sections.py output
- ../audio_embeddings/data/praat/sentences.csv   Praat features per sentence (committed)

Writes data/features/round3/:
- calls.csv           one row per round-3 call: counts, log sizes, prosody and sentiment
                      per section (and equal-length versions)
- text_<model>.npy    (n_calls, 2, 768) mean sentence vector of the remarks and answers;
  text_<model>_eq.npy the same from equal-length subsets
- hubert.npy          (n_calls, 2, 13, 768) float16, remarks and answers
- meta.json, acceptance.md

The sample (fixed by the round-3 design): the 488 modelling calls with at least 10
audio_ok sentences in both the remarks and the answers. Sizes count all of a section's
sentences; prosody and audio use its audio_ok sentences; text uses all its sentences.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sections import SECTIONS

MIN_OK = 10  # audio_ok sentences required in each of remarks and answers
TEXT_MODELS = ("finbert_tone", "prosus_finbert")
PROSODY = ["f0_sd_st", "speech_rate", "intensity_sd_db", "voiced_frac"]  # round 1's prosody block
PROBS = ["p_pos", "p_neg", "p_neu"]
SENT_COLS = ["tone", "frac_pos", "frac_neg", "p_pos", "p_neg", "p_neu"]  # round 1's text_sent
PARTS = ("rem", "ans")
PART_SECTION = {"rem": "remarks", "ans": "answers"}
EXPECTED = {"calls": 457, "firms": 232}
EMB_RTOL, EMB_ATOL = 2e-3, 1e-2  # float16 storage plus GPU differences
PROB_ATOL = 1e-3


def sentiment(probs: np.ndarray) -> dict:
    """Round 1's six summaries of (n, 3) [p_pos, p_neg, p_neu] (text_embeddings/embed.py)."""
    top = probs.argmax(axis=1)
    return {"tone": float((probs[:, 0] - probs[:, 1]).mean()), "frac_pos": float((top == 0).mean()),
            "frac_neg": float((top == 1).mean())} | dict(zip(PROBS, probs.mean(axis=0).tolist()))  # fmt: skip


def equal_subsets(rem: np.ndarray, ans: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """The same number of rows from each section: all of the smaller one, a random
    subset (kept in order) of the larger."""
    k = min(len(rem), len(ans))
    pick = lambda idx: np.sort(rng.choice(idx, size=k, replace=False)) if len(idx) > k else idx  # noqa: E731
    return pick(rem), pick(ans)


def round3_sample(sections: pd.DataFrame, calls: pd.DataFrame, praat: pd.DataFrame) -> list[str]:
    """Sorted event_ids: modelling calls with >= MIN_OK audio_ok sentences in each section."""
    s = sections.merge(praat[["event_id", "sent_idx", "audio_ok"]], on=["event_id", "sent_idx"])
    s = s[s["audio_ok"].astype(bool) & s["event_id"].isin(calls.loc[calls["in_sample"], "event_id"])]
    n = s.pivot_table(index="event_id", columns="section", values="sent_idx", aggfunc="size", fill_value=0)
    n = n.reindex(columns=["remarks", "answers"], fill_value=0)
    return sorted(n.index[(n["remarks"] >= MIN_OK) & (n["answers"] >= MIN_OK)])


def text_features(event_ids, labels: pd.DataFrame, probs: np.ndarray, emb: np.ndarray, model: str):
    """(features table, (n, 2, dim) means, (n, 2, dim) equal-length means).

    labels: event_id, sent_idx, section, row-aligned with probs and emb.
    """
    by_call = labels.groupby("event_id").indices
    rows, means, eq = [], np.empty((len(event_ids), 2, emb.shape[1])), np.empty((len(event_ids), 2, emb.shape[1]))
    for i, event_id in enumerate(event_ids):
        idx = by_call[event_id]
        sec = labels["section"].to_numpy()[idx]
        part_idx = {p: idx[sec == PART_SECTION[p]] for p in PARTS}
        eq_idx = dict(zip(PARTS, equal_subsets(part_idx["rem"], part_idx["ans"], np.random.default_rng([0, i]))))
        row = {"event_id": event_id}
        for k, p in enumerate(PARTS):
            means[i, k], eq[i, k] = emb[part_idx[p]].mean(axis=0), emb[eq_idx[p]].mean(axis=0)
            row |= {f"{model}_{p}_{c}": v for c, v in sentiment(probs[part_idx[p]]).items()}
            row |= {f"{model}_eq_{p}_{c}": v for c, v in sentiment(probs[eq_idx[p]]).items()}
        rows.append(row)
    return pd.DataFrame(rows), means.astype(np.float32), eq.astype(np.float32)


def prosody_features(event_ids, praat: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per call and section: NaN-aware means of PROSODY over audio_ok sentences, the
    equal-length version, the shift (answers − remarks) and the counts."""
    p = praat.merge(labels, on=["event_id", "sent_idx"])
    p = p[p["audio_ok"].astype(bool)]
    by_call = dict(tuple(p.groupby("event_id")))
    rows = []
    for i, event_id in enumerate(event_ids):
        g = by_call[event_id]
        part = {q: g[g["section"] == PART_SECTION[q]] for q in PARTS}
        eq = dict(zip(PARTS, equal_subsets(part["rem"].index.to_numpy(), part["ans"].index.to_numpy(),
                                           np.random.default_rng([1, i]))))  # fmt: skip
        row = {"event_id": event_id} | {f"n_{q}_ok": len(part[q]) for q in PARTS}
        for q in PARTS:
            row |= {f"prosody_{q}_{c}": v for c, v in part[q][PROSODY].mean().items()}
            row |= {f"prosody_eq_{q}_{c}": v for c, v in g.loc[eq[q], PROSODY].mean().items()}
        row |= {f"shift_{c}": row[f"prosody_ans_{c}"] - row[f"prosody_rem_{c}"] for c in PROSODY}
        rows.append(row)
    return pd.DataFrame(rows)


def recombine(means: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Call means from (n, sections, ...) section means and (n, sections) counts; empty
    sections (NaN means, zero count) contribute nothing."""
    w = counts.reshape(counts.shape + (1,) * (means.ndim - 2)).astype(np.float64)
    total = np.where(w > 0, np.nan_to_num(means.astype(np.float64)) * w, 0.0).sum(axis=1)
    return total / counts.sum(axis=1).reshape((-1,) + (1,) * (means.ndim - 2))


def section_means(labels: pd.DataFrame, values: np.ndarray, event_ids) -> tuple[np.ndarray, np.ndarray]:
    """(n_calls, 4, dim) means over each section's rows of values, and (n_calls, 4) counts."""
    sec = labels["section"].map({s: k for k, s in enumerate(SECTIONS)}).to_numpy()
    call = labels["event_id"].map({e: i for i, e in enumerate(event_ids)}).to_numpy()
    sums = np.zeros((len(event_ids), len(SECTIONS), values.shape[1]))
    counts = np.zeros((len(event_ids), len(SECTIONS)), dtype=int)
    np.add.at(sums, (call, sec), values)
    np.add.at(counts, (call, sec), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return sums / counts[:, :, None], counts


def check_text(model: str, ft: Path, root: Path, labels_all: pd.DataFrame) -> dict:
    """The rerun against text_embeddings' committed outputs, for all 572 calls."""
    new = pd.read_csv(ft / model / "sentences.csv")
    stage = root / "text_embeddings/data" / model
    old = pd.read_csv(stage / "sentences.csv")
    m = new.merge(old, on=["event_id", "sent_idx"], suffixes=("", "_old"))
    emb = np.load(ft / model / "sentence_emb.npy").astype(np.float64)
    labels = new[["event_id", "sent_idx"]].merge(labels_all, on=["event_id", "sent_idx"], how="left")
    old_calls = pd.read_csv(stage / "calls.csv")
    ids = old_calls["event_id"].tolist()
    means, counts = section_means(labels, emb, ids)
    vec_ok = np.allclose(recombine(means, counts), np.load(stage / "call_emb.npy"), rtol=EMB_RTOL, atol=EMB_ATOL)
    pm, _ = section_means(labels, new[PROBS].to_numpy(), ids)
    p_call = recombine(pm, counts)
    prob_diff = float(np.abs(p_call - old_calls[PROBS].to_numpy()).max())
    return {
        "sentences_matched": len(m), "sentences_expected": len(old),
        "max_sentence_prob_diff": float(np.abs(m[PROBS].to_numpy() - m[[f"{c}_old" for c in PROBS]].to_numpy()).max()),
        "call_vectors_recombine": bool(vec_ok), "max_call_prob_diff": prob_diff,
        "ok": bool(len(m) == len(old) == len(new) and vec_ok and prob_diff <= PROB_ATOL
                   and np.abs(m[PROBS].to_numpy() - m[[f"{c}_old" for c in PROBS]].to_numpy()).max() <= EMB_ATOL),
    }  # fmt: skip


def check_prosody(praat: pd.DataFrame, labels_all: pd.DataFrame, root: Path) -> dict:
    """NaN-aware section means recombine to the committed Praat call features."""
    p = praat[praat["audio_ok"].astype(bool)].merge(labels_all, on=["event_id", "sent_idx"])
    committed = pd.read_csv(root / "audio_embeddings/data/praat/calls.csv").set_index("event_id")
    worst = 0.0
    for c in PROSODY:
        g = p.groupby(["event_id", "section"])[c].agg(["sum", "count"]).groupby(level=0).sum()
        recombined = g["sum"] / g["count"]
        ref = committed.loc[recombined.index, c]
        worst = max(worst, float((np.abs(recombined - ref) / np.maximum(1.0, np.abs(ref))).max()))
    return {"max_rel_diff": worst, "ok": worst < 1e-9}


def check_audio(hub: Path, root: Path) -> dict:
    meta = json.loads((hub / "meta.json").read_text())
    counts = pd.read_csv(hub / "section_counts.csv")
    means = np.load(hub / "section_emb.npy")
    stage = root / "audio_embeddings/data/hubert"
    ids = pd.read_csv(stage / "calls.csv")["event_id"].tolist()
    committed = np.load(stage / "call_emb.npy")[[ids.index(e) for e in counts["event_id"]]].astype(np.float64)
    calls = recombine(means, counts[list(SECTIONS)].to_numpy())
    have = ~np.isnan(calls).any(axis=(1, 2))
    stored_ok = bool(np.allclose(calls[have], committed[have], rtol=EMB_RTOL, atol=EMB_ATOL))
    return {"build_check": meta["check"]["ok"], "full_run": meta["limit_calls"] is None,
            "stored_means_recombine": stored_ok,
            "ok": bool(meta["check"]["ok"] and meta["limit_calls"] is None and stored_ok)}  # fmt: skip


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path(".."))
    p.add_argument("--data", type=Path, default=Path("data"))
    args = p.parse_args(argv)
    ft, out = args.data / "features" / "text", args.data / "features" / "round3"

    labels_all = pd.read_csv(args.data / "sentences.csv")[["event_id", "sent_idx", "section"]]
    calls = pd.read_csv(args.data / "calls.csv")
    praat = pd.read_csv(args.root / "audio_embeddings/data/praat/sentences.csv")
    events = pd.read_csv(args.root / "event_study/data/events.csv").set_index("event_id")
    ids = round3_sample(labels_all, calls, praat)

    checks = {f"text {m}": check_text(m, ft, args.root, labels_all) for m in TEXT_MODELS}
    checks["prosody"] = check_prosody(praat, labels_all, args.root)
    checks["hubert"] = check_audio(args.data / "features" / "hubert", args.root)
    checks["sample"] = {"calls": len(ids), "firms": int(events.loc[ids, "ticker"].nunique())}
    checks["sample"]["ok"] = checks["sample"]["calls"] == EXPECTED["calls"] and checks["sample"]["firms"] == EXPECTED["firms"]

    sizes = labels_all[labels_all["event_id"].isin(ids) & labels_all["section"].isin(["remarks", "answers"])]
    sizes = sizes.pivot_table(index="event_id", columns="section", values="sent_idx", aggfunc="size").loc[ids]
    table = pd.DataFrame({"event_id": ids, "ticker": events.loc[ids, "ticker"].to_numpy(),
                          "n_rem": sizes["remarks"].to_numpy(), "n_ans": sizes["answers"].to_numpy()})  # fmt: skip
    table["size_rem"], table["size_ans"] = np.log(table["n_rem"]), np.log(table["n_ans"])
    table = table.merge(prosody_features(ids, praat, labels_all), on="event_id")

    out.mkdir(parents=True, exist_ok=True)
    for m in TEXT_MODELS:
        sent = pd.read_csv(ft / m / "sentences.csv")
        labels = sent[["event_id", "sent_idx"]].merge(labels_all, on=["event_id", "sent_idx"], how="left")
        emb = np.load(ft / m / "sentence_emb.npy").astype(np.float32)
        keep = labels["event_id"].isin(ids).to_numpy()
        feats, means, eq = text_features(ids, labels[keep].reset_index(drop=True), sent.loc[keep, PROBS].to_numpy(),
                                         emb[keep], m)  # fmt: skip
        table = table.merge(feats, on="event_id")
        np.save(out / f"text_{m}.npy", means)
        np.save(out / f"text_{m}_eq.npy", eq)
    hub = args.data / "features" / "hubert"
    hub_ids = pd.read_csv(hub / "section_counts.csv")["event_id"].tolist()
    rows = [hub_ids.index(e) for e in ids]
    np.save(out / "hubert.npy", np.load(hub / "section_emb.npy")[rows][:, [SECTIONS.index("remarks"), SECTIONS.index("answers")]])

    numeric = table.drop(columns=["event_id", "ticker"])
    checks["no_missing_values"] = {"ok": bool(numeric.notna().all().all()),
                                   "columns_with_nan": [c for c in numeric if numeric[c].isna().any()]}  # fmt: skip
    table.to_csv(out / "calls.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"min_ok": MIN_OK, "n_calls": len(ids), "checks": checks}, indent=2) + "\n")
    lines = ["# Round 3 features: acceptance checks", "", "| check | pass | details |", "|---|---|---|"]
    lines += [f"| {k} | {'yes' if v['ok'] else 'no'} | " + ", ".join(f"{a}={b}" for a, b in v.items() if a != "ok") + " |"
              for k, v in checks.items()]  # fmt: skip
    (out / "acceptance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(v["ok"] for v in checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

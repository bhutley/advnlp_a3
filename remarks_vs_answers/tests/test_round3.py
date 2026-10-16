"""Round-3 modelling on synthetic data: blocks, planted signals, leakage, comparisons, CLI."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import round3 as r3
import run_round3
from cv import inner_cv, outer_splits

FAST = {"ks": (8, None), "layers": (3, 6), "n_jobs": 1, "alphas": (0.1, 1.0, 10.0, 100.0)}
TEXT_DIM, AUDIO_LAYERS, AUDIO_DIM = 12, 13, 6


def make_data(signal_in: str | None = "ans", n_firms: int = 40, calls_per_firm: int = 3, seed: int = 0):
    """Synthetic Round3Data: vol_n = 0.5 * past + (sum of the first 3 dims of that
    section's text vector) + noise; everything else is noise."""
    rng = np.random.default_rng(seed)
    n = n_firms * calls_per_firm
    table = pd.DataFrame({"event_id": [f"E{i:04d}" for i in range(n)],
                          "ticker": np.repeat([f"F{f}" for f in range(n_firms)], calls_per_firm)})  # fmt: skip
    arrays = {}
    for key in ("text_emb_rem", "text_emb_ans", "text_emb", "text_emb_rem@eq", "text_emb_ans@eq",
                "text_emb_rem@prosus", "text_emb_ans@prosus"):  # fmt: skip
        arrays[key] = rng.normal(size=(n, TEXT_DIM)).astype(np.float32)
    for key in ("audio_emb_rem", "audio_emb_ans"):
        arrays[key] = rng.normal(size=(n, AUDIO_LAYERS * AUDIO_DIM)).astype(np.float32)
    planted = arrays[f"text_emb_{signal_in}"][:, :3].sum(axis=1) if signal_in else np.zeros(n)
    for h in r3.VOL_HORIZONS:
        table[f"vol_pre_{h}"] = rng.normal(-4.6, 0.5, n)
        table[f"vol_{h}"] = 0.5 * table[f"vol_pre_{h}"] + planted + rng.normal(0, 0.3, n)
    for w in ("0_1", "0_5", "2_5"):
        table[f"dir_{w}"] = (planted + rng.normal(0, 0.5, n) > 0).astype(int)
    table["fmt_a"] = rng.integers(0, 2, n)
    table["fmt_b"] = 1 - table["fmt_a"]
    table["size_rem"], table["size_ans"] = rng.normal(4, 0.3, n), rng.normal(4, 0.3, n)
    cols = [f"call_{c}" for c in r3.SENT_COLS]
    for model in ("finbert_tone", "prosus_finbert"):
        for eq in ("", "eq_"):
            cols += [f"{model}_{eq}{p}_{c}" for p in ("rem", "ans") for c in r3.SENT_COLS]
    cols += [f"prosody_{eq}{p}_{c}" for eq in ("", "eq_") for p in ("rem", "ans") for c in r3.PROSODY]
    cols += [f"shift_{c}" for c in r3.PROSODY]
    table = pd.concat([table, pd.DataFrame(rng.uniform(0, 1, (n, len(cols))), columns=cols)], axis=1)
    return r3.Round3Data(table=table, arrays=arrays, groups=table["ticker"].to_numpy(),
                         format_cols=["fmt_a", "fmt_b"], audio_dim=AUDIO_DIM)  # fmt: skip


def mse(oof):
    return float(((oof["y"] - oof["y_hat"]) ** 2).mean())


@pytest.fixture(scope="module")
def splits():
    return outer_splits(make_data().groups, n_splits=5, n_repeats=1)


@pytest.mark.parametrize("target", ["vol_3", "dir_2_5"])
def test_every_rung_assembles(target):
    data = make_data()
    for rung in r3.RUNGS:
        X, slices = r3.assemble(data, rung, target)
        assert X.shape[0] == 120 and not np.isnan(X).any()
        assert ("past" in slices) == (target == "vol_3")  # direction drops past
    _, s = r3.assemble(data, "AR", "vol_3")
    assert s["audio_emb_rem"].stop - s["audio_emb_rem"].start == AUDIO_LAYERS * AUDIO_DIM


def test_variants_read_their_own_columns():
    data = make_data()
    assert r3.small_columns(data, "text_sent_rem@eq", "vol_3")[0] == "finbert_tone_eq_rem_tone"
    assert r3.small_columns(data, "text_sent_ans@prosus", "vol_3")[0] == "prosus_finbert_ans_tone"
    assert r3.small_columns(data, "prosody_ans@eq", "vol_3")[0] == "prosody_eq_ans_f0_sd_st"
    assert r3.small_columns(data, "past", "vol_15") == ["vol_pre_15"]


@pytest.mark.parametrize("signal_in, better, worse", [("ans", "TA", "TR"), ("rem", "TR", "TA")])
def test_planted_signal_is_found_in_its_own_section(signal_in, better, worse, splits):
    data = make_data(signal_in)
    s1, good, bad = (mse(r3.run_rung(data, r, "vol_3", splits, **FAST)) for r in ("S1", better, worse))
    assert good < 0.5 * s1 and good < 0.5 * bad and bad > 0.9 * s1


def test_no_signal_beats_nothing(splits):
    data = make_data(None)
    s1 = mse(r3.run_rung(data, "S1", "vol_3", splits, **FAST))
    for rung in ("TA", "TR", "PD", "AA"):
        assert mse(r3.run_rung(data, rung, "vol_3", splits, **FAST)) > 0.95 * s1


class RecordingScaler(StandardScaler):
    """Records the first column's values at every fit: the row ids it was fitted on."""

    seen: list = []

    def fit(self, X, y=None, sample_weight=None):
        RecordingScaler.seen.append(frozenset(np.asarray(X)[:, 0].round().astype(int)))
        return super().fit(X, y, sample_weight)


@pytest.mark.parametrize("rung", ["TB", "AA"])
def test_no_fit_sees_test_rows(rung):
    data = make_data()
    X, slices = r3.assemble(data, rung, "vol_3")
    ids = np.arange(len(X))
    X[:, slices["past"].start] = ids  # the first small column
    for key, s in slices.items():
        if key.startswith("text_emb"):
            X[:, s.start] = ids
        if key.startswith("audio_emb"):
            for layer in range(AUDIO_LAYERS):
                X[:, s.start + layer * AUDIO_DIM] = ids
    split = outer_splits(data.groups, n_splits=4, n_repeats=1)[0]
    search = GridSearchCV(r3.build_pipeline(slices, "regression", AUDIO_DIM, scaler=RecordingScaler),
                          r3.param_grid(slices, "regression", **{k: FAST[k] for k in ("ks", "layers", "alphas")}),
                          cv=inner_cv(data.groups[split.train]), n_jobs=1)  # fmt: skip
    RecordingScaler.seen = []
    search.fit(X[split.train], data.table["vol_3"].to_numpy()[split.train])
    train, test = set(split.train), set(split.test)
    assert RecordingScaler.seen
    assert all(rows <= train and not rows & test for rows in RecordingScaler.seen)


def test_grid_shares_k_across_text_blocks_and_tunes_the_layer():
    data = make_data()
    _, tb = r3.assemble(data, "TB", "vol_3")
    grid = r3.param_grid(tb, "regression", ks=(8, None))
    assert grid[0]["features__text_emb_rem__pca__n_components"] == grid[0]["features__text_emb_ans__pca__n_components"] == [8]
    assert grid[1]["features__text_emb_rem__pca"] == ["passthrough"]
    _, ar = r3.assemble(data, "AR", "vol_3")
    assert r3.param_grid(ar, "regression", layers=(3, 6))[0]["features__audio_emb_rem__layer__layer"] == [3, 6]
    _, pd_ = r3.assemble(data, "PD", "dir_0_1")
    assert r3.param_grid(pd_, "classification") == [{"model__C": [1 / a for a in r3.ALPHAS]}]


def test_comparisons_holm_within_each_hypothesis(splits):
    data = make_data("ans")
    oof = pd.concat([r3.run_rung(data, r, t, splits, **FAST) for t in ("vol_3", "vol_7", "dir_0_1")
                     for r in ("S1", "TA", "TR", "PA", "PR", "PD")])  # fmt: skip
    groups = data.table.set_index("event_id")["ticker"]
    c = run_round3.comparisons(oof, groups, ["vol_3", "vol_7", "dir_0_1"])
    conf = c[c["status"] == "confirmatory"]
    assert sorted(conf["hypothesis"].unique()) == ["H6", "H7", "H8", "H9"]
    assert set(conf["target"]) == {"vol_3", "vol_7"} and conf["p_holm"].notna().all()
    h6 = conf[conf["hypothesis"] == "H6"].set_index("target")
    assert (h6["mean_diff"] > 0).all() and (h6["p_holm"] < 0.05).all()  # the planted answers signal
    assert set(c.loc[c["target"] == "dir_0_1", "status"]) == {"exploratory"}


def test_cli_quick_keeps_timings_only_and_full_run_writes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(r3, "load_round3", lambda root, data: make_data("ans", n_firms=20))
    out = tmp_path / "results" / "round3"
    base = ["--out", str(out), "--n-jobs", "1", "--rungs", "S1,TA,TR,PD", "--targets", "vol_3,dir_2_5"]
    assert run_round3.main(base + ["--quick"]) == 0
    trial = json.loads((tmp_path / "results" / "round3_trial" / "timing.json").read_text())
    assert trial["decision"]["audio_repeats"] in (5, 10) and len(trial["timing"]) == 8
    assert not out.exists() and sorted(p.name for p in (tmp_path / "results" / "round3_trial").iterdir()) == ["timing.json"]

    assert run_round3.main(base + ["--repeats", "2"]) == 0
    for name in ("oof.parquet", "metrics.csv", "comparisons.csv", "run.json"):
        assert (out / name).exists()
    oof = pd.read_parquet(out / "oof.parquet")
    assert set(oof["repeat"]) == {0, 1} and set(oof["rung"]) == {"S1", "TA", "TR", "PD"}


def test_load_round3_from_files(tmp_path):
    """The loader joins features, targets, formats and round 1's call-level text."""
    root, data_dir = tmp_path / "repo", tmp_path / "data"
    ids = ["A_1", "B_1", "C_1"]
    f = data_dir / "features" / "round3"
    f.mkdir(parents=True)
    calls = pd.DataFrame({"event_id": ids, "ticker": ["A", "B", "C"], "size_rem": [1.0, 2, 3], "size_ans": [1.0, 2, 3]})
    calls.to_csv(f / "calls.csv", index=False)
    for name in ("text_finbert_tone", "text_finbert_tone_eq", "text_prosus_finbert"):
        np.save(f / f"{name}.npy", np.ones((3, 2, 4), np.float32))
    np.save(f / "hubert.npy", np.ones((3, 2, 13, 5), np.float16))
    ev = root / "event_study/data"
    ev.mkdir(parents=True)
    events = pd.DataFrame({"event_id": ids[::-1], "car_mm_0_1": [0.1, -0.1, 0.2], "car_mm_0_5": [0.0, 0.3, -0.2]})
    for h in r3.VOL_HORIZONS:
        events[f"vol_{h}"], events[f"vol_pre_{h}"] = [-4.0, -4.1, -4.2], [-4.5, -4.6, -4.7]
    events.to_csv(ev / "events.csv", index=False)
    ar = pd.DataFrame([(e, d, v) for e, v in zip(ids, (0.01, -0.01, 0.0)) for d in range(-1, 7)],
                      columns=["event_id", "rel_day", "ar_mm"])  # fmt: skip
    ar.to_csv(ev / "ar_daily.csv", index=False)
    (root / "audio_embeddings/data").mkdir(parents=True)
    pd.DataFrame({"event_id": ids, "sample_rate": [44100, 11025, 44100], "kbps": [64, 16, 64],
                  "channels": [1, 1, 2]}).to_csv(root / "audio_embeddings/data/formats.csv", index=False)  # fmt: skip
    t = root / "text_embeddings/data/finbert_tone"
    t.mkdir(parents=True)
    pd.DataFrame({"event_id": ids[::-1]} | {c: [0.1, 0.2, 0.3] for c in r3.SENT_COLS}).to_csv(t / "calls.csv", index=False)
    np.save(t / "call_emb.npy", np.arange(12, dtype=np.float32).reshape(3, 4))

    d = r3.load_round3(root, data_dir)
    tab = d.table.set_index("event_id")
    assert tab.loc["C_1", "vol_3"] == -4.0 and tab.loc["A_1", "dir_0_1"] == 1  # events were in reverse order
    assert tab["dir_2_5"].tolist() == [1, 0, 0]  # sum of days +2..+5: 0.04, -0.04, 0
    assert tab.loc["A_1", "call_tone"] == 0.3 and (d.arrays["text_emb"][0] == [8, 9, 10, 11]).all()
    assert d.format_cols == ["fmt_11025_16", "fmt_44100_64", "stereo"]
    assert d.arrays["audio_emb_ans"].shape == (3, 13 * 5) and d.audio_dim == 5


def test_figures_from_synthetic_metrics(tmp_path):
    import make_figures

    rows = [{"target": t, "rung": r, "metric": met, "mean": 0.5 + 0.01 * i, "sd": 0.01, "n_repeats": 10}
            for i, r in enumerate(make_figures.RUNGS)
            for t, met in [(t, "mse") for t in make_figures.VOL] + [(t, "auc") for t in make_figures.DIR]]  # fmt: skip
    pd.DataFrame(rows).to_csv(tmp_path / "metrics.csv", index=False)
    assert make_figures.main(["--results", str(tmp_path)]) == 0
    for name in ("mse_round3.png", "auc_round3.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 10_000

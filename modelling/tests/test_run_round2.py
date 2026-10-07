import json

import numpy as np
import pandas as pd
import pytest

import run_ladder
from attention import load_sentences
from conftest import make_project
from run_round2 import main

GRID = ["--alphas", "1,100", "--ks", "2,4,none", "--layers", "0,6", "--n-jobs", "1"]
D_FAST = ["--d-pca", "4", "--d-seeds", "1", "--d-max-iter", "20", "--d-wd", "0.01,1"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    make_project(root, n_firms=20, calls_per_firm=4)
    round1 = root / "results"
    assert run_ladder.main(["--root", str(root), "--out", str(round1), "--quick",
                            "--targets", "vol_3,dir_0_1"] + GRID) == 0  # fmt: skip
    out = root / "results_round2"
    code = main(["--root", str(root), "--round1", str(round1), "--out", str(out), "--quick",
                 "--targets", "vol_3,dir_0_1"] + GRID + D_FAST)  # fmt: skip
    return code, root, out


def test_files(run):
    code, _, out = run
    assert code == 0
    for name in ("oof.parquet", "metrics.csv", "comparisons.csv", "attention_weights.parquet", "run.json"):
        assert (out / name).exists(), name
    for name in ("mse_round2.png", "naive_vs_grouped.png"):
        assert (out / "figures" / name).exists(), name


def test_oof_rungs_and_cv_tags(run):
    _, _, out = run
    oof = pd.read_parquet(out / "oof.parquet")
    grouped = oof[oof["cv"] == "grouped"]
    naive = oof[oof["cv"] == "naive"]
    assert set(grouped.loc[grouped.target == "vol_3", "rung"]) == {"R1P", "R3_resid", "ATT", "UNI"}
    assert set(grouped.loc[grouped.target == "dir_0_1", "rung"]) == {"R1P", "R3_resid"}  # D: volatility only
    assert set(naive["rung"]) == {"B0_naive", "R1_naive", "R2_naive", "R3_naive", "R4_naive"}
    assert set(naive["target"]) == {"vol_3"}
    assert set(oof.loc[oof.rung == "ATT", "repeat"]) == {0}  # --quick: 1 D repeat
    n = oof["event_id"].nunique()
    assert (oof.groupby(["target", "rung", "repeat"]).size() == n).all()


def test_comparisons(run):
    _, _, out = run
    comp = pd.read_csv(out / "comparisons.csv")
    conf = comp[comp["status"] == "confirmatory"]
    assert set(zip(conf["hypothesis"], conf["a"], conf["b"])) == {
        ("H3", "R1P", "R1"), ("H4", "R3_resid", "R3"), ("H5", "ATT", "UNI")}
    assert set(conf["target"]) == {"vol_3"} and (conf["p_holm"] >= conf["p"]).all()
    expl = comp[comp["status"] == "exploratory"]
    assert {("R3_resid", "R1"), ("ATT", "R1"), ("UNI", "R4")} <= set(zip(expl["a"], expl["b"]))
    assert "dir_0_1" in set(expl["target"])
    desc = comp[comp["status"] == "descriptive"]
    assert set(zip(desc["hypothesis"], desc["a"], desc["b"])) == {
        ("H1_naive", "R2_naive", "R1_naive"), ("H2_naive", "R4_naive", "R2_naive")}
    assert comp.loc[comp["status"] != "confirmatory", "p_holm"].isna().all()


def test_attention_weights_point_at_real_sentences(run):
    _, root, out = run
    w = pd.read_parquet(out / "attention_weights.parquet")
    assert set(w.columns) >= {"event_id", "target", "repeat", "fold", "modality", "rank", "sent_idx", "weight"}
    assert set(w["modality"]) == {"text", "audio"} and set(w["rank"]) <= {1, 2, 3}
    ids = sorted(w["event_id"].unique())
    _, _, indices = load_sentences(root, ids, return_idx=True)
    valid = {e: set(ix) for e, ix in zip(ids, indices)}
    assert all(s in valid[e] for e, s in zip(w["event_id"], w["sent_idx"]))
    assert (w["sent_idx"] != 0).all()  # sentence 0 is never audio_ok in the synthetic data


def test_run_json(run):
    _, _, out = run
    meta = json.loads((out / "run.json").read_text())
    assert meta["d_repeats"] == 1 and meta["repeats"] == 2
    assert meta["d"]["weight_decays"] == [0.01, 1.0]
    assert {"runtime_s", "torch", "sklearn"} <= set(meta)

import json

import pandas as pd
import pytest

from conftest import make_project
from run_ladder import main, parse_list

SMALL_GRID = ["--alphas", "1,100", "--ks", "2,4,none", "--layers", "0,6", "--n-jobs", "1"]


def test_parse_list():
    assert parse_list("1,100", float) == (1.0, 100.0)
    assert parse_list("2,4,none", int) == (2, 4, None)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    make_project(root, n_firms=20, calls_per_firm=4)
    out = root / "results"
    code = main(["--root", str(root), "--out", str(out), "--quick",
                 "--targets", "vol_3,dir_0_1"] + SMALL_GRID)  # fmt: skip
    return code, out


def test_exit_code_and_files(run):
    code, out = run
    assert code == 0
    for name in ("oof.parquet", "metrics.csv", "comparisons.csv", "sanity.csv", "run.json"):
        assert (out / name).exists(), name
    assert (out / "figures" / "mse_by_rung.png").exists()
    assert (out / "figures" / "permutation_null.png").exists()


def test_oof_contents(run):
    _, out = run
    oof = pd.read_parquet(out / "oof.parquet")
    assert set(oof["target"]) == {"vol_3", "dir_0_1"}
    expected = {"B0", "R1", "R2", "R3", "R4", "R4_nopast", "R4_prosus_finbert", "R4_wav2vec2"}
    assert set(oof["rung"]) == expected
    assert set(oof["repeat"]) == {0, 1}  # --quick: 2 repeats
    n_calls = oof["event_id"].nunique()
    assert (oof.groupby(["target", "rung", "repeat"]).size() == n_calls).all()


def test_comparisons(run):
    _, out = run
    comp = pd.read_csv(out / "comparisons.csv")
    primary = comp[comp["hypothesis"].isin(["H1", "H2"])]
    assert set(zip(primary["hypothesis"], primary["a"], primary["b"])) == {("H1", "R2", "R1"), ("H2", "R4", "R2")}
    assert set(primary["target"]) == {"vol_3"}  # hypotheses are about volatility only
    assert primary["p_holm"].notna().all() and (primary["p_holm"] >= primary["p"]).all()
    assert comp.loc[comp["hypothesis"] == "exploratory", "p_holm"].isna().all()
    assert {"mean_diff", "ci_low", "ci_high", "n_calls", "n_firms"} <= set(comp.columns)


def test_sanity(run):
    _, out = run
    sanity = pd.read_csv(out / "sanity.csv")
    perm = sanity[sanity["check"] == "permutation"]
    assert len(perm) == 2 and set(perm["target"]) == {"vol_3"}  # --quick: 2 permutations
    assert {"mse_R1", "mse_R4", "gain"} <= set(perm.columns)
    group = sanity[sanity["check"] == "grouping"]
    assert len(group) == 1 and {"mse_grouped", "mse_ungrouped"} <= set(group.columns)


def test_run_json(run):
    _, out = run
    meta = json.loads((out / "run.json").read_text())
    assert meta["repeats"] == 2 and meta["targets"] == ["vol_3", "dir_0_1"]
    assert meta["grid"]["ks"] == [2, 4, None]
    assert {"sklearn", "runtime_s", "n_calls", "n_firms"} <= set(meta)

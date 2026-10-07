import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from conftest import AUDIO_DIM, AUDIO_LAYERS, make_project
from cv import inner_cv, outer_splits
from data import load_call_data
from features import RUNGS, assemble, rung_blocks
from pipelines import ALPHAS, PCA_KS, AUDIO_LAYER_GRID, LayerSelect, build_pipeline, param_grid

SMALL_KS = (2, 4)  # the synthetic embeddings are only 6–8 wide


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    make_project(root, n_firms=12, calls_per_firm=4)
    return load_call_data(root)


def test_layer_select():
    X = np.arange(2 * 3 * 4, dtype=float).reshape(2, 12)  # 3 layers × dim 4
    sel = LayerSelect(layer=1, dim=4)
    np.testing.assert_array_equal(sel.fit(X).transform(X), X[:, 4:8])
    np.testing.assert_array_equal(sel.fit_transform(X[::-1]), X[::-1, 4:8])  # stateless


def test_default_grids():
    assert len(ALPHAS) == 13 and ALPHAS[0] == pytest.approx(1e-2) and ALPHAS[-1] == pytest.approx(1e4)
    assert PCA_KS == (8, 16, 32, 64, None)  # None = no PCA: ridge on the scaled embedding
    assert AUDIO_LAYER_GRID == (3, 6, 9, 12)


def n_combinations(grid):
    return sum(int(np.prod([len(v) for v in g.values()])) for g in grid)


@pytest.mark.parametrize("rung, expected", [("R1", 13), ("R2", 65), ("R3", 260), ("R4", 260)])
def test_grid_sizes(data, rung, expected):
    _, slices = assemble(data, rung_blocks(rung, "vol_3"), "vol_3")
    assert n_combinations(param_grid(slices, "regression")) == expected


def test_shared_pca_k(data):
    _, slices = assemble(data, rung_blocks("R4", "vol_3"), "vol_3")
    grid = param_grid(slices, "regression")
    for g in grid:
        if "features__text_emb__pca" in g:  # the no-PCA entry: both blocks skip PCA
            assert g["features__text_emb__pca"] == g["features__audio_emb__pca"] == ["passthrough"]
        else:
            assert g["features__text_emb__pca__n_components"] == g["features__audio_emb__pca__n_components"]
    assert sum("features__text_emb__pca" in g for g in grid) == 1


def test_no_pca_pipeline_keeps_all_columns(data):
    X, slices = assemble(data, rung_blocks("R2", "vol_3"), "vol_3")
    pipe = build_pipeline(slices, "regression", audio_dim=AUDIO_DIM)
    pipe.set_params(features__text_emb__pca="passthrough", model__alpha=1.0)
    pipe.fit(X, data.table["vol_3"])
    n_small = sum(s.stop - s.start for b, s in slices.items() if b != "text_emb")
    width = slices["text_emb"].stop - slices["text_emb"].start
    assert pipe.named_steps["features"].transform(X).shape[1] == n_small + width


@pytest.mark.parametrize("rung", [r for r in RUNGS if r != "B0"])
@pytest.mark.parametrize("target", ["vol_3", "dir_0_1"])
def test_pipeline_fits_every_rung(data, rung, target):
    X, slices = assemble(data, rung_blocks(rung, target), target)
    y = data.table[target].to_numpy()
    task = "regression" if target.startswith("vol") else "classification"
    pipe = build_pipeline(slices, task, audio_dim=AUDIO_DIM)
    grid = param_grid(slices, task, ks=SMALL_KS, layers=(0, AUDIO_LAYERS - 1))
    pipe.set_params(**{k: v[0] for k, v in grid[0].items()})

    pipe.fit(X, y)

    out = pipe.predict(X) if task == "regression" else pipe.predict_proba(X)[:, 1]
    assert out.shape == (len(y),) and np.isfinite(out).all()


class RecordingScaler(StandardScaler):
    """Records the first column's values at every fit: the row ids it was fitted on."""

    seen: list[frozenset] = []

    def fit(self, X, y=None, sample_weight=None):
        RecordingScaler.seen.append(frozenset(np.asarray(X)[:, 0].round().astype(int)))
        return super().fit(X, y, sample_weight)


def test_no_fit_sees_test_rows(data):
    X, slices = assemble(data, rung_blocks("R4", "vol_3"), "vol_3")
    # Every scaler (small blocks, text, each audio layer) sees a first column = the row id.
    ids = np.arange(len(X))
    X[:, slices["past"].start] = ids
    X[:, slices["text_emb"].start] = ids
    for layer in range(AUDIO_LAYERS):
        X[:, slices["audio_emb"].start + layer * AUDIO_DIM] = ids
    y = data.table["vol_3"].to_numpy()
    split = outer_splits(data.groups, n_splits=4, n_repeats=1)[0]
    pipe = build_pipeline(slices, "regression", audio_dim=AUDIO_DIM, scaler=RecordingScaler)
    search = GridSearchCV(pipe, param_grid(slices, "regression", ks=SMALL_KS, layers=(0, 6)),
                          cv=inner_cv(data.groups[split.train]), n_jobs=1)  # fmt: skip
    RecordingScaler.seen = []

    search.fit(X[split.train], y[split.train])

    train_ids, test_ids = set(split.train), set(split.test)
    assert RecordingScaler.seen, "the spy scaler was never fitted"
    for rows in RecordingScaler.seen:
        assert rows <= train_ids and not rows & test_ids
    n_scalers = 3  # small blocks, text_emb, audio_emb
    refit, inner = RecordingScaler.seen[-n_scalers:], RecordingScaler.seen[:-n_scalers]
    assert all(r == frozenset(train_ids) for r in refit)  # the final refit uses all training rows
    assert inner and all(len(r) < len(train_ids) for r in inner)  # inner-CV fits use fewer

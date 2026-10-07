"""sklearn pipelines for the ladder: per-block preprocessing, then Ridge or logistic regression.

Small blocks are standardised; embedding blocks are standardised and reduced by PCA;
the audio block first selects one hidden-state layer. Because the whole pipeline is what
cross-validation fits, every scaler and PCA is fitted on training rows only.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = tuple(np.logspace(-2, 4, 13))  # Ridge α; logistic regression uses C = 1/α
# Shared by the embedding blocks in a rung. None = no PCA: ridge on the scaled embedding.
# PCA keeps the highest-variance directions, not the most predictive ones, so the inner
# CV may also choose to let ridge regularise the full embedding instead.
PCA_KS = (8, 16, 32, 64, None)
AUDIO_LAYER_GRID = (3, 6, 9, 12)  # hidden-state index; 0 = CNN output
SMALL_BLOCKS = ("past", "format", "text_sent", "prosody")


class LayerSelect(BaseEstimator, TransformerMixin):
    """Pick one layer's columns from a layer-major flattened (layers × dim) block."""

    def __init__(self, layer: int = 6, dim: int = 768):
        self.layer = layer
        self.dim = dim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.layer * self.dim : (self.layer + 1) * self.dim]


def _columns(s: slice) -> list[int]:
    return list(range(s.start, s.stop))


def build_pipeline(slices: dict[str, slice], task: str, audio_dim: int = 768,
                   scaler=StandardScaler) -> Pipeline:  # fmt: skip
    """Pipeline(features → model) for a rung's blocks. scaler is injectable for tests."""
    transformers = []
    small = [c for b in SMALL_BLOCKS if b in slices for c in _columns(slices[b])]
    if small:
        transformers.append(("small", scaler(), small))
    if "text_emb" in slices:
        transformers.append(("text_emb", Pipeline([
            ("scale", scaler()), ("pca", PCA(svd_solver="full")),
        ]), _columns(slices["text_emb"])))  # fmt: skip
    if "audio_emb" in slices:
        transformers.append(("audio_emb", Pipeline([
            ("layer", LayerSelect(dim=audio_dim)), ("scale", scaler()), ("pca", PCA(svd_solver="full")),
        ]), _columns(slices["audio_emb"])))  # fmt: skip
    model = Ridge() if task == "regression" else LogisticRegression(max_iter=5000)
    return Pipeline([("features", ColumnTransformer(transformers)), ("model", model)])


def param_grid(slices: dict[str, slice], task: str, alphas=ALPHAS, ks=PCA_KS,
               layers=AUDIO_LAYER_GRID) -> list[dict]:  # fmt: skip
    """GridSearchCV grid: α (or C) × PCA k (shared across embedding blocks; None skips PCA)
    × audio layer."""
    reg = {"model__alpha": list(alphas)} if task == "regression" else {"model__C": [1 / a for a in alphas]}
    embeddings = [b for b in ("text_emb", "audio_emb") if b in slices]
    if not embeddings:
        return [reg]
    grid = []
    for k in ks:
        if k is None:
            g = dict(reg) | {f"features__{b}__pca": ["passthrough"] for b in embeddings}
        else:
            g = dict(reg) | {f"features__{b}__pca__n_components": [k] for b in embeddings}
        if "audio_emb" in slices:
            g["features__audio_emb__layer__layer"] = list(layers)
        grid.append(g)
    return grid

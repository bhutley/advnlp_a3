"""Round 3 (pre-registered: docs/plans/2026-10-13-round3-design.md): data, rungs, pipelines.

The evaluation machinery is round 1's, imported from ../modelling: firm-grouped outer
splits and inner CV (cv.py), out-of-fold scoring, metrics, the firm-clustered bootstrap
and Holm (evaluate.py), and LayerSelect and the grids (pipelines.py). What is new here
is the blocks: each section (remarks, answers) has its own text, sentiment, prosody and
audio blocks, plus section sizes and the prosody shift.

Block keys are "<type>" or "<type>@<variant>": variant "eq" is the equal-length version,
"prosus" uses ProsusAI FinBERT instead of FinBERT-tone.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modelling"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import features as r1_features  # noqa: E402  (round 1's target → task table)
from cv import inner_cv  # noqa: E402
from data import format_columns  # noqa: E402
from evaluate import OOF_COLUMNS, SCORING, _jsonable  # noqa: E402
from pipelines import ALPHAS, AUDIO_LAYER_GRID, PCA_KS, LayerSelect  # noqa: E402

VOL_HORIZONS = (3, 7, 15, 30)
TARGETS_VOL = [f"vol_{h}" for h in VOL_HORIZONS]
TARGETS_DIR = ["dir_0_1", "dir_0_5", "dir_2_5"]
TASKS = {t: "regression" for t in TARGETS_VOL} | {t: "classification" for t in TARGETS_DIR}
# evaluate.compare / metrics look tasks up in round 1's table; add the new [+2, +5] window.
r1_features.TASKS.update(TASKS)

SENT_COLS = ["tone", "frac_pos", "frac_neg", "p_pos", "p_neg", "p_neu"]
PROSODY = ["f0_sd_st", "speech_rate", "intensity_sd_db", "voiced_frac"]
TEXT_MODEL = {"": "finbert_tone", "eq": "finbert_tone", "prosus": "prosus_finbert"}

SMALL_TYPES = ("past", "format", "size", "text_sent_rem", "text_sent_ans", "text_sent",
               "prosody_rem", "prosody_ans", "shift")  # fmt: skip
EMB_TYPES = ("text_emb_rem", "text_emb_ans", "text_emb")
AUDIO_TYPES = ("audio_emb_rem", "audio_emb_ans")

BASE = ["past", "format", "size"]
TEXT_REM, TEXT_ANS = ["text_emb_rem", "text_sent_rem"], ["text_emb_ans", "text_sent_ans"]
RUNGS = {
    "S1": BASE,
    "TR": BASE + TEXT_REM,
    "TA": BASE + TEXT_ANS,
    "TB": BASE + TEXT_REM + TEXT_ANS,
    "PR": BASE + ["prosody_rem"],
    "PA": BASE + ["prosody_ans"],
    "PD": BASE + ["shift"],
    "AR": BASE + ["audio_emb_rem"],
    "AA": BASE + ["audio_emb_ans"],
    "R1": ["past", "format"],
    "R2": ["past", "format", "text_emb", "text_sent"],
    "TR_eq": BASE + [f"{b}@eq" for b in TEXT_REM],
    "TA_eq": BASE + [f"{b}@eq" for b in TEXT_ANS],
    "PR_eq": BASE + ["prosody_rem@eq"],
    "PA_eq": BASE + ["prosody_ans@eq"],
    "TR_prosus": BASE + [f"{b}@prosus" for b in TEXT_REM],
    "TA_prosus": BASE + [f"{b}@prosus" for b in TEXT_ANS],
}
AUDIO_RUNGS = ("AR", "AA")  # the runtime rule may give these fewer repeats

CONFIRMATORY = {"H6": ("TA", "S1"), "H7": ("TA", "TR"), "H8": ("PA", "PR"), "H9": ("PD", "S1")}
EXPLORATORY_VOL = [("TR", "S1"), ("PR", "S1"), ("PA", "S1"), ("TB", "TA"), ("AR", "S1"), ("AA", "S1"),
                   ("AA", "AR"), ("TB", "R2"), ("TA_eq", "TR_eq"), ("PA_eq", "PR_eq"),
                   ("TA_prosus", "S1"), ("TA_prosus", "TR_prosus")]  # fmt: skip
EXPLORATORY_DIR = list(CONFIRMATORY.values()) + [("TR", "S1")]


def split_key(key: str) -> tuple[str, str]:
    kind, _, variant = key.partition("@")
    return kind, variant


@dataclass
class Round3Data:
    table: pd.DataFrame  # one row per call: ids, targets, small-block columns
    arrays: dict  # embedding block key → (n, dim) float32
    groups: np.ndarray  # ticker per row, for grouped CV
    format_cols: list
    audio_dim: int = 768


def small_columns(data: Round3Data, key: str, target: str) -> list[str]:
    kind, variant = split_key(key)
    part = kind.rsplit("_", 1)[-1]  # rem / ans
    if kind == "past":
        return [target.replace("vol_", "vol_pre_")]
    if kind == "format":
        return list(data.format_cols)
    if kind == "size":
        return ["size_rem", "size_ans"]
    if kind == "text_sent":
        return [f"call_{c}" for c in SENT_COLS]
    if kind.startswith("text_sent_"):
        eq = "eq_" if variant == "eq" else ""
        return [f"{TEXT_MODEL[variant]}_{eq}{part}_{c}" for c in SENT_COLS]
    if kind.startswith("prosody_"):
        return [f"prosody_{'eq_' if variant == 'eq' else ''}{part}_{c}" for c in PROSODY]
    if kind == "shift":
        return [f"shift_{c}" for c in PROSODY]
    raise KeyError(key)


def rung_blocks(rung: str, target: str) -> list[str]:
    """The rung's blocks; direction targets drop 'past', as in round 1."""
    blocks = RUNGS[rung]
    return [b for b in blocks if b != "past"] if TASKS[target] == "classification" else list(blocks)


def assemble(data: Round3Data, rung: str, target: str) -> tuple[np.ndarray, dict]:
    """(X, {block key: column slice}) with the rung's blocks side by side."""
    parts, slices, start = [], {}, 0
    for key in rung_blocks(rung, target):
        kind, _ = split_key(key)
        if kind in SMALL_TYPES:
            part = data.table[small_columns(data, key, target)].to_numpy(dtype=np.float64)
        else:
            part = np.asarray(data.arrays[key], dtype=np.float64)
        parts.append(part)
        slices[key] = slice(start, start + part.shape[1])
        start += part.shape[1]
    return np.hstack(parts), slices


def _cols(s: slice) -> list[int]:
    return list(range(s.start, s.stop))


def build_pipeline(slices: dict, task: str, audio_dim: int = 768, scaler=StandardScaler) -> Pipeline:
    """Small blocks standardised together; each text embedding standardised then PCA; each
    audio embedding has its layer selected, then standardised and PCA (as round 1)."""
    small = [c for k, s in slices.items() if split_key(k)[0] in SMALL_TYPES for c in _cols(s)]
    transformers = [("small", scaler(), small)] if small else []
    for key, s in slices.items():
        kind, _ = split_key(key)
        if kind in EMB_TYPES:
            transformers.append((key, Pipeline([("scale", scaler()), ("pca", PCA(svd_solver="full"))]), _cols(s)))
        elif kind in AUDIO_TYPES:
            transformers.append((key, Pipeline([("layer", LayerSelect(dim=audio_dim)), ("scale", scaler()),
                                                ("pca", PCA(svd_solver="full"))]), _cols(s)))  # fmt: skip
    model = Ridge() if task == "regression" else LogisticRegression(max_iter=5000)
    return Pipeline([("features", ColumnTransformer(transformers)), ("model", model)])


def param_grid(slices: dict, task: str, alphas=ALPHAS, ks=PCA_KS, layers=AUDIO_LAYER_GRID) -> list[dict]:
    """α (C for direction) × PCA k shared by the rung's embedding blocks × audio layer."""
    reg = {"model__alpha": list(alphas)} if task == "regression" else {"model__C": [1 / a for a in alphas]}
    emb = [k for k in slices if split_key(k)[0] in EMB_TYPES + AUDIO_TYPES]
    if not emb:
        return [reg]
    audio = [k for k in emb if split_key(k)[0] in AUDIO_TYPES]
    grid = []
    for k in ks:
        g = dict(reg) | ({f"features__{b}__pca": ["passthrough"] for b in emb} if k is None
                         else {f"features__{b}__pca__n_components": [k] for b in emb})  # fmt: skip
        if audio:  # one layer for all of a rung's audio blocks
            g |= {f"features__{b}__layer__layer": list(layers) for b in audio}
            if len(audio) > 1:
                raise ValueError("a rung with several audio blocks would need a shared layer")
        grid.append(g)
    return grid


def run_rung(data: Round3Data, rung: str, target: str, splits, inner_seed: int = 0, n_jobs: int = -1,
             alphas=ALPHAS, ks=PCA_KS, layers=AUDIO_LAYER_GRID, scaler=StandardScaler) -> pd.DataFrame:  # fmt: skip
    """Out-of-fold predictions for one rung and target, tuned by inner grouped CV (as
    round 1's evaluate.run_rung). y_hat is P(class 1) for direction."""
    task = TASKS[target]
    y = data.table[target].to_numpy(dtype=float)
    X, slices = assemble(data, rung, target)
    ids = data.table["event_id"].to_numpy()
    frames = []
    for split in splits:
        search = GridSearchCV(build_pipeline(slices, task, data.audio_dim, scaler),
                              param_grid(slices, task, alphas, ks, layers), scoring=SCORING[task],
                              cv=inner_cv(data.groups[split.train], seed=inner_seed + split.repeat), n_jobs=n_jobs)  # fmt: skip
        search.fit(X[split.train], y[split.train] if task == "regression" else y[split.train].astype(int))
        y_hat = search.predict(X[split.test]) if task == "regression" else search.predict_proba(X[split.test])[:, 1]
        frames.append(pd.DataFrame({"event_id": ids[split.test], "target": target, "rung": rung,
                                    "repeat": split.repeat, "fold": split.fold, "y": y[split.test],
                                    "y_hat": y_hat, "params": _jsonable(search.best_params_)}))  # fmt: skip
    return pd.concat(frames, ignore_index=True)[OOF_COLUMNS]


def load_round3(root: Path, data_dir: Path) -> Round3Data:
    """The 457 round-3 calls with targets and every block, row-aligned and sorted by event_id."""
    root, f = Path(root), Path(data_dir) / "features" / "round3"
    calls = pd.read_csv(f / "calls.csv")
    ids = calls["event_id"].tolist()
    if ids != sorted(ids):
        raise ValueError("round3/calls.csv must be sorted by event_id")
    events = pd.read_csv(root / "event_study/data/events.csv").set_index("event_id").loc[ids]
    table = calls.copy()
    for h in VOL_HORIZONS:
        table[f"vol_{h}"] = events[f"vol_{h}"].to_numpy()
        table[f"vol_pre_{h}"] = events[f"vol_pre_{h}"].to_numpy()
    for w in ("0_1", "0_5"):
        table[f"dir_{w}"] = (events[f"car_mm_{w}"].to_numpy() > 0).astype(int)
    ar = pd.read_csv(root / "event_study/data/ar_daily.csv")
    car = ar[ar["rel_day"].between(2, 5)].groupby("event_id")["ar_mm"].agg(["sum", "count"]).reindex(ids)
    if (car["count"] != 4).any():
        raise ValueError("CAR [+2, +5] needs all four days for every call")
    table["dir_2_5"] = (car["sum"].to_numpy() > 0).astype(int)

    formats = pd.read_csv(root / "audio_embeddings/data/formats.csv").set_index("event_id").loc[ids].reset_index()
    fmt = format_columns(formats)
    table = pd.concat([table, fmt.reset_index(drop=True)], axis=1)
    r1_text = root / "text_embeddings/data/finbert_tone"
    r1_calls = pd.read_csv(r1_text / "calls.csv")
    pos = pd.Series(range(len(r1_calls)), index=r1_calls["event_id"]).loc[ids].to_numpy()
    for c in SENT_COLS:
        table[f"call_{c}"] = r1_calls[c].to_numpy()[pos]

    arrays = {"text_emb": np.load(r1_text / "call_emb.npy")[pos].astype(np.float32)}
    for variant, model in (("", "finbert_tone"), ("eq", "finbert_tone"), ("prosus", "prosus_finbert")):
        emb = np.load(f / f"text_{model}{'_eq' if variant == 'eq' else ''}.npy")
        suffix = f"@{variant}" if variant else ""
        arrays[f"text_emb_rem{suffix}"], arrays[f"text_emb_ans{suffix}"] = emb[:, 0], emb[:, 1]
    hub = np.load(f / "hubert.npy")
    arrays["audio_emb_rem"] = hub[:, 0].reshape(len(ids), -1).astype(np.float32)
    arrays["audio_emb_ans"] = hub[:, 1].reshape(len(ids), -1).astype(np.float32)
    if any(np.isnan(a).any() for a in arrays.values()) or table.drop(columns=["event_id", "ticker"]).isna().any().any():
        raise ValueError("missing values in the round-3 inputs")
    return Round3Data(table=table, arrays=arrays, groups=table["ticker"].to_numpy(),
                      format_cols=list(fmt.columns), audio_dim=hub.shape[-1])  # fmt: skip

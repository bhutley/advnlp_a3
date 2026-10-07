"""Round 2, follow-up D: attention pooling over a call's sentences.

Each sentence has a text vector (FinBERT) and an audio vector (HuBERT layer 6). A learned
attention vector per modality weights the call's sentences; the pooled vectors, with past
volatility and format, feed a linear head. The control (learn_attention=False) freezes the
attention at zero, so the weights are uniform and pooling is a plain mean: comparing the
two isolates attention itself.
"""

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn


def load_sentences(root: Path, event_ids, text_model: str = "finbert_tone",
                   audio_model: str = "hubert", return_idx: bool = False):  # fmt: skip
    """Per call, in event_ids order: (text, audio) sentence vectors for its audio_ok
    sentences, in sent_idx order, joined on (event_id, sent_idx). With return_idx, also
    each call's sent_idx array.

    Raises ValueError if a call has no usable sentence or a text row is missing.
    """
    root = Path(root)
    text_dir = root / "text_embeddings/data" / text_model
    audio_dir = root / "audio_embeddings/data" / audio_model
    text_rows = pd.read_csv(text_dir / "sentences.csv", usecols=["event_id", "sent_idx"])
    audio_rows = pd.read_csv(audio_dir / "sentences.csv", usecols=["event_id", "sent_idx", "audio_ok"])
    text_emb = np.load(text_dir / "sentence_emb.npy", mmap_mode="r")
    audio_emb = np.load(audio_dir / "sentence_emb.npy", mmap_mode="r")

    text_pos = pd.Series(np.arange(len(text_rows)),
                         index=pd.MultiIndex.from_frame(text_rows[["event_id", "sent_idx"]]))  # fmt: skip
    ok = audio_rows[audio_rows["audio_ok"].astype(bool)].copy()
    ok["row"] = ok.index
    by_call = {e: g.sort_values("sent_idx") for e, g in ok.groupby("event_id")}

    texts, audios, indices = [], [], []
    for event_id in event_ids:
        if event_id not in by_call:
            raise ValueError(f"{event_id}: no audio_ok sentences")
        g = by_call[event_id]
        keys = list(zip(g["event_id"], g["sent_idx"]))
        missing = [k for k in keys if k not in text_pos.index]
        if missing:
            raise ValueError(f"{event_id}: no text row for sentence {missing[0][1]}")
        texts.append(np.asarray(text_emb[text_pos.loc[keys].to_numpy()], dtype=np.float32))
        audios.append(np.asarray(audio_emb[g["row"].to_numpy()], dtype=np.float32))
        indices.append(g["sent_idx"].to_numpy())
    return (texts, audios, indices) if return_idx else (texts, audios)


def pad(arrays: list[np.ndarray], length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack variable-length (n_i, d) arrays into a zero-padded [B, S, d] tensor and a
    [B, S] boolean mask of real positions."""
    length = length or max(len(a) for a in arrays)
    dim = arrays[0].shape[1]
    x = torch.zeros(len(arrays), length, dim)
    mask = torch.zeros(len(arrays), length, dtype=torch.bool)
    for i, a in enumerate(arrays):
        x[i, : len(a)] = torch.as_tensor(np.asarray(a), dtype=torch.float32)
        mask[i, : len(a)] = True
    return x, mask


class AttentionPool(nn.Module):
    """Attention-pooled text and audio vectors plus small features → a linear prediction."""

    def __init__(self, text_dim: int, audio_dim: int, small_dim: int, learn_attention: bool = True,
                 init_scale: float = 0.1):  # fmt: skip
        super().__init__()
        self.learn_attention = learn_attention
        scale = init_scale if learn_attention else 0.0
        self.w_text = nn.Parameter(torch.randn(text_dim) * scale, requires_grad=learn_attention)
        self.w_audio = nn.Parameter(torch.randn(audio_dim) * scale, requires_grad=learn_attention)
        self.head = nn.Linear(text_dim + audio_dim + small_dim, 1)

    @staticmethod
    def _softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)

    def weights(self, text, audio, mask) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, S] attention weights per modality; they sum to 1 over real sentences, 0 on padding."""
        return self._softmax(text @ self.w_text, mask), self._softmax(audio @ self.w_audio, mask)

    def pooled(self, text, audio, mask) -> tuple[torch.Tensor, torch.Tensor]:
        a_text, a_audio = self.weights(text, audio, mask)
        return (a_text.unsqueeze(-1) * text).sum(1), (a_audio.unsqueeze(-1) * audio).sum(1)

    def forward(self, text, audio, mask, small) -> torch.Tensor:
        p_text, p_audio = self.pooled(text, audio, mask)
        return self.head(torch.cat([p_text, p_audio, small], dim=1)).squeeze(-1)


CACHE_SIZE = 6  # GridSearchCV cycles 4 inner training sets per candidate, then refits once


@dataclass
class SentenceStore:
    """Per-call sentence vectors (all calls), plus per-call small features (past + format).

    Shared, not copied: GridSearchCV deep-copies estimator parameters on every clone, and
    the store holds hundreds of MB. It also caches the in-fold reduction (scaler + PCA,
    fitted on a training set's sentences) so a weight-decay grid reuses one PCA fit.
    """

    text: list
    audio: list
    small: np.ndarray
    sent_idx: list
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)

    def __deepcopy__(self, memo):
        return self

    def reduce(self, train_idx: np.ndarray, pca_dim: int) -> dict:
        """Scalers and PCAs fitted on the training calls' sentences, and every call's
        reduced sentences. Cached (LRU) by training set."""
        key = (hashlib.sha1(np.sort(train_idx).tobytes()).hexdigest(), pca_dim)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        entry = {}
        for name, arrays in (("text", self.text), ("audio", self.audio)):
            fit_rows = np.vstack([arrays[i] for i in train_idx])
            scaler = StandardScaler().fit(fit_rows)
            k = min(pca_dim, fit_rows.shape[1])
            pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(scaler.transform(fit_rows))
            lengths = np.cumsum([len(a) for a in arrays])[:-1]
            reduced = pca.transform(scaler.transform(np.vstack(arrays))).astype(np.float32)
            entry |= {f"{name}_scaler": scaler, f"{name}_pca": pca,
                      f"{name}_reduced": np.split(reduced, lengths)}  # fmt: skip
        self._cache[key] = entry
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return entry


def make_store(texts, audios, indices, small: np.ndarray) -> SentenceStore:
    return SentenceStore(text=texts, audio=audios, small=np.asarray(small, dtype=np.float32), sent_idx=indices)


class AttentionRegressor(BaseEstimator, RegressorMixin):
    """sklearn estimator over call indices: X is (n, 1) with indices into store.

    fit: in-fold scaler + PCA per modality (store.reduce), scaled small features and
    standardised y, then n_seeds AttentionPool models trained by L-BFGS on MSE plus weight
    decay (head bias unpenalised). predict averages the seeds.
    """

    def __init__(self, store=None, learn_attention: bool = True, weight_decay: float = 1e-2,
                 pca_dim: int = 32, n_seeds: int = 3, max_iter: int = 200, seed: int = 0):  # fmt: skip
        self.store = store
        self.learn_attention = learn_attention
        self.weight_decay = weight_decay
        self.pca_dim = pca_dim
        self.n_seeds = n_seeds
        self.max_iter = max_iter
        self.seed = seed

    def _batch(self, idx):
        text, mask = pad([self._entry["text_reduced"][i] for i in idx])
        audio, _ = pad([self._entry["audio_reduced"][i] for i in idx])
        small = torch.as_tensor(self.small_scaler_.transform(self.store.small[idx]), dtype=torch.float32)
        return text, audio, mask, small

    def _train(self, batch, y, seed: int) -> AttentionPool:
        torch.manual_seed(seed)
        text, audio, mask, small = batch
        model = AttentionPool(text.shape[2], audio.shape[2], small.shape[1], self.learn_attention)
        params = [p for p in model.parameters() if p.requires_grad]
        penalised = [p for n, p in model.named_parameters() if p.requires_grad and n != "head.bias"]
        opt = torch.optim.LBFGS(params, max_iter=self.max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.mean((model(text, audio, mask, small) - y) ** 2)
            loss = loss + self.weight_decay * sum((p**2).sum() for p in penalised)
            loss.backward()
            return loss

        opt.step(closure)
        return model.eval()

    def fit(self, X, y):
        idx = np.asarray(X)[:, 0].astype(int)
        self._entry = self.store.reduce(idx, self.pca_dim)
        for name in ("text_scaler", "text_pca", "audio_scaler", "audio_pca"):
            setattr(self, f"{name}_", self._entry[name])
        self.small_scaler_ = StandardScaler().fit(self.store.small[idx])
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = y.mean(), y.std() or 1.0
        target = torch.as_tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)
        batch = self._batch(idx)
        self.models_ = [self._train(batch, target, self.seed + s) for s in range(self.n_seeds)]
        return self

    @torch.no_grad()
    def predict(self, X):
        batch = self._batch(np.asarray(X)[:, 0].astype(int))
        pred = torch.stack([m(*batch) for m in self.models_]).mean(0).numpy()
        return pred * self.y_std_ + self.y_mean_

    @torch.no_grad()
    def top_sentences(self, X, k: int = 3) -> pd.DataFrame:
        """The k highest-weighted sentences per call and modality (weights averaged over seeds)."""
        idx = np.asarray(X)[:, 0].astype(int)
        text, audio, mask, _ = self._batch(idx)
        weights = [m.weights(text, audio, mask) for m in self.models_]
        rows = []
        for m_i, modality in enumerate(("text", "audio")):
            w = torch.stack([w[m_i] for w in weights]).mean(0).numpy()
            for row, call in enumerate(idx):
                n = int(mask[row].sum())
                order = np.argsort(-w[row, :n], kind="stable")[:k]
                rows += [{"call": int(call), "modality": modality, "rank": r + 1,
                          "sent_idx": int(self.store.sent_idx[call][pos]), "weight": float(w[row, pos])}
                         for r, pos in enumerate(order)]  # fmt: skip
        return pd.DataFrame(rows)


def run_attention(data, store: SentenceStore, rung: str, target: str, splits, weight_decays,
                  inner_seed: int = 0, n_seeds: int = 3, max_iter: int = 200, pca_dim: int = 32):  # fmt: skip
    """Out-of-fold predictions for ATT (learned attention) or UNI (uniform control), with the
    weight decay tuned by inner grouped CV on each outer training fold. For ATT, also the top
    3 sentences per modality for each test call, from that fold's refit.

    Returns (oof, weights); weights is None for UNI.
    """
    import json

    from sklearn.model_selection import GridSearchCV

    from cv import inner_cv
    from evaluate import OOF_COLUMNS

    learn = {"ATT": True, "UNI": False}[rung]
    y = data.table[target].to_numpy(dtype=float)
    ids = data.table["event_id"].to_numpy()
    frames, weight_frames = [], []
    for split in splits:
        X_train, X_test = split.train.reshape(-1, 1), split.test.reshape(-1, 1)
        search = GridSearchCV(
            AttentionRegressor(store, learn_attention=learn, pca_dim=pca_dim, n_seeds=n_seeds,
                               max_iter=max_iter, seed=inner_seed),
            {"weight_decay": list(weight_decays)},
            scoring="neg_mean_squared_error",
            cv=inner_cv(data.groups[split.train], seed=inner_seed + split.repeat),
            n_jobs=1,
        )  # fmt: skip
        search.fit(X_train, y[split.train])
        frames.append(pd.DataFrame({
            "event_id": ids[split.test], "target": target, "rung": rung, "repeat": split.repeat,
            "fold": split.fold, "y": y[split.test], "y_hat": search.predict(X_test),
            "params": json.dumps({"weight_decay": search.best_params_["weight_decay"]}),
        }))  # fmt: skip
        if learn:
            top = search.best_estimator_.top_sentences(X_test, k=3)
            top.insert(0, "event_id", ids[top.pop("call").to_numpy()])
            weight_frames.append(top.assign(target=target, repeat=split.repeat, fold=split.fold))
    oof = pd.concat(frames, ignore_index=True)[OOF_COLUMNS]
    return oof, (pd.concat(weight_frames, ignore_index=True) if weight_frames else None)

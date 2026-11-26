"""Dataset container shared by every learner."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from amifx.config import CACHE


def signed_log(x):
    """Fixed, label-free, per-value transform (no fitted statistics -> no leakage)."""
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    src: np.ndarray
    dst: np.ndarray
    views: dict
    meta: dict
    _cache: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.frame)

    @property
    def step(self):
        return self.frame["step"].values

    @property
    def y(self):
        return self.frame["y"].values

    def X(self, view: str, transform: bool = False) -> np.ndarray:
        key = (view, transform)
        if key not in self._cache:
            cols = self.views[view]
            a = self.frame[cols].to_numpy(dtype="float32", na_value=np.nan)
            if transform:
                a = np.nan_to_num(signed_log(a), nan=0.0).astype("float32")
            self._cache[key] = a
        return self._cache[key]

    def ctx(self):
        return self.X("ctx")

    # ---- graph helpers (graph is a union of per-step disconnected graphs) ----
    def norm_adj(self):
        """Row-normalised undirected adjacency (mean aggregator) as torch sparse."""
        if "adj" not in self._cache:
            src = np.concatenate([self.src, self.dst]); dst = np.concatenate([self.dst, self.src])
            deg = np.bincount(dst, minlength=self.n).astype("float32")
            w = 1.0 / np.maximum(deg[dst], 1)
            A = torch.sparse_coo_tensor(torch.tensor(np.vstack([dst, src])), torch.tensor(w, dtype=torch.float32),
                                        (self.n, self.n)).coalesce()
            self._cache["adj"] = A
        return self._cache["adj"]

    def edge_index(self):
        if "ei" not in self._cache:
            ei = np.vstack([np.concatenate([self.src, self.dst]), np.concatenate([self.dst, self.src])])
            self._cache["ei"] = torch.tensor(ei, dtype=torch.long)
        return self._cache["ei"]

    def neighbours(self):
        """(i, j) pairs, j a neighbour of i (undirected)."""
        if "nbr" not in self._cache:
            self._cache["nbr"] = (np.concatenate([self.dst, self.src]), np.concatenate([self.src, self.dst]))
        return self._cache["nbr"]


def load(name: str) -> Dataset:
    f = pd.read_parquet(CACHE / f"{name}_cases.parquet")
    e = pd.read_parquet(CACHE / f"{name}_edges.parquet")
    meta = json.loads((CACHE / f"{name}_meta.json").read_text())
    pref = lambda p: [c for c in f.columns if c.startswith(p)]
    views = {
        "tx": pref("tx__"), "nb": pref("nb__"), "tp": pref("tp__"), "ac": pref("ac__"), "gs": pref("gs__"),
        "ctx": pref("ctx__"),
    }
    views["std"] = views["tx"] + views["nb"]                     # conventional feature space
    views["all"] = views["tx"] + views["nb"] + views["tp"] + views["ac"] + views["gs"]  # union of evidence views
    views["graphin"] = views["tx"] + views["gs"]
    return Dataset(name, f, e.src.values.astype(np.int64), e.dst.values.astype(np.int64), views, meta)

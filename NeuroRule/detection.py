"""Shared error-mask generation upstream of every compared method."""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .dataset import TableDataset
from .discretizer import MISSING_VALUES

LABELING_BUDGET = 20


def _column_value_space(df: pd.DataFrame, col: str) -> np.ndarray:
    """Standardized values for a numeric column, frequencies for a categorical one."""
    s = df[col].astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        x = num.to_numpy(dtype=float)
        x[~np.isfinite(x)] = np.nanmedian(x[np.isfinite(x)]) \
            if np.isfinite(x).any() else 0.0
    else:
        x = s.map(s.value_counts()).to_numpy(dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    z = (x - mu) / (sd if sd and np.isfinite(sd) and sd > 0 else 1.0)
    return z.reshape(-1, 1)


# cells per dense indicator matrix the outlier scans may materialize
INDICATOR_CELL_BUDGET = 5 * 10**9


def _column_indicator_space(df: pd.DataFrame, col: str) -> np.ndarray:
    """One point per distinct value; wide columns raise past the budget."""
    s = df[col].astype(str).str.strip()
    vals, inv = np.unique(s.to_numpy(), return_inverse=True)
    if len(s) * len(vals) > INDICATOR_CELL_BUDGET:
        raise ValueError(f"indicator matrix too wide: {len(s)} x {len(vals)}")
    ind = np.zeros((len(s), len(vals)))
    ind[np.arange(len(s)), inv] = 1.0
    return ind


def _flag_per_column(df: pd.DataFrame, flag_fn) -> np.ndarray:
    mask = np.zeros(df.shape, dtype=bool)
    for j, col in enumerate(df.columns):
        flags = np.zeros(len(df), dtype=bool)
        for space_fn in (_column_value_space, _column_indicator_space):
            try:
                flags |= flag_fn(space_fn(df, col))
            except Exception:
                warnings.warn(f"{col}: outlier scan failed, "
                              "representation left unflagged")
        mask[:, j] = flags
    return mask


def detect_iforest(df: pd.DataFrame, contamination: float = 0.10) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    def fn(z):
        model = IsolationForest(contamination=contamination, random_state=0)
        return model.fit_predict(z) == -1

    return _flag_per_column(df, fn)


def detect_lof(df: pd.DataFrame, n_neighbors: int = 20,
               threshold: float = 1.2) -> np.ndarray:
    """Flag cells whose local density sits a factor ``threshold`` below their neighborhood."""
    from sklearn.neighbors import LocalOutlierFactor

    def fn(z):
        nn = min(n_neighbors, max(2, len(z) - 1))
        model = LocalOutlierFactor(n_neighbors=nn)
        with warnings.catch_warnings():
            # value-indicator columns score duplicate points by design
            warnings.simplefilter("ignore", UserWarning)
            model.fit_predict(z)
        score = np.nan_to_num(-model.negative_outlier_factor_, nan=1.0)
        return score > threshold

    return _flag_per_column(df, fn)


def detect_dbscan(df: pd.DataFrame, min_samples: int = 5) -> np.ndarray:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    def fn(z):
        nn = min(min_samples, max(2, len(z) - 1))
        kdist = NearestNeighbors(n_neighbors=nn).fit(z).kneighbors(z)[0][:, -1]
        eps = float(np.percentile(kdist, 90))
        labels = DBSCAN(eps=max(eps, 1e-9), min_samples=nn).fit_predict(z)
        return labels == -1

    return _flag_per_column(df, fn)


def detect_missing(df: pd.DataFrame) -> np.ndarray:
    vals = df.astype(str)
    return vals.apply(lambda s: s.str.strip().str.lower().isin(MISSING_VALUES)).to_numpy()


def detect_raha(dataset: TableDataset, budget: int = LABELING_BUDGET) -> np.ndarray:
    import raha

    app = raha.Detection()
    app.LABELING_BUDGET = budget
    app.SAVE_RESULTS = False
    app.VERBOSE = False
    dd = {
        "name": dataset.name,
        "path": os.path.join(dataset.root,
                             "raw.csv" if os.path.exists(os.path.join(dataset.root, "raw.csv"))
                             else "dirty.csv"),
        "clean_path": os.path.join(dataset.root,
                                   os.path.basename(dataset.config.get("clean_data_path", "clean.csv"))),
    }
    detected = app.run(dd)
    mask = np.zeros(dataset.dirty.shape, dtype=bool)
    for (r, c) in detected.keys():
        mask[int(r), int(c)] = True
    return mask


def generate_mask(dataset: TableDataset, detector: str = "default",
                  cache_dir: Optional[str] = None) -> np.ndarray:
    """Full-table error mask; cached on disk when a cache directory is given."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{dataset.name}_{detector}.csv")
        if os.path.exists(cache_path):
            m = pd.read_csv(cache_path, dtype=bool).to_numpy()
            if m.shape == dataset.dirty.shape:
                return m

    if detector == "default":
        # Raha is the labeling-budget default; ``default_detector`` pins iforest
        pinned = str(dataset.config.get("default_detector", "") or "").strip()
        if pinned == "iforest":
            mask = detect_iforest(dataset.dirty) | detect_missing(dataset.dirty)
        elif pinned:
            raise ValueError(f"{dataset.name}: unknown default_detector: {pinned}")
        elif dataset.clean is not None:
            mask = detect_raha(dataset)
        else:
            mask = detect_iforest(dataset.dirty) | detect_missing(dataset.dirty)
    elif detector == "lof":
        mask = detect_lof(dataset.dirty) | detect_missing(dataset.dirty)
    elif detector == "dbscan":
        mask = detect_dbscan(dataset.dirty) | detect_missing(dataset.dirty)
    else:
        raise ValueError(f"unknown detector: {detector}")

    if cache_path:
        pd.DataFrame(mask, columns=dataset.dirty.columns).to_csv(cache_path, index=False)
    return mask

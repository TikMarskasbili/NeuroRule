"""Benchmark dataset loading and validation (schemas in ``datasets_config.json``)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "datasets")
)
DATASET_CONFIGS_PATH = os.path.join(DEFAULT_ROOT, "datasets_config.json")


def _read_csv_str(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False,
                           low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False,
                           low_memory=False, encoding="latin-1")


@dataclass
class TableDataset:
    name: str
    root: str
    config: dict
    dirty: pd.DataFrame
    clean: Optional[pd.DataFrame]
    numerical_features: List[str]
    categorical_features: List[str]
    target_feature: str
    # clean-table placeholders marking unrecoverable cells, excluded from scoring
    nongold_values: Optional[List[str]] = None
    _labels: Optional[np.ndarray] = field(default=None, init=False,
                                          repr=False, compare=False)

    @property
    def feature_columns(self) -> List[str]:
        return self.numerical_features + self.categorical_features

    def features(self) -> pd.DataFrame:
        return self.dirty[self.feature_columns]

    def labels(self) -> np.ndarray:
        """Target as contiguous class indices over the full table (cached)."""
        if self._labels is None:
            codes, _ = pd.factorize(
                self.dirty[self.target_feature].astype(str), sort=True)
            self._labels = codes.astype(np.int64)
        return self._labels

    @property
    def n_classes(self) -> int:
        return int(self.labels().max()) + 1

    def __len__(self) -> int:
        return len(self.dirty)


def load_dataset(name: str, root: str = DEFAULT_ROOT) -> TableDataset:
    ddir = os.path.join(root, name)
    with open(DATASET_CONFIGS_PATH) as f:
        configs = json.load(f)
    if name not in configs:
        raise FileNotFoundError(f"{name}: no entry in {DATASET_CONFIGS_PATH}")
    config = configs[name]

    dirty_path = None
    for cand in ("raw.csv", "dirty.csv"):
        p = os.path.join(ddir, cand)
        if os.path.exists(p):
            dirty_path = p
            break
    if dirty_path is None:
        raise FileNotFoundError(f"{name}: no raw.csv/dirty.csv under {ddir}")
    dirty = _read_csv_str(dirty_path)

    clean = None
    clean_name = os.path.basename(config.get("clean_data_path", ""))
    if clean_name:
        p = os.path.join(ddir, clean_name)
        if os.path.exists(p) and "mode_clean" not in clean_name:
            clean = _read_csv_str(p)

    omitted = set(config.get("omitted_features") or [])
    target = (config.get("target_feature") or [""])[0]

    missing_cols = [c for c in [target] + list(config.get("numerical_features", []))
                    + list(config.get("categorical_features", []))
                    if c and c not in dirty.columns]
    if missing_cols:
        raise ValueError(f"{name}: config columns not in table: {missing_cols}")

    numerical = [c for c in config.get("numerical_features", [])
                 if c not in omitted and c != target and c in dirty.columns]
    categorical = [c for c in config.get("categorical_features", [])
                   if c not in omitted and c != target and c in dirty.columns]

    if clean is not None:
        if clean.shape[0] != dirty.shape[0]:
            raise ValueError(
                f"{name}: dirty/clean row mismatch (dirty={dirty.shape}, clean={clean.shape})")
        if set(clean.columns) != set(dirty.columns):
            raise ValueError(
                f"{name}: dirty/clean column sets differ "
                f"(only-dirty={set(dirty.columns) - set(clean.columns)}, "
                f"only-clean={set(clean.columns) - set(dirty.columns)})")
        clean = clean[dirty.columns]

    return TableDataset(
        name=name, root=ddir, config=config, dirty=dirty, clean=clean,
        numerical_features=numerical, categorical_features=categorical,
        target_feature=target,
        nongold_values=config.get("nongold_values") or None,
    )

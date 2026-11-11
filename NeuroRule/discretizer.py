"""Per-column discretization into the shared id space (PAD/MISSING/UNSEEN + values)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

PAD = 0
MISSING = 1
UNSEEN = 2
OFFSET = 3  # first id of a real value

MISSING_VALUES = {
    "", "empty", "nan", "null", "none", "?", "na", "n/a", "-", " ",
    "null_", "undefined", "missing",
}


def is_missing_value(disc: str) -> bool:
    return disc.strip().lower() in MISSING_VALUES


@dataclass
class ColumnDiscretizer:
    """Discrete value domain of one column, plus the quantile bins for numerics."""

    col: str
    is_numerical: bool
    n_bins: int = 16
    domain: Dict[str, int] = field(default_factory=dict)
    inv_domain: Dict[int, str] = field(default_factory=dict)
    bin_edges: np.ndarray = None       # numerical only
    bin_centers: np.ndarray = None     # numerical only
    bin_means: np.ndarray = None       # numerical only

    @classmethod
    def fit(cls, col: str, vals: pd.Series, is_numerical: bool, n_bins: int = 16) -> "ColumnDiscretizer":
        disc = cls(col=col, is_numerical=is_numerical, n_bins=n_bins)
        if is_numerical:
            num = pd.to_numeric(vals, errors="coerce")
            finite = num.dropna()
            finite = finite[np.isfinite(finite)]
            if len(finite) == 0:
                qs = np.array([0.0, 1.0])
            else:
                qs = np.unique(np.quantile(finite.to_numpy(dtype=float), np.linspace(0, 1, n_bins + 1)))
                if len(qs) < 2:
                    qs = np.array([qs[0], qs[0] + 1.0])
            disc.bin_edges = qs
            lows, highs = qs[:-1], qs[1:]
            disc.bin_centers = (lows + highs) / 2.0
            # observed mean per interval (bin centers can sit far from realized values)
            pos = np.clip(np.searchsorted(qs, finite.to_numpy(dtype=float), side="right") - 1,
                          0, len(lows) - 1)
            means = []
            for b in range(len(lows)):
                sel = finite.to_numpy(dtype=float)[pos == b]
                means.append(float(sel.mean()) if len(sel) else float(disc.bin_centers[b]))
            disc.bin_means = np.array(means, dtype=float)
            disc.domain = {f"bin_{i}": OFFSET + i for i in range(len(lows))}
        else:
            counter: Dict[str, int] = {}
            for v in vals:
                s = str(v).strip()
                if is_missing_value(s):
                    continue
                counter[s] = counter.get(s, 0) + 1
            top = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n_bins]
            disc.domain = {s: OFFSET + i for i, (s, _) in enumerate(top)}
        disc.inv_domain = {v: k for k, v in disc.domain.items()}
        return disc

    @property
    def domain_size(self) -> int:
        """Number of value ids defined for this column (incl. PAD/MISSING/UNSEEN)."""
        if self.is_numerical:
            return OFFSET + len(self.bin_centers)
        return OFFSET + len(self.domain)

    def transform(self, vals) -> np.ndarray:
        n = len(vals)
        out = np.empty(n, dtype=np.int64)
        if self.is_numerical:
            num = pd.to_numeric(pd.Series(list(vals), dtype=object), errors="coerce").to_numpy(dtype=float)
            fin = np.isfinite(num)
            out[:] = MISSING
            pos = np.clip(np.searchsorted(self.bin_edges, num[fin], side="right") - 1,
                          0, len(self.bin_centers) - 1)
            out[fin] = OFFSET + pos
            return out
        for i, v in enumerate(vals):
            s = str(v).strip()
            if is_missing_value(s):
                out[i] = MISSING
            else:
                out[i] = self.domain.get(s, UNSEEN)
        return out

    def inverse_transform(self, ids) -> List[str]:
        out = []
        for i in ids:
            i = int(i)
            if i in self.inv_domain:
                out.append(self.inv_domain[i])
            elif i == MISSING:
                out.append("empty")
            elif i == UNSEEN:
                out.append("")       # caller keeps the dirty value in this case
            else:
                out.append("")
        return out

    def inverse_transform_numeric(self, ids) -> np.ndarray:
        """Read numeric column ids back to original-scale values."""
        means = self.bin_means if self.bin_means is not None else self.bin_centers
        out = np.empty(len(ids), dtype=float)
        for i, disc in enumerate(ids):
            disc = int(disc)
            if disc >= OFFSET and means is not None and (disc - OFFSET) < len(means):
                out[i] = means[disc - OFFSET]
            else:
                out[i] = np.nan
        return out

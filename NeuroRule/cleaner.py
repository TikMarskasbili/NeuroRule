"""Repair-only cleaner: (dirty table, error mask) -> (repaired table, rules)."""

from __future__ import annotations

import copy
import resource
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import EngineConfig, NeuroRuleEngine
from .discretizer import MISSING, OFFSET, ColumnDiscretizer, is_missing_value

TASK_LR = 1e-3
TASK_HIDDEN = 256
CAT_DOMAIN_CAP = 16
AUG_MASK_PROB = 0.15
REPAIR_STATE_BYTES = 1 << 30


@dataclass
class RepairConfig(EngineConfig):
    """End-to-end training configuration."""

    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-2
    beta0: float = 0.1
    beta_max: float = 1.0
    beta_warm_epochs: int = 5
    patience: int = 5
    seed: int = 43
    device: str = "auto"
    use_task_loss: bool = True
    use_rep_loss: bool = True
    cat_bins: Optional[int] = None  # categorical cap; None falls back to n_bins


class _GuideMLP(nn.Module):
    """Downstream task model C_w (2-layer MLP with ReLU)."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, TASK_HIDDEN), nn.ReLU(),
            nn.Linear(TASK_HIDDEN, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class _TableEncoder:
    """Value states -> downstream features (numerics standardized, categoricals one-hot)."""

    def __init__(self, discretizers: List[ColumnDiscretizer], num_cols: List[int],
                 train_df: pd.DataFrame):
        self.discretizers = discretizers
        self.num_cols = set(num_cols)
        self.cat_pos: Dict[int, int] = {}
        self.cat_maps: List[Dict[str, int]] = []
        self.cat_offsets: List[int] = []
        self.num_index: Dict[int, int] = {}
        dim = 0
        for j, disc in enumerate(discretizers):
            if j in self.num_cols:
                self.num_index[j] = dim
                dim += 1
            else:
                values = pd.Series([str(v).strip() for v in train_df.iloc[:, j]
                                    if not is_missing_value(str(v).strip())])
                top = values.value_counts().index[:CAT_DOMAIN_CAP]
                m = {v: i for i, v in enumerate(sorted(top))}
                self.cat_pos[j] = len(self.cat_maps)
                self.cat_maps.append(m)
                self.cat_offsets.append(dim)
                dim += len(m)
        self.dim = dim
        self.num_mean: Dict[int, float] = {}
        self.num_std: Dict[int, float] = {}
        for j in self.num_cols:
            col = pd.to_numeric(train_df.iloc[:, j], errors="coerce")
            mu = float(col.mean()) if col.notna().any() else 0.0
            sd = float(col.std()) if col.notna().sum() > 1 else 1.0
            self.num_mean[j] = mu
            self.num_std[j] = sd if sd and np.isfinite(sd) and sd > 0 else 1.0

    def render_soft(self, state: torch.Tensor, engine: NeuroRuleEngine,
                    cell_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Downstream features with the masked cells taken from the soft state."""
        b = cell_ids.shape[0]
        x = torch.zeros(b, self.dim, device=cell_ids.device)
        for j, disc in enumerate(self.discretizers):
            if j in self.num_cols:
                vals = disc.inverse_transform_numeric(cell_ids[:, j].cpu().numpy())
                vals = np.where(np.isfinite(vals), vals, self.num_mean[j])
                z = (vals - self.num_mean[j]) / self.num_std[j]
                x[:, self.num_index[j]] = torch.as_tensor(z, dtype=x.dtype, device=x.device)
        ci = 0
        for j, disc in enumerate(self.discretizers):
            if j in self.num_cols:
                continue
            m = self.cat_maps[ci]
            off = self.cat_offsets[ci]
            ci += 1
            if not m:
                continue
            id_to_k = {disc.domain[v]: k for v, k in m.items() if v in disc.domain}
            ids = cell_ids[:, j].cpu().numpy()
            ks = np.array([id_to_k.get(int(t), -1) for t in ids])
            hit = ks >= 0
            if hit.any():
                x[torch.as_tensor(np.nonzero(hit)[0], device=x.device),
                  off + torch.as_tensor(ks[hit], device=x.device)] = 1.0

        for j, disc in enumerate(self.discretizers):
            masked = mask[:, j] > 0.5
            if not bool(masked.any()):
                continue
            if j in self.num_cols:
                ev = engine.soft_render_numeric(state, j)
                z = (ev - self.num_mean[j]) / self.num_std[j]
                x[masked, self.num_index[j]] = z[masked]
            else:
                ci = self.cat_pos[j]
                m = self.cat_maps[ci]
                off = self.cat_offsets[ci]
                if not m:
                    continue
                vals_sorted = sorted(m, key=m.get)
                target_cols = torch.as_tensor([m[v] for v in vals_sorted], dtype=torch.long,
                                              device=x.device)
                tok_ids = torch.as_tensor([disc.domain.get(v, -1) for v in vals_sorted],
                                          dtype=torch.long, device=x.device)
                keep = tok_ids >= 0
                p = state[masked, j, :]
                soft = torch.zeros(int(masked.sum()), len(m), dtype=x.dtype, device=x.device)
                soft.index_add_(1, target_cols[keep], p[:, tok_ids[keep]])
                x[masked, off: off + len(m)] = soft
        return x


class NeuroRuleRepairer:
    """Fits the rule set on the dirty table and repairs the masked cells."""

    def __init__(self, config: RepairConfig = None):
        self.cfg = config or RepairConfig()
        self.device = self._resolve_device(self.cfg.device)
        self.engine: Optional[NeuroRuleEngine] = None
        self.discretizers: List[ColumnDiscretizer] = []
        self.encoder: Optional[_TableEncoder] = None
        self.num_cols: List[int] = []
        self.meta: Dict = {}
        self.indicator_: Optional[np.ndarray] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(self.discretizers)), dtype=np.int64)
        for j, disc in enumerate(self.discretizers):
            out[:, j] = disc.transform(df.iloc[:, j])
        return out

    def fit(
        self,
        train_df: pd.DataFrame,
        train_mask: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        numerical_cols: Optional[List[str]] = None,
    ) -> "NeuroRuleRepairer":
        """Fit the rule set on the dirty table under the joint objective."""
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        t0 = time.time()

        names = list(train_df.columns)
        self.num_cols = [j for j, c in enumerate(names) if numerical_cols and c in numerical_cols]
        cat_bins = cfg.cat_bins or cfg.n_bins
        self.discretizers = [
            ColumnDiscretizer.fit(col, train_df.iloc[:, j], j in self.num_cols,
                                cfg.n_bins if j in self.num_cols else cat_bins)
            for j, col in enumerate(names)
        ]
        self.encoder = _TableEncoder(self.discretizers, self.num_cols, train_df)

        engine = NeuroRuleEngine(self.discretizers, cfg).to(self.device)

        ids = torch.as_tensor(self._transform(train_df), device=self.device)
        msk = torch.as_tensor(train_mask.astype(np.float32), device=self.device)
        engine.init_binding_stats(ids, (1.0 - msk))
        self.num_domain = {}
        for j in self.num_cols:
            keep = train_df.iloc[:, j][train_mask[:, j] == 0]
            num = pd.to_numeric(keep, errors="coerce").dropna()
            uniq = {float(v): str(v) for v in num.unique()}
            self.num_domain[j] = (np.array(sorted(uniq)), uniq)

        use_task = cfg.use_task_loss and y_train is not None and self.encoder.dim > 0
        task_model = None
        y = None
        if use_task:
            y = torch.as_tensor(np.asarray(y_train, dtype=np.int64), device=self.device)
            n_classes = int(y.max().item()) + 1
            task_model = _GuideMLP(self.encoder.dim, n_classes).to(self.device)
            opt = torch.optim.Adam(
                [{"params": engine.parameters(), "lr": cfg.lr},
                 {"params": task_model.parameters(), "lr": TASK_LR}]
            )
        else:
            opt = torch.optim.Adam(engine.parameters(), lr=cfg.lr)

        best_state, best_score, best_epoch = None, float("inf"), -1
        fallback_state, fallback_score, fallback_epoch = None, float("inf"), -1
        epochs_run = 0
        # best-state selection opens only after the β warm-up
        sel_from = cfg.beta_warm_epochs if use_task else 0
        for epoch in range(cfg.epochs):
            epochs_run = epoch + 1
            beta = 0.0 if not use_task else min(
                cfg.beta_max,
                cfg.beta0 + (cfg.beta_max - cfg.beta0) * min(
                    1.0, epoch / max(1, cfg.beta_warm_epochs)),
            )
            depth = int(cfg.depth)

            perm = torch.randperm(ids.shape[0], device=self.device)
            ep_rep, nb = 0.0, 0
            for s in range(0, ids.shape[0], cfg.batch_size):
                idx = perm[s: s + cfg.batch_size]
                b_ids, b_msk = ids[idx], msk[idx]
                # hold out random clean cells as L_rep supervision
                aug = (torch.rand_like(b_msk) < AUG_MASK_PROB) & (b_msk < 0.5)
                eff = b_msk + aug.to(b_msk.dtype)
                fwd_ids = b_ids.clone()
                fwd_ids[eff > 0.5] = MISSING
                out = engine(fwd_ids, eff, num_steps=depth)
                # cross-entropy over held-out cells (value-id targets only)
                keep = (aug > 0.5) & (b_ids >= OFFSET)
                if bool(keep.any()):
                    l_rep = F.cross_entropy(out["state"][keep], b_ids[keep])
                else:
                    l_rep = torch.zeros((), device=self.device)
                loss = l_rep if cfg.use_rep_loss \
                    else torch.zeros((), device=self.device)
                if use_task and beta > 0:
                    x_soft = self.encoder.render_soft(
                        out["state"], engine, b_ids, eff)
                    loss = loss + beta * F.cross_entropy(task_model(x_soft), y[idx])
                loss = loss + engine.sparsity_penalty()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in engine.parameters() if p.requires_grad], max_norm=5.0
                )
                opt.step()
                ep_rep += float(l_rep.item()) * len(idx)
                nb += len(idx)
            score = ep_rep / max(nb, 1)

            if score < fallback_score - 1e-5:
                fallback_score, fallback_epoch = score, epoch
                fallback_state = copy.deepcopy(
                    {"engine": engine.state_dict(),
                     "task": task_model.state_dict() if task_model is not None else None}
                )
            if epoch >= sel_from and score < best_score - 1e-5:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(
                    {"engine": engine.state_dict(),
                     "task": task_model.state_dict() if task_model is not None else None}
                )
            min_epochs = cfg.beta_warm_epochs + cfg.patience
            if epoch + 1 >= min_epochs and epoch - best_epoch >= cfg.patience:
                break

        if best_state is None:
            best_state, best_epoch = fallback_state, fallback_epoch
        if best_state is not None:
            engine.load_state_dict(best_state["engine"])
            if task_model is not None and best_state["task"] is not None:
                task_model.load_state_dict(best_state["task"])
        best_epoch = best_epoch + 1 if best_epoch is not None else None

        self.engine = engine
        runtime = time.time() - t0
        self.meta.update({
            "T_used": int(cfg.depth),
            "runtime_sec": runtime,
            "peak_memory_mb": (torch.cuda.max_memory_allocated() / (1024 ** 2)
                               if self.device == "cuda" else
                               resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
        })
        return self

    def repair(self, df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        """Rewrite the masked cells with the argmax of the final chaining state."""
        depth = int(self.cfg.depth)
        ids = torch.as_tensor(self._transform(df), device=self.device)
        msk = torch.as_tensor(mask.astype(np.float32), device=self.device)
        ids = ids.clone()
        ids[msk > 0.5] = MISSING
        # rows are independent and the argmax is per row: chunking is exact
        n, d = ids.shape
        pred = torch.empty((n, d), dtype=torch.long, device=self.device)
        step = max(1, REPAIR_STATE_BYTES // (4 * self.engine.v_max * d))
        with torch.no_grad():
            for s in range(0, n, step):
                out = self.engine(ids[s:s + step], msk[s:s + step],
                                  num_steps=depth)
                pred[s:s + step] = out["state"].argmax(dim=-1)
        pred = pred.cpu().numpy()
        indicator = np.zeros(mask.shape, dtype=np.int64)

        repaired = df.astype(object).copy()
        for j, disc in enumerate(self.discretizers):
            col_mask = mask[:, j].astype(bool)
            if not col_mask.any():
                continue
            dirty = df.iloc[:, j].astype(object).to_numpy()
            obs_tok = disc.transform(dirty)
            rendered = np.array(dirty, dtype=object)
            if j in self.num_cols:
                vals = disc.inverse_transform_numeric(pred[:, j])
                fin = np.isfinite(vals)
                # same bin as observed: canonical re-render, no value change
                same_bin = col_mask & fin & (obs_tok == pred[:, j])
                canonical = np.array(
                    [f"{v:.10g}" if np.isfinite(v) else "" for v in
                     pd.to_numeric(pd.Series(dirty), errors="coerce").to_numpy(dtype=float)],
                    dtype=object)
                rendered[same_bin] = canonical[same_bin]
                snap = col_mask & fin & (obs_tok != pred[:, j])
                indicator[snap, j] = 1
                if snap.any():
                    rendered[snap] = [self._snap_number(j, v) for v in vals[snap]]
            else:
                dec = np.array(disc.inverse_transform(pred[:, j]), dtype=object)
                value_tok = col_mask & (pred[:, j] >= OFFSET) & (obs_tok != pred[:, j])
                indicator[value_tok, j] = 1
                rendered[value_tok] = dec[value_tok]
            repaired.iloc[col_mask, j] = rendered[col_mask]
        self.indicator_ = indicator
        return repaired

    def _snap_number(self, j: int, value: float) -> str:
        """Render a repaired numeric as the nearest observed domain string."""
        keys, strings = self.num_domain[j]
        if not np.isfinite(value):
            return "empty"
        if len(keys) == 0:
            centers = self.discretizers[j].bin_centers
            pool = np.array((list(centers) if centers is not None else []) + [value],
                            dtype=float)
            if np.allclose(pool.round(), pool, atol=1e-9):
                return str(int(round(value)))
            return f"{value:g}"
        i = int(np.searchsorted(keys, value))
        best = min((abs(keys[c] - value), c) for c in (i - 1, i)
                   if 0 <= c < len(keys))[1]
        return strings[float(keys[best])]


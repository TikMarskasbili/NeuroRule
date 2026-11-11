"""Differentiable forward chaining over weighted logic rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .discretizer import OFFSET, PAD, ColumnDiscretizer

# inference conditioning sharpness μ (fixed)
MU = 1.5


@dataclass
class EngineConfig:
    """Hyperparameters of the chaining engine."""

    n_bins: int = 16              # numeric quantile bins (categorical top-V when cat_bins is None)
    depth: int = 1                # fixed chaining depth T
    rho: float = 0.5              # binarization threshold of σ(A)
    min_support: float = 3        # rows needed before a clause exists
    max_pairs: int = 16           # pair candidates kept per head (0 = off)
    pair_support: int = 10        # rows before a literal pair is mined
    lambda_sparse: float = 0.01   # graph sparsity weight


class NeuroRuleEngine(nn.Module):
    """Weighted logic rules executed by chaining (W/A clauses and W2/A2 pairs)."""

    def __init__(self, discretizers: List[ColumnDiscretizer], cfg: EngineConfig = None):
        super().__init__()
        self.discretizers = discretizers
        self.cols = [t.col for t in discretizers]
        self.d = len(discretizers)
        self.cfg = cfg or EngineConfig()

        domain_sizes = [t.domain_size for t in discretizers]
        self.register_buffer("domain_sizes", torch.tensor(domain_sizes, dtype=torch.long))
        self.v_max = int(max(domain_sizes))

        # flat predicate space: body literal (column, value id) pairs
        self.register_buffer("pair_col", torch.repeat_interleave(
            torch.arange(self.d, dtype=torch.long), torch.tensor(domain_sizes)))
        self.register_buffer("pair_ids", torch.cat(
            [torch.arange(vs) for vs in domain_sizes]))
        self.n_pairs = int(self.pair_col.shape[0])
        self.register_buffer("col_start",
                             torch.cumsum(torch.tensor(domain_sizes), 0) - torch.tensor(domain_sizes))

        # valid-value mask over the padded domain: (d, v_max) in {0,1}.
        valid = torch.zeros(self.d, self.v_max)
        for c, vs in enumerate(domain_sizes):
            valid[c, :vs] = 1.0
            valid[c, PAD] = 0.0          # PAD is never a value
        self.register_buffer("valid_mask", valid)
        # head-eligible mask: only value ids (id >= OFFSET) are inference
        # targets; MISSING/UNSEEN stay legal premises but never head outcomes.
        head = torch.zeros(self.d, self.v_max)
        for c, vs in enumerate(domain_sizes):
            head[c, OFFSET:vs] = 1.0
        self.register_buffer("head_mask", head)

        # Clause weights W: (d, v_max, P) — log-PMI per head atom given the
        # flat body literal.
        self.rule_w = nn.Parameter(torch.zeros(self.d, self.v_max, self.n_pairs))

        # Premise affinities A: (d, d) — A[j, c] = weight of column c as a
        # premise of head j. Self-premises are forbidden at read time.
        self.rule_a = nn.Parameter(torch.zeros(self.d, self.d))

        # Two-condition rules (order 2): per head up to max_pairs mined
        # body-literal pairs. rule_w2 (d, v_max, M) = pair clause weights,
        # rule_a2 (d, M) = pair affinities; literal indices and validity
        # mask are filled by mining (empty until init_binding_stats runs).
        m_max = int(self.cfg.max_pairs)
        self.rule_w2 = nn.Parameter(torch.zeros(self.d, self.v_max, m_max))
        self.rule_a2 = nn.Parameter(torch.zeros(self.d, m_max))
        self.register_buffer("pair_lit1", torch.zeros(self.d, m_max, dtype=torch.long))
        self.register_buffer("pair_lit2", torch.zeros(self.d, m_max, dtype=torch.long))
        self.register_buffer("pair_mask", torch.zeros(self.d, m_max, dtype=torch.bool))

        # value-prior log per head (set by init_binding_stats)
        self.register_buffer("head_log_prior", torch.zeros(self.d, self.v_max))

    @torch.no_grad()
    def init_binding_stats(self, cell_ids: torch.Tensor, observation: torch.Tensor):
        """Initialize W, A, priors and the mined pairs from the clean-cell statistics."""
        ids = cell_ids.detach().cpu().numpy()
        obs = observation.detach().cpu().numpy().astype(bool)
        W = torch.zeros(self.d, self.v_max, self.n_pairs)
        A = torch.zeros(self.d, self.d)
        prior_np = np.zeros((self.d, self.v_max), dtype=np.float64)

        seen_per_col = {}
        for j in range(self.d):
            seen = obs[:, j] & (ids[:, j] >= OFFSET)
            seen_per_col[j] = seen
            cnt = np.bincount(ids[seen, j], minlength=self.v_max).astype(float)
            p = cnt / max(1.0, cnt.sum())
            prior_np[j] = np.log(p + 1e-8)
        prior = torch.as_tensor(prior_np, dtype=torch.float32,
                                device=self.head_mask.device)
        prior = prior.masked_fill(self.head_mask == 0, -1e9)
        self.head_log_prior.copy_(prior)

        valid_j = self.valid_mask.detach().cpu().numpy().astype(bool)
        head_j = self.head_mask.detach().cpu().numpy().astype(bool)
        col_start = self.col_start.detach().cpu().numpy()
        for j in range(self.d):
            hs = seen_per_col[j]
            if hs.sum() < 5:
                continue
            base = np.exp(prior_np[j])
            for c in range(self.d):
                if c == j:
                    continue
                pc = seen_per_col[c]
                both = hs & pc
                if both.sum() < 3:
                    continue
                lit_base = both & (ids[:, c] != PAD)
                pmi_mass = 0.0
                for k in range(int(self.domain_sizes[c])):
                    lit = lit_base & (ids[:, c] == k)
                    m = int(lit.sum())
                    if m < self.cfg.min_support:
                        continue
                    q = np.bincount(ids[lit, j], minlength=self.v_max).astype(float)
                    q[~valid_j[j]] = 0.0
                    q[~head_j[j]] = 0.0          # heads are value ids only
                    m_val = float(q.sum())
                    if m_val < 1.0:
                        continue                  # literal never co-occurs with a value
                    q = q / m_val
                    log_pmi = np.log(q + 1e-8) - prior_np[j]
                    log_pmi[~head_j[j]] = 0.0
                    log_pmi = np.clip(log_pmi, -5.0, 5.0)
                    W[j, :, col_start[c] + k] = torch.as_tensor(
                        log_pmi, dtype=torch.float32)
                    pmi_mass += 0.5 * float(np.abs(q - base).sum()) * m
                # A[j, c]: mean TV distance per observed row, scaled to [0, 1]
                A[j, c] = float(np.clip(3.0 * pmi_mass / float(both.sum()), 0.0, 1.0))
        self.rule_w.copy_(W)
        self.rule_a.copy_(A)
        self._mine_pairs(cell_ids, observation)

    @torch.no_grad()
    def _mine_pairs(self, cell_ids: torch.Tensor, observation: torch.Tensor):
        """Keep per head the top pairs whose joint PMI beats either literal by a margin."""
        if int(self.cfg.max_pairs) <= 0 or self.d < 3:
            return
        ids = cell_ids.detach().cpu().numpy()
        obs = observation.detach().cpu().numpy().astype(bool)
        unary = self.rule_w.detach().cpu().numpy()          # (d, v_max, P)
        prior = self.head_log_prior.detach().cpu().numpy()  # (d, v_max)
        vs = self.domain_sizes.cpu().numpy()
        col_start = self.col_start.cpu().numpy()
        min_sup = int(self.cfg.pair_support)
        margin_eps = 0.25                                   # min interaction (log)
        cand = [[] for _ in range(self.d)]

        head_tok_mask = self.head_mask.detach().cpu().numpy().astype(bool)
        priors = {}
        for j in range(self.d):
            seen = obs[:, j] & (ids[:, j] >= OFFSET)
            cnt = np.bincount(ids[seen, j], minlength=int(vs[j])).astype(float)
            p = np.zeros(int(vs[j]))
            p[head_tok_mask[j, : int(vs[j])]] = (
                cnt[head_tok_mask[j, : int(vs[j])]] / max(1.0, cnt.sum()))
            priors[j] = p

        for c1 in range(self.d):
            for c2 in range(c1 + 1, self.d):
                both = obs[:, c1] & obs[:, c2] \
                    & (ids[:, c1] != PAD) & (ids[:, c2] != PAD)
                if int(both.sum()) < 2 * min_sup:
                    continue
                idx = np.nonzero(both)[0]
                code = ids[idx, c1] * int(vs[c2]) + ids[idx, c2]
                n_pairs = int(vs[c1]) * int(vs[c2])
                for j in range(self.d):
                    if j == c1 or j == c2:
                        continue
                    sel = obs[idx, j] & (ids[idx, j] >= OFFSET)
                    if int(sel.sum()) < min_sup:
                        continue
                    rows = idx[sel]
                    vj = int(vs[j])
                    counts = np.bincount(
                        code[sel] * vj + ids[rows, j],
                        minlength=n_pairs * vj).astype(float)
                    cnt3 = counts.reshape(n_pairs, vj)
                    pair_n = cnt3.sum(axis=1)
                    alive = pair_n >= min_sup
                    if not alive.any():
                        continue
                    q = cnt3[alive] / np.maximum(pair_n[alive, None], 1.0)
                    pmi = np.log(q + 1e-8) - np.log(priors[j] + 1e-8)
                    pmi[:, ~head_tok_mask[j, :vj]] = 0.0
                    np.clip(pmi, -5.0, 5.0, out=pmi)
                    pi = np.nonzero(alive)[0]
                    k1 = pi // int(vs[c2])
                    k2 = pi % int(vs[c2])
                    ua = unary[j][:, col_start[c1] + k1].max(axis=0)
                    ub = unary[j][:, col_start[c2] + k2].max(axis=0)
                    margin = pmi.max(axis=1) - np.maximum(ua, ub)
                    keep = margin > margin_eps
                    for r in np.nonzero(keep)[0]:
                        cand[j].append((float(margin[r]), int(c1), int(k1[r]),
                                        int(c2), int(k2[r]), pmi[r].copy()))

        m_max = int(self.cfg.max_pairs)
        for j in range(self.d):
            if not cand[j]:
                continue
            cand[j].sort(key=lambda x: -x[0])
            for m, (mg, c1, k1, c2, k2, pmi) in enumerate(cand[j][:m_max]):
                self.pair_lit1[j, m] = col_start[c1] + k1
                self.pair_lit2[j, m] = col_start[c2] + k2
                self.pair_mask[j, m] = True
                self.rule_w2[j, : int(vs[j]), m] = torch.as_tensor(
                    pmi, dtype=torch.float32)
                self.rule_a2[j, m] = float(np.clip(mg, 0.0, 2.0))

    def dep_matrix(self) -> torch.Tensor:
        """Soft adjacency σ(A) ∈ [0,1]^{d×d} of the premise graph."""
        a = torch.sigmoid(self.rule_a)
        return a * (1.0 - torch.eye(self.d, device=a.device))

    def sparsity_penalty(self) -> torch.Tensor:
        a = self.dep_matrix()
        pen = self.cfg.lambda_sparse * a.mean()
        if bool(self.pair_mask.any()):
            a2 = torch.sigmoid(self.rule_a2) * self.pair_mask.float()
            pen = pen + self.cfg.lambda_sparse * (
                a2.sum() / self.pair_mask.float().sum().clamp_min(1.0))
        return pen

    def forward(
        self,
        cell_ids: torch.Tensor,       # (B, d) observed value ids
        mask: torch.Tensor,           # (B, d) float: 1 = cell masked for repair
        num_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if num_steps is None:
            num_steps = int(self.cfg.depth)

        b = cell_ids.shape[0]
        v = torch.zeros(b, self.d, self.v_max, device=cell_ids.device)
        v.scatter_(2, cell_ids.unsqueeze(-1), 1.0)
        v = v * self.valid_mask.unsqueeze(0)                           # PAD ids carry no mass
        obs = (1.0 - mask).unsqueeze(-1)
        masked = mask.unsqueeze(-1)

        # effective clause weights: premise affinities folded in, self-premises forbidden
        no_self = 1.0 - torch.eye(self.d, device=self.rule_a.device)
        a_scale = (self.rule_a * no_self)[:, self.pair_col]            # (d, P)
        w_eff = self.rule_w * a_scale.unsqueeze(1) * self.valid_mask.unsqueeze(-1)
        w2d = w_eff.reshape(self.d * self.v_max, self.n_pairs)
        prior = self.head_log_prior                                     # (d, v_max)

        for _ in range(num_steps):
            # flat predicate-truth vector over the ground atoms
            vf = torch.cat([v[:, c, : int(self.domain_sizes[c])]
                            for c in range(self.d)], dim=1)
            prem = (vf @ w2d.T).reshape(b, self.d, self.v_max)
            if bool(self.pair_mask.any()):
                # order-2 vote: pair truth = product t-norm of both literals
                pmask = self.pair_mask.float()
                truth = (vf[:, self.pair_lit1] * vf[:, self.pair_lit2]) * pmask
                gate = torch.sigmoid(self.rule_a2) * pmask
                prem = prem + torch.einsum(
                    'bdm,dtm->bdt', truth, self.rule_w2 * gate.unsqueeze(1))
            logits = prior.unsqueeze(0) + MU * prem                     # (B, d, v_max)
            # only value ids compete as head outcomes
            logits = logits.masked_fill(self.head_mask.unsqueeze(0) == 0, -1e9)
            v_new = torch.softmax(logits, dim=-1)
            v = obs * v + masked * v_new

        return {"state": v}

    def soft_render_numeric(self, state: torch.Tensor, col: int) -> torch.Tensor:
        """Expected original-scale value of a numerical column: (B,)."""
        disc = self.discretizers[col]
        means = disc.bin_means if disc.bin_means is not None else disc.bin_centers
        n_bins = len(means)
        p = state[:, col, OFFSET: OFFSET + n_bins]
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        centers = torch.as_tensor(means, dtype=state.dtype, device=state.device)
        return (p * centers.unsqueeze(0)).sum(dim=-1)

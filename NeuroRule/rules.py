"""Logic-rule extraction from the trained clause weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

from .engine import NeuroRuleEngine
from .discretizer import MISSING, ColumnDiscretizer


@dataclass
class LogicRule:
    head_col: str
    body: List[Tuple[str, str]]       # [(column, value), ...]
    head_value: str = ""
    weight: float = 0.0

    def render(self) -> str:
        if self.body:
            conj = " AND ".join(f"{c}={v}" for c, v in self.body)
            return f"{self.head_col}({self.head_value or 'any'}) <- {conj}"
        return f"{self.head_col}({self.head_value or 'any'}) <- true"


@torch.no_grad()
def extract_rules(
    engine: NeuroRuleEngine,
    discretizers: List[ColumnDiscretizer],
    max_rules_per_head: int = 4,
) -> List[LogicRule]:
    """Extract the weighted logic-rule set from the trained clause weights."""
    a = torch.sigmoid(engine.rule_a).cpu()
    w = engine.rule_w.detach().cpu()
    prior = engine.head_log_prior.detach().cpu()
    pair_col = engine.pair_col.cpu()
    pair_ids = engine.pair_ids.cpu()
    thr = engine.cfg.rho

    rules: List[LogicRule] = []
    for j in range(engine.d):
        per_head: List[Tuple[float, LogicRule]] = []
        for c in range(engine.d):
            if c == j:
                continue
            strength = float(a[j, c])
            if strength < thr:
                continue
            span = (engine.col_start[c].item(),
                    engine.col_start[c].item() + int(engine.domain_sizes[c]))
            # clause strength of literal (c, k): its best clause weight per head atom
            strengths = w[j, :, span[0]: span[1]].max(dim=0).values
            valid_lit = pair_ids[span[0]: span[1]] >= 3                 # value ids only
            strengths = strengths * valid_lit.float()
            order = torch.argsort(strengths, descending=True)
            for rank, k in enumerate(order[: max_rules_per_head]):
                if float(strengths[k]) <= 0:
                    break
                p = span[0] + int(k)
                head_tok_id = int((prior[j] + w[j, :, p]).argmax())
                head_str = _token_str(discretizers[j], head_tok_id)
                val_str = _token_str(discretizers[c], int(pair_ids[p]))
                if val_str == "" or head_str == "":
                    continue
                rule = LogicRule(
                    head_col=discretizers[j].col,
                    body=[(discretizers[c].col, val_str)],
                    head_value=head_str,
                    weight=strength,
                )
                per_head.append((rule.weight, rule))
        # two-condition clauses surviving the affinity threshold
        a2 = torch.sigmoid(engine.rule_a2).cpu()
        w2 = engine.rule_w2.detach().cpu()
        for m in range(a2.shape[1]):
            if not bool(engine.pair_mask[j, m]) or float(a2[j, m]) < thr:
                continue
            strengths = w2[j, :, m]
            if float(strengths.max()) <= 0:
                continue
            head_tok_id = int((prior[j] + strengths).argmax())
            head_str = _token_str(discretizers[j], head_tok_id)
            body = []
            for p in (int(engine.pair_lit1[j, m]), int(engine.pair_lit2[j, m])):
                c, k = int(pair_col[p]), int(pair_ids[p])
                val_str = _token_str(discretizers[c], k)
                if val_str == "":
                    body = None
                    break
                body.append((discretizers[c].col, val_str))
            if body is None or head_str == "":
                continue
            rule = LogicRule(
                head_col=discretizers[j].col,
                body=body,
                head_value=head_str,
                weight=float(a2[j, m]),
            )
            per_head.append((rule.weight, rule))
        per_head.sort(key=lambda wr: -wr[0])
        rules += [r for _, r in per_head[: max_rules_per_head]]
    return rules


def _token_str(disc: ColumnDiscretizer, value_id: int) -> str:
    if value_id in disc.inv_domain:
        return disc.inv_domain[value_id]
    if value_id == MISSING:
        return "MISSING"
    return ""                                                         # PAD / UNSEEN

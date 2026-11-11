"""NeuroRule experiment driver: the paper's main-benchmark protocol."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from NeuroRule.cleaner import NeuroRuleRepairer, RepairConfig
from NeuroRule.dataset import load_dataset
from NeuroRule.detection import generate_mask
from NeuroRule.rules import extract_rules

SEED = 43
EPOCHS = 30
DEVICE = "cpu"
THREADS = 8

MAIN_DATASETS = [
    "Hospital", "Flights", "Beers", "Titanic", "Marketing",
    "Bank", "Restaurant", "Airbnb", "Sensor", "USCensus",
]

# per-dataset RepairConfig overrides; a "dataset/detector" key overrides further
DATASET_CONFIGS: Dict[str, Dict] = {
    "Hospital": {"beta_max": 1.0, "lr": 0.03, "pair_support": 20,
                 "min_support": 5},
    "Flights": {"beta_max": 2.5, "cat_bins": 320, "lambda_sparse": 0.03,
                "lr": 0.05, "min_support": 5},
    "Beers": {},
    "Restaurant": {"cat_bins": 640},
    "Airbnb": {},
    "Titanic": {"max_pairs": 0},
    "Marketing": {},
    "Bank": {},
    "USCensus": {},
    "Sensor": {},
}

# loss-combination arms (the ablation grid); "full" is the paper objective
VARIANTS: Dict[str, Dict] = {
    "full": {},
    "wo_task": {"use_task_loss": False},
    "wo_sparsity": {"lambda_sparse": 0.0},
    "wo_rep": {"use_rep_loss": False},
    "rep_only": {"use_task_loss": False, "lambda_sparse": 0.0},
    "task_only": {"use_rep_loss": False, "lambda_sparse": 0.0},
}


def build_config(dataset: str, detector: str, variant: str) -> RepairConfig:
    """Merge order (later wins): run fields, registry, detector entry, variant."""
    overrides = dict(epochs=EPOCHS, seed=SEED, device=DEVICE)
    overrides.update(DATASET_CONFIGS.get(dataset, {}))
    overrides.update(DATASET_CONFIGS.get(f"{dataset}/{detector}", {}))
    overrides.update(VARIANTS[variant])
    return RepairConfig(**overrides)


# ---------------------------------------------------------------------------
# task score: stratified k-fold CV of LR / MLP over the repaired table
# ---------------------------------------------------------------------------
CAT_DOMAIN_CAP = 16


class _TorchLR(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


class _TorchMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def _fit_torch_model(model, x_tr, y_tr, x_val, y_val, seed=43,
                     lr=1e-3, epochs=500, batch_size=32, patience=30,
                     device: str = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    xt = torch.as_tensor(x_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.long, device=device)

    best_loss, best_state, bad = float("inf"), None, 0
    n = len(xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s: s + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(F.cross_entropy(model(xv), yv).item())
        if vloss < best_loss - 1e-5:
            best_loss, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, device


def evaluate_task_models_cv(
    repaired: pd.DataFrame,
    y: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    k: int = 5,
    seed: int = 43,
    device: str = None,
) -> Dict[str, Dict[str, float]]:
    """Stratified k-fold CV of LR and MLP over the repaired table."""
    from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

    def to_matrix(df, scaler=None, categories=None):
        parts = []
        if num_cols:
            num = df[num_cols].apply(pd.to_numeric, errors="coerce")
            num = num.fillna(num.mean(numeric_only=True))
            if num.isna().any().any():
                num = num.fillna(0.0)
            if scaler is None:
                scaler = StandardScaler().fit(num)
            parts.append(scaler.transform(num))
        for c in cat_cols:
            s = df[c].astype(str).str.strip()
            codes = s.map(categories[c]).fillna(-1).to_numpy(dtype=int)
            onehot = np.zeros((len(s), len(categories[c])))
            hit = codes >= 0
            onehot[np.arange(len(s))[hit], codes[hit]] = 1.0
            parts.append(onehot)
        return np.concatenate(parts, axis=1), scaler

    n_classes = int(y.max()) + 1
    per_model_acc: Dict[str, List[float]] = {"LR": [], "MLP": []}
    per_model_f1: Dict[str, List[float]] = {"LR": [], "MLP": []}

    counts = np.bincount(y)
    if counts[counts > 0].min() >= 2:
        k = max(2, min(k, int(counts[counts > 0].min())))
        folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed) \
            .split(np.zeros(len(y)), y)
    else:
        folds = KFold(n_splits=k, shuffle=True, random_state=seed) \
            .split(np.zeros(len(y)))

    for tr_idx, te_idx in folds:
        if len(tr_idx) > 60:
            try:
                tr_idx, val_idx = train_test_split(
                    tr_idx, test_size=0.1, random_state=seed, stratify=y[tr_idx])
            except ValueError:
                tr_idx, val_idx = train_test_split(
                    tr_idx, test_size=0.1, random_state=seed)
        else:
            val_idx = tr_idx
        tr_df = repaired.iloc[tr_idx].reset_index(drop=True)
        val_df = repaired.iloc[val_idx].reset_index(drop=True)
        te_df = repaired.iloc[te_idx].reset_index(drop=True)
        categories = {}
        for c in cat_cols:
            vc = tr_df[c].astype(str).str.strip().value_counts()
            categories[c] = {v: i for i, v in enumerate(list(vc.index[:CAT_DOMAIN_CAP]))}
        x_tr, scaler = to_matrix(tr_df, categories=categories)
        x_val, _ = to_matrix(val_df, scaler, categories)
        x_te, _ = to_matrix(te_df, scaler, categories)
        y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]

        for i, (name, make) in enumerate((
            ("LR", lambda: _TorchLR(x_tr.shape[1], n_classes)),
            ("MLP", lambda: _TorchMLP(x_tr.shape[1], n_classes)),
        )):
            net, eval_device = _fit_torch_model(
                make(), x_tr, y_tr, x_val, y_val, seed=seed + i, device=device)
            with torch.no_grad():
                logits = net(torch.as_tensor(x_te, dtype=torch.float32,
                                             device=eval_device))
                pred = logits.argmax(dim=-1).cpu().numpy()
            per_model_acc[name].append(float(accuracy_score(y_te, pred)))
            per_model_f1[name].append(float(
                f1_score(y_te, pred, average="weighted", zero_division=0)))

    per_model = {name: {"acc": float(np.mean(per_model_acc[name])),
                        "f1": float(np.mean(per_model_f1[name]))}
                 for name in ("LR", "MLP")}
    per_model["avg"] = {"acc": float(np.mean([m["acc"] for m in per_model.values()])),
                        "f1": float(np.mean([m["f1"] for m in per_model.values()]))}
    per_model["best"] = {
        "acc": max(m["acc"] for m in [per_model["LR"], per_model["MLP"]]),
        "f1": max(m["f1"] for m in [per_model["LR"], per_model["MLP"]]),
    }
    return per_model


# ---------------------------------------------------------------------------
# cell-level repair metrics
# ---------------------------------------------------------------------------
def _num_or_none(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _match(a: object, b: object) -> bool:
    sa, sb = str(a), str(b)
    if sa == sb:
        return True
    va, vb = _num_or_none(sa), _num_or_none(sb)
    return va is not None and vb is not None and abs(va - vb) <= 1e-9 * max(1.0, abs(va), abs(vb))


def repair_cell_metrics(
    repaired: pd.DataFrame,
    dirty: pd.DataFrame,
    clean: Optional[pd.DataFrame],
    mask: np.ndarray,
    numerical_cols: Optional[list] = None,
    no_gold: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Cell-level repair precision / recall / F1 over the masked cells."""
    if clean is None:
        return {"repair_p": None, "repair_r": None, "repair_f1": None}

    flat_mask = mask.astype(bool)
    if no_gold is not None:
        flat_mask = flat_mask & ~no_gold.astype(bool)
    r_vals = repaired.values.astype(str)
    d_vals = dirty.values.astype(str)
    c_vals = clean.values.astype(str)

    vmatch = np.vectorize(_match, otypes=[bool])
    correct = np.zeros(mask.shape, dtype=bool)
    is_error = np.zeros(mask.shape, dtype=bool)
    rewritten = np.zeros(mask.shape, dtype=bool)
    for j in range(mask.shape[1]):
        col = flat_mask[:, j]
        if not col.any():
            continue
        correct[:, j] = col & vmatch(r_vals[:, j], c_vals[:, j])
        is_error[:, j] = col & ~vmatch(d_vals[:, j], c_vals[:, j])
        rewritten[:, j] = col & ~vmatch(r_vals[:, j], d_vals[:, j])

    n_pred = int((rewritten & flat_mask).sum())
    n_actual = int((is_error & flat_mask).sum())
    tp = int((correct & is_error & flat_mask).sum())
    if n_actual == 0:
        return {"repair_p": None, "repair_r": None, "repair_f1": None}
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_actual
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"repair_p": precision, "repair_r": recall, "repair_f1": f1}


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def run_one(dataset: str, detector: str, variant: str) -> Dict:
    ds = load_dataset(dataset)
    t0 = time.time()
    mask = generate_mask(ds, detector)
    mask_sec = time.time() - t0
    # the mask spans all raw columns; keep the feature columns only
    feat_idx = [ds.dirty.columns.get_loc(c) for c in ds.feature_columns]
    mask = mask[:, feat_idx]
    table = ds.features().reset_index(drop=True)
    y = ds.labels()

    cfg = build_config(dataset, detector, variant)
    repairer = NeuroRuleRepairer(cfg)
    repairer.fit(table, mask, y, numerical_cols=ds.numerical_features)
    repaired = repairer.repair(table, mask)

    task = evaluate_task_models_cv(
        repaired, y,
        num_cols=ds.numerical_features, cat_cols=ds.categorical_features,
        seed=SEED, device=DEVICE,
    )
    clean_tbl = None
    if ds.clean is not None:
        clean_idx = [ds.clean.columns.get_loc(c) for c in ds.feature_columns]
        clean_tbl = ds.clean.iloc[:, clean_idx].reset_index(drop=True)
    no_gold = None
    if ds.nongold_values and clean_tbl is not None:
        no_gold = np.isin(clean_tbl.values.astype(str), ds.nongold_values)
    cell = repair_cell_metrics(
        repaired, table, clean_tbl, mask,
        numerical_cols=ds.numerical_features, no_gold=no_gold,
    )

    rules = extract_rules(repairer.engine, repairer.discretizers)
    meta = repairer.meta
    return {
        "dataset": dataset, "detector": detector, "variant": variant,
        "acc_avg": task["avg"]["acc"], "f1_avg": task["avg"]["f1"],
        "acc_best": task["best"]["acc"], "f1_best": task["best"]["f1"],
        "lr_acc": task["LR"]["acc"], "lr_f1": task["LR"]["f1"],
        "mlp_acc": task["MLP"]["acc"], "mlp_f1": task["MLP"]["f1"],
        **cell,
        "epochs_run": meta["epochs_run"], "best_epoch": meta["best_epoch"],
        "runtime_sec": meta["runtime_sec"], "mask_sec": mask_sec,
        "n_rules": len(rules),
        "n_rows": len(ds.dirty), "n_features": len(ds.feature_columns),
        "mask_rate": float(mask.mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=str, default="all",
                    help="comma list or 'all' (%s)" % ", ".join(MAIN_DATASETS))
    ap.add_argument("--detector", type=str, default="default",
                    help="default | lof | dbscan")
    ap.add_argument("--variants", type=str, default="full",
                    help="comma list from: %s" % ", ".join(VARIANTS))
    ap.add_argument("--output", type=str, default="",
                    help="optional CSV path for the result rows")
    args = ap.parse_args()

    torch.set_num_threads(THREADS)
    datasets = MAIN_DATASETS if args.datasets == "all" \
        else [d.strip() for d in args.datasets.split(",") if d.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant: {v} (known: {', '.join(VARIANTS)})")

    rows = []
    for dataset in datasets:
        for variant in variants:
            row = run_one(dataset, args.detector, variant)
            rows.append(row)
            rf1 = row["repair_f1"]
            print(f"{dataset:12s} {variant:11s} "
                  f"acc {row['acc_avg']:.4f}/{row['acc_best']:.4f}  "
                  f"f1 {row['f1_avg']:.4f}/{row['f1_best']:.4f}  "
                  f"repair_f1 {'-' if rf1 is None else f'{rf1:.4f}'}  "
                  f"rules {row['n_rules']:3d}  "
                  f"{row['runtime_sec']:.0f}s")

    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()

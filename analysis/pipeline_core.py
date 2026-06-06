# -*- coding: utf-8 -*-
"""
pipeline_core.py — re-implementation of the Lim et al. (2026) cross-diagnostic
item-selection framework, extended to a 7th (sleep) domain.

Core pieces (shared by 02_optimize.py and 03_network_shap.py):
  - load_cohort()           load item-level data + labels
  - make_split()            strict 15% lockbox vs 85% development
  - univariate_utility()    U(j,k) = mean(AUROC, AUPRC) of item k for domain j
  - cosine_redundancy()     cosine similarity between standardized item vectors
  - optimize_panel()        joint objective: max Σ U − λ Σ cos, 1 item per domain
  - evaluate_panel()        lockbox AUROC/AUPRC for panel vs anchor-only

The objective reproduces:
    max  Σ_j U(j, k_j)  −  λ Σ_{j<j'} cos( x_{k_j}, x_{k_j'} )
    s.t. exactly one item k_j per domain j, chosen from domain j's own instrument
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")


def load_cohort():
    items = pd.read_csv(os.path.join(DATA, "simulated_cohort_items.csv"))
    meta = pd.read_csv(os.path.join(DATA, "simulated_cohort_meta.csv"))
    return items, meta


def domain_pool(domain):
    """Item codes belonging to a domain's own instrument."""
    return [it[0] for it in C.ITEMS[domain]]


def make_split(n, lockbox_frac=None, seed=None):
    lockbox_frac = C.LOCKBOX_FRAC if lockbox_frac is None else lockbox_frac
    seed = C.SEED if seed is None else seed
    rng = np.random.default_rng(seed + 7)
    idx = rng.permutation(n)
    n_lb = int(round(n * lockbox_frac))
    lockbox = np.sort(idx[:n_lb])
    dev = np.sort(idx[n_lb:])
    return dev, lockbox


def univariate_utility(items, meta, idx, domains=None):
    """U[domain][item_code] = mean(AUROC, AUPRC) using the single item as a
    ranking score for that domain's binary label. Rank-based => no model fit,
    no overfitting; equivalent to CV mean for a univariate predictor."""
    domains = C.DOMAINS if domains is None else domains
    U = {}
    for d in domains:
        y = meta.loc[idx, f"label_{d}"].values
        U[d] = {}
        if y.sum() == 0 or y.sum() == len(y):
            for code in domain_pool(d):
                U[d][code] = 0.0
            continue
        for code in domain_pool(d):
            x = items.loc[idx, code].values.astype(float)
            auroc = roc_auc_score(y, x)
            auprc = average_precision_score(y, x)
            U[d][code] = float(0.5 * (auroc + auprc))
    return U


def cosine_redundancy(items, idx, codes):
    """Cosine similarity matrix between standardized item response vectors."""
    X = items.loc[idx, codes].values.astype(float)
    X = StandardScaler().fit_transform(X)
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1e-9
    S = (X.T @ X) / np.outer(norms, norms)
    return pd.DataFrame(S, index=codes, columns=codes)


def optimize_panel(U, cos_df, lam=None, domains=None, n_restarts=60, seed=0):
    """Coordinate-ascent with random restarts for:
        max Σ_j U[j][k_j] − λ Σ_{j<j'} cos[k_j, k_{j'}]
    Returns dict domain->item_code and the objective value."""
    lam = C.REDUNDANCY_LAMBDA if lam is None else lam
    domains = C.DOMAINS if domains is None else domains
    rng = np.random.default_rng(seed)
    pools = {d: domain_pool(d) for d in domains}

    def objective(sel):
        val = sum(U[d][sel[d]] for d in domains)
        pen = 0.0
        for i, a in enumerate(domains):
            for b in domains[i + 1:]:
                pen += cos_df.loc[sel[a], sel[b]]
        return val - lam * pen

    best_sel, best_val = None, -np.inf
    for r in range(n_restarts):
        # random init
        sel = {d: pools[d][rng.integers(len(pools[d]))] for d in domains}
        improved = True
        while improved:
            improved = False
            for d in domains:
                cur = sel[d]
                best_k, best_local = cur, -np.inf
                others = [sel[o] for o in domains if o != d]
                for k in pools[d]:
                    score = U[d][k] - lam * sum(cos_df.loc[k, o] for o in others)
                    if score > best_local:
                        best_local, best_k = score, k
                if best_k != cur:
                    sel[d] = best_k
                    improved = True
        val = objective(sel)
        if val > best_val:
            best_val, best_sel = val, dict(sel)
    return best_sel, float(best_val)


def _fit_eval(items, meta, dev, lb, feat_codes, domain):
    """Train logistic on dev with feat_codes, evaluate on lockbox for domain label."""
    Xtr = items.loc[dev, feat_codes].values.astype(float)
    Xte = items.loc[lb, feat_codes].values.astype(float)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    ytr = meta.loc[dev, f"label_{domain}"].values
    yte = meta.loc[lb, f"label_{domain}"].values
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), float(average_precision_score(yte, p)), clf, sc


def evaluate_panel(items, meta, dev, lb, panel, domains=None):
    """For each domain: lockbox AUROC/AUPRC using full panel vs anchor-only,
    and the cross-diagnostic gain (panel − anchor)."""
    domains = C.DOMAINS if domains is None else domains
    panel_codes = [panel[d] for d in domains]
    rows = []
    for d in domains:
        a_roc, a_prc, _, _ = _fit_eval(items, meta, dev, lb, [panel[d]], d)
        p_roc, p_prc, _, _ = _fit_eval(items, meta, dev, lb, panel_codes, d)
        rows.append({
            "domain": d, "label": C.DOMAIN_LABEL[d], "anchor_item": panel[d],
            "anchor_auroc": round(a_roc, 3), "anchor_auprc": round(a_prc, 3),
            "panel_auroc": round(p_roc, 3), "panel_auprc": round(p_prc, 3),
            "crossdx_gain_auroc": round(p_roc - a_roc, 3),
        })
    return pd.DataFrame(rows)

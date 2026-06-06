# -*- coding: utf-8 -*-
"""
sim_core.py — generative model + simulator for the calibrated synthetic cohort.

Pipeline (importable):
  simulate_cohort() -> (df_items, df_meta, item_dict, calib_report)

The generative model:
  1. Draw 7 correlated standardized disorder latents theta ~ MVN(0, Sigma_pd),
     where Sigma_pd is the nearest positive-definite matrix to LATENT_CORR.
  2. Apply an additive male sex effect to the alcohol latent.
  3. For each item, eta = sum(loading_d * theta_d) + residual noise, standardized
     to ~N(0,1), then discretized with graded thresholds (base + per-instrument
     difficulty delta). delta is tuned by bisection to hit the target caseness
     prevalence for that instrument.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import config as C


def nearest_pd(corr):
    """Nearest positive-definite correlation matrix via eigenvalue clipping."""
    A = np.array(corr, float)
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 1e-4, None)
    B = V @ np.diag(w) @ V.T
    # renormalize to unit diagonal (correlation)
    d = np.sqrt(np.diag(B))
    B = B / np.outer(d, d)
    return B


def _item_eta(theta, sex_c, domain, own_load, cross, rng):
    """Continuous latent response for one item, standardized to ~N(0,1)."""
    di = C.DOMAINS.index(domain)
    eta = own_load * theta[:, di]
    used = own_load ** 2
    for (cd, cl) in cross:
        eta = eta + cl * theta[:, C.DOMAINS.index(cd)]
        used += cl ** 2
    resid_sd = np.sqrt(max(1e-3, 1.0 - used))
    eta = eta + resid_sd * rng.standard_normal(theta.shape[0])
    # standardize empirically (correlated latents change variance slightly)
    eta = (eta - eta.mean()) / eta.std()
    return eta


def _discretize(eta, n_cat, delta):
    """Graded-response discretization: score = #thresholds exceeded."""
    thr = np.array(C.base_thresholds(n_cat)) + delta
    score = np.zeros(len(eta), dtype=int)
    for t in thr:
        score += (eta > t).astype(int)
    return score


def _prevalence_for_delta(etas, n_cats, delta, cutoff):
    tot = np.zeros(len(etas[0]), dtype=int)
    for eta, nc in zip(etas, n_cats):
        tot = tot + _discretize(eta, nc, delta)
    return (tot >= cutoff).mean(), tot


def _tune_delta(etas, n_cats, cutoff, target, lo=-3.0, hi=3.5, iters=40):
    """Bisection on global difficulty delta to hit target caseness prevalence.
    Higher delta -> higher thresholds -> lower scores -> lower prevalence."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        prev, _ = _prevalence_for_delta(etas, n_cats, mid, cutoff)
        if prev > target:   # too many cases -> make items harder (raise delta)
            lo = mid
        else:
            hi = mid
    delta = (lo + hi) / 2
    prev, tot = _prevalence_for_delta(etas, n_cats, delta, cutoff)
    return delta, prev, tot


def simulate_cohort(seed=None, n=None, verbose=True):
    seed = C.SEED if seed is None else seed
    n = C.N_SUBJECTS if n is None else n
    rng = np.random.default_rng(seed)

    Sigma = nearest_pd(C.LATENT_CORR)
    theta = rng.multivariate_normal(np.zeros(len(C.DOMAINS)), Sigma, size=n)
    # sex (male=1) and alcohol sex effect
    sex_male = (rng.random(n) < C.SEX_MALE_PROB).astype(int)
    sex_c = sex_male - C.SEX_MALE_PROB
    alc_idx = C.DOMAINS.index("alcohol")
    theta[:, alc_idx] = theta[:, alc_idx] + C.ALCOHOL_SEX_BETA * sex_c
    # restandardize alcohol latent
    theta[:, alc_idx] = (theta[:, alc_idx] - theta[:, alc_idx].mean()) / theta[:, alc_idx].std()

    # build all item etas, grouped by instrument/domain
    item_cols = {}
    item_dict_rows = []
    per_domain_eta = {d: [] for d in C.DOMAINS}
    per_domain_ncat = {d: [] for d in C.DOMAINS}
    per_domain_codes = {d: [] for d in C.DOMAINS}
    for d in C.DOMAINS:
        for (code, label, own, ncat, cross) in C.ITEMS[d]:
            eta = _item_eta(theta, sex_c, d, own, cross, rng)
            per_domain_eta[d].append(eta)
            per_domain_ncat[d].append(ncat)
            per_domain_codes[d].append(code)
            item_dict_rows.append({
                "item": code, "domain": d, "label": label,
                "n_categories": ncat, "own_loading": own,
                "cross_loadings": ";".join(f"{cd}:{cl}" for cd, cl in cross) if cross else "",
            })

    # tune difficulty per instrument to hit target prevalence, then materialize items
    calib_rows = []
    totals = {}
    labels = {}
    for d in C.DOMAINS:
        etas = per_domain_eta[d]
        ncats = per_domain_ncat[d]
        cutoff = C.CUTOFF[d]
        target = C.TARGET_PREVALENCE[d]
        delta, prev, tot = _tune_delta(etas, ncats, cutoff, target)
        totals[d] = tot
        labels[d] = (tot >= cutoff).astype(int)
        for code, eta, nc in zip(per_domain_codes[d], etas, ncats):
            item_cols[code] = _discretize(eta, nc, delta)
        calib_rows.append({
            "domain": d, "label": C.DOMAIN_LABEL[d], "cutoff": cutoff,
            "target_prev": target, "achieved_prev": round(float(prev), 4),
            "delta": round(float(delta), 3),
            "total_mean": round(float(tot.mean()), 2),
            "total_sd": round(float(tot.std()), 2),
            "n_positive": int(labels[d].sum()),
        })

    df_items = pd.DataFrame(item_cols)
    df_items.insert(0, "subject_id", np.arange(n))
    # meta: totals, labels, sex
    meta = {"subject_id": np.arange(n), "sex_male": sex_male}
    for d in C.DOMAINS:
        meta[f"total_{d}"] = totals[d]
        meta[f"label_{d}"] = labels[d]
    # secondary sleep label (subthreshold ISI>=8)
    meta["label_sleep_sub"] = (totals["sleep"] >= C.SLEEP_SUBTHRESHOLD_CUTOFF).astype(int)
    df_meta = pd.DataFrame(meta)

    item_dict = pd.DataFrame(item_dict_rows)
    calib = pd.DataFrame(calib_rows)

    # achieved total-score correlation matrix (for calibration check vs LATENT_CORR)
    tot_df = pd.DataFrame({d: totals[d] for d in C.DOMAINS})
    achieved_corr = tot_df.corr().round(3)

    if verbose:
        print(f"[sim] N={n}  seed={seed}  items={df_items.shape[1]-1}")
        print(calib.to_string(index=False))
        print("\n[sim] achieved total-score correlations (vs latent targets):")
        print(achieved_corr.to_string())

    return df_items, df_meta, item_dict, calib, achieved_corr


if __name__ == "__main__":
    simulate_cohort()

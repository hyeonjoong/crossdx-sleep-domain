# -*- coding: utf-8 -*-
"""
01_simulate_cohort.py — generate and save the calibrated synthetic cohort.

Outputs (data/):
  simulated_cohort_items.csv   item-level responses (64 items) + subject_id
  simulated_cohort_meta.csv    total scores, binary caseness labels, sex
  item_dictionary.csv          item -> domain/label/loadings map
Outputs (results/tables/):
  T1_calibration.csv           target vs achieved prevalence, totals
  T1b_achieved_correlations.csv total-score correlation matrix

The committed data/simulated_cohort_*.csv files are the canonical input of the
paper. The draw depends on the sign convention of the SVD in the local
BLAS/LAPACK build (numpy's multivariate_normal), so on some builds the same seed
gives a different cohort. This step therefore compares its draw with the
committed files first; if they differ it writes nothing and exits with status 3.
--force overwrites them anyway.

To run on REAL data instead: do not overwrite these files. Put the two CSVs in
data/private/ and pass --data-dir (see README, "Running on real data").
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import sim_core

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
TBL = os.path.join(ROOT, "results", "tables")
EXIT_DIFFERENT = 3


def blas_lapack():
    """Name of the BLAS and LAPACK libraries numpy was built with."""
    try:
        deps = np.show_config(mode="dicts").get("Build Dependencies", {})
        return "BLAS %s, LAPACK %s" % (deps.get("blas", {}).get("name", "unknown"),
                                       deps.get("lapack", {}).get("name", "unknown"))
    except Exception:
        return "BLAS/LAPACK unknown"


def difference(new_df, path):
    """None if the file at `path` holds the same table as new_df after a CSV
    round trip, else a short description of how it differs."""
    old = pd.read_csv(path)
    new = pd.read_csv(io.StringIO(new_df.to_csv(index=False)))
    if old.equals(new):
        return None
    if list(old.columns) != list(new.columns) or old.shape != new.shape:
        return "columns or shape differ (%d x %d committed, %d x %d regenerated)" % (
            old.shape + new.shape)
    n = int((old.values != new.values).sum())
    return "%d of %d cells differ" % (n, old.size)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Simulate the calibrated cohort (step 01).")
    ap.add_argument("--force", action="store_true",
                    help="overwrite data/simulated_cohort_*.csv even if the regenerated "
                         "cohort differs from the committed one")
    args = ap.parse_args(argv)

    df_items, df_meta, item_dict, calib, achieved_corr = sim_core.simulate_cohort()
    targets = [(df_items, "simulated_cohort_items.csv"), (df_meta, "simulated_cohort_meta.csv")]
    diffs = []
    for df, name in targets:
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            d = difference(df, path)
            if d:
                diffs.append("data/%s: %s" % (name, d))
    if diffs and not args.force:
        msg = ["", "[01] STOP: the regenerated cohort differs from the committed one, so "
                   "nothing was written."]
        msg += ["     " + d for d in diffs]
        msg += ["     numpy %s (%s), Python %s" % (np.__version__, blas_lapack(),
                                                   sys.version.split()[0]),
                "     The draw depends on the SVD sign convention of the local LAPACK build;",
                "     the committed files are the canonical input (README, section Reproduce).",
                "     To compare every other output with the committed results without",
                "     changing this checkout:",
                "         python analysis/run_all.py --check --data-dir data",
                "     To overwrite the committed cohort anyway:",
                "         python analysis/01_simulate_cohort.py --force"]
        sys.stdout.flush()
        print("\n".join(msg), file=sys.stderr)
        sys.exit(EXIT_DIFFERENT)

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(TBL, exist_ok=True)
    df_items.to_csv(os.path.join(DATA, "simulated_cohort_items.csv"), index=False)
    df_meta.to_csv(os.path.join(DATA, "simulated_cohort_meta.csv"), index=False)
    item_dict.to_csv(os.path.join(DATA, "item_dictionary.csv"), index=False)
    calib.to_csv(os.path.join(TBL, "T1_calibration.csv"), index=False)
    achieved_corr.to_csv(os.path.join(TBL, "T1b_achieved_correlations.csv"))
    if diffs:
        print("\n[01] --force: replaced the committed cohort (%s)" % "; ".join(diffs))
    print("\n[01] saved cohort to data/ and calibration tables to results/tables/")


if __name__ == "__main__":
    main()

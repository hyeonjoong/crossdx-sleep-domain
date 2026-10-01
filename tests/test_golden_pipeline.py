# -*- coding: utf-8 -*-
"""Golden test: steps 02, 03, 04 and 09 on the committed cohort reproduce the
committed tables (work order C10).

Text columns and the selected items must match exactly. Numeric columns may
differ by one unit in the third decimal (the tables are rounded to 3 dp), which
allows for other BLAS/LAPACK builds. With CROSSDX_STRICT=1 every file these
steps write must also be byte-identical (true on the reference build, macOS
with Accelerate).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, run_py, sha256

pytestmark = pytest.mark.slow

ATOL = 1e-3 + 1e-9
TABLES = ["T2_item_utility.csv", "T3_selected_panel.csv", "T4_lockbox_performance.csv",
          "T5_value_of_sleep.csv", "T6_bootstrap_stability.csv",
          "T7_lambda_sensitivity.csv", "T8_ggm_partial_corr.csv",
          "T9_bridge_centrality.csv", "T10_crossdx_contribution.csv",
          "T14_deconfounding.csv"]
FIGURES = ["F1_calibration.png", "F2_item_utility.png", "F3_lockbox_performance.png",
           "F4_ggm_network.png", "F5_contribution_heatmap.png", "F6_value_of_sleep.png",
           "F7_bootstrap_stability.png"]


def _compare_csv(ref_path, new_path):
    a, b = pd.read_csv(ref_path), pd.read_csv(new_path)
    name = os.path.basename(ref_path)
    assert list(a.columns) == list(b.columns), name
    assert a.shape == b.shape, name
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            ok = np.isclose(x.astype(float), y.astype(float), rtol=0, atol=ATOL,
                            equal_nan=True)
            assert ok.all(), "%s column %s: %s vs %s" % (
                name, col, x[~ok].tolist()[:5], y[~ok].tolist()[:5])
        else:
            assert x.astype(str).tolist() == y.astype(str).tolist(), "%s column %s" % (name, col)


def _compare_json(a, b, where="panel.json"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b), where
        for k in a:
            _compare_json(a[k], b[k], "%s/%s" % (where, k))
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert abs(float(a) - float(b)) <= ATOL, where
    else:
        assert a == b, where


def test_steps_02_03_04_09_reproduce_committed_results(repo_copy):
    tdir = os.path.join(repo_copy, "results", "tables")
    # remove the outputs first so a step that silently skips cannot pass
    for t in TABLES:
        os.remove(os.path.join(tdir, t))
    os.remove(os.path.join(repo_copy, "results", "panel.json"))
    for f in FIGURES[1:]:                       # F1 is drawn from T1 (step 01)
        os.remove(os.path.join(repo_copy, "results", "figures", f))

    logs = {}
    for step in ["02_optimize.py", "03_network_contrib.py", "04_figures.py",
                 "09_deconfounding.py"]:
        r = run_py(repo_copy, step)
        assert r.returncode == 0, step + "\n" + r.stdout + r.stderr
        logs[step] = r.stdout

    for t in TABLES:
        _compare_csv(os.path.join(ROOT, "results", "tables", t), os.path.join(tdir, t))
    with open(os.path.join(ROOT, "results", "panel.json")) as fh:
        ref = json.load(fh)
    with open(os.path.join(repo_copy, "results", "panel.json")) as fh:
        new = json.load(fh)
    _compare_json(ref, new)

    # console messages name the items actually selected (work order C12)
    sleep, dep6, dep7 = (new["panel_7domain"]["sleep"], new["panel_6domain"]["dep"],
                         new["panel_7domain"]["dep"])
    assert "SLEEP anchor (%s)" % sleep in logs["03_network_contrib.py"]
    assert "6-domain = %s -> 7-domain = %s" % (dep6, dep7) in logs["09_deconfounding.py"]

    if os.environ.get("CROSSDX_STRICT") == "1":
        written = [os.path.join("results", "tables", t) for t in TABLES] + \
                  [os.path.join("results", "panel.json")] + \
                  [os.path.join("results", "figures", f) for f in FIGURES]
        differ = [p for p in written
                  if sha256(os.path.join(ROOT, p)) != sha256(os.path.join(repo_copy, p))]
        assert differ == []


def test_09_names_no_shift_when_anchors_agree(repo_copy):
    pj = os.path.join(repo_copy, "results", "panel.json")
    with open(pj) as fh:
        panel = json.load(fh)
    panel["panel_6domain"]["dep"] = panel["panel_7domain"]["dep"]
    with open(pj, "w") as fh:
        json.dump(panel, fh, indent=2)
    r = run_py(repo_copy, "09_deconfounding.py")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "fatigue -> depressed mood" not in r.stdout

# -*- coding: utf-8 -*-
"""Steps 01 and 05 regenerate the committed cohort and ablation exactly.

These depend on the sign convention of the SVD in the local LAPACK build
(numpy's multivariate_normal). They are expected to fail on builds that differ
from the one that produced the committed files, until the orientation is pinned
in sim_core (work order C1b, awaiting the owner's decision).
"""
import os

import pytest

from conftest import ROOT, run_py, sha256

pytestmark = [pytest.mark.slow,
              pytest.mark.xfail(strict=False, reason="C1b: platform SVD sign")]


def test_step01_regenerates_committed_cohort(repo_copy):
    r = run_py(repo_copy, "01_simulate_cohort.py", "--force")
    assert r.returncode == 0, r.stdout + r.stderr
    for name in ("simulated_cohort_items.csv", "simulated_cohort_meta.csv"):
        assert sha256(os.path.join(repo_copy, "data", name)) == \
            sha256(os.path.join(ROOT, "data", name)), name


def test_step05_regenerates_committed_ablation(repo_copy):
    r = run_py(repo_copy, "05_robustness_baselines.py")
    assert r.returncode == 0, r.stdout + r.stderr
    for rel in (("results", "tables", "T13_ablation.csv"),
                ("results", "robustness_summary.json")):
        assert sha256(os.path.join(repo_copy, *rel)) == sha256(os.path.join(ROOT, *rel)), rel

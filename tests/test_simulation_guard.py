# -*- coding: utf-8 -*-
"""Step 01 must not silently replace the committed cohort (work order C1)."""
import os

import pytest

from conftest import run_py, sha256

pytestmark = pytest.mark.slow


def _alter_one_cell(path):
    with open(path, encoding="utf-8", newline="") as fh:
        lines = fh.read().split("\n")
    cells = lines[1].split(",")
    cells[1] = "3" if cells[1] != "3" else "2"
    lines[1] = ",".join(cells)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines))


def _outputs(root):
    return [os.path.join(root, "data", "simulated_cohort_items.csv"),
            os.path.join(root, "data", "simulated_cohort_meta.csv"),
            os.path.join(root, "data", "item_dictionary.csv"),
            os.path.join(root, "results", "tables", "T1_calibration.csv"),
            os.path.join(root, "results", "tables", "T1b_achieved_correlations.csv")]


def test_refuses_to_overwrite_a_different_cohort(repo_copy):
    items = os.path.join(repo_copy, "data", "simulated_cohort_items.csv")
    _alter_one_cell(items)
    before = {p: sha256(p) for p in _outputs(repo_copy)}
    r = run_py(repo_copy, "01_simulate_cohort.py")
    assert r.returncode == 3, r.stdout + r.stderr
    assert {p: sha256(p) for p in _outputs(repo_copy)} == before
    out = r.stdout + r.stderr
    assert "simulated_cohort_items.csv" in out
    assert "--force" in out
    assert "Traceback" not in out


def test_force_overwrites(repo_copy):
    items = os.path.join(repo_copy, "data", "simulated_cohort_items.csv")
    _alter_one_cell(items)
    altered = sha256(items)
    r = run_py(repo_copy, "01_simulate_cohort.py", "--force")
    assert r.returncode == 0, r.stdout + r.stderr
    assert sha256(items) != altered


def test_first_run_writes_the_cohort(repo_copy):
    for name in ("simulated_cohort_items.csv", "simulated_cohort_meta.csv"):
        os.remove(os.path.join(repo_copy, "data", name))
    r = run_py(repo_copy, "01_simulate_cohort.py")
    assert r.returncode == 0, r.stdout + r.stderr
    for p in _outputs(repo_copy):
        assert os.path.getsize(p) > 0

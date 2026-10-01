# -*- coding: utf-8 -*-
"""10_figures_brm.py: value checks and the lettering-size gate (work order C8, C10)."""
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from conftest import ROOT, run_py  # noqa: E402

N_CHECKS = 57


def _module():
    spec = importlib.util.spec_from_file_location(
        "figures_brm", os.path.join(ROOT, "analysis", "10_figures_brm.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cap_measurement_sees_white_text():
    from matplotlib.patches import Rectangle
    m = _module()
    fig, ax = plt.subplots(figsize=(m.WIDTH_IN, 2.0))
    ax.add_patch(Rectangle((0, 0), 1, 1, color=m.TEAL_E, transform=ax.transAxes))
    texts = [ax.text(0.08 + 0.14 * i, 0.5, "DEP\nISI5", color="white", fontsize=9,
                     fontweight="bold", ha="center", va="center", transform=ax.transAxes)
             for i in range(7)]
    ax.axis("off")
    try:
        r = m.measure_cap_mm(fig)
        assert r["modal_cap_mm"] > 2.0, r
        assert all(t.get_color() == "white" for t in texts)     # colours restored
    finally:
        plt.close(fig)


def _metrics(outdir):
    with open(os.path.join(outdir, "figure_metrics.json")) as fh:
        return json.load(fh)


@pytest.mark.slow
def test_committed_tables_pass_all_checks(repo_copy, tmp_path):
    out = tmp_path / "figs"
    # no --root: it defaults to the folder above analysis/
    r = run_py(repo_copy, "10_figures_brm.py", "--outdir", out, "--strict")
    assert r.returncode == 0, r.stdout + r.stderr
    m = _metrics(out)
    assert len(m["metrics"]) == 10
    assert all(x["passes"] for x in m["metrics"]), [
        (x["figure"], x["modal_cap_mm"]) for x in m["metrics"]]
    assert m["failures"] == [] and m["value_mismatches"] == []
    assert len(m["checks"]) == N_CHECKS
    assert all(c["ok"] for c in m["checks"])
    assert len([f for f in os.listdir(out) if f.endswith(".tif")]) == 10


@pytest.mark.slow
def test_value_mismatch_is_reported_and_fails(repo_copy, tmp_path):
    t2 = os.path.join(repo_copy, "results", "tables", "T2_item_utility.csv")
    u = pd.read_csv(t2)
    u.loc[u.item == "ISI5", "utility"] = 0.700
    u.to_csv(t2, index=False)
    out = tmp_path / "figs"
    r = run_py(repo_copy, "10_figures_brm.py", "--root", repo_copy, "--outdir", out)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "Traceback" not in r.stderr
    m = _metrics(out)
    bad = [c["quantity"] for c in m["value_mismatches"]]
    assert bad == ["utility ISI5"]
    assert len(m["checks"]) == N_CHECKS          # every figure was still checked


@pytest.mark.slow
def test_strict_fails_below_the_floor(repo_copy, tmp_path):
    m = _module()
    m.CAP_FLOOR_MM = 50.0
    out = str(tmp_path / "figs")
    with pytest.raises(SystemExit) as e:
        m.main(["--root", repo_copy, "--outdir", out, "--strict"])
    assert e.value.code == 1
    assert len(_metrics(out)["failures"]) == 10

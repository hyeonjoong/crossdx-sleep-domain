# -*- coding: utf-8 -*-
"""README and doc builders match the repository (work order C7)."""
import importlib.util
import os
import re

import pytest

from conftest import ROOT, run_py

README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()


def test_every_script_named_in_readme_exists():
    named = set(re.findall(r"\b(?:analysis/)?([0-9A-Za-z_]+\.py)\b", README))
    assert named, "README names no scripts"
    missing = [n for n in sorted(named)
               if not os.path.exists(os.path.join(ROOT, "analysis", n))
               and not os.path.exists(os.path.join(ROOT, "tests", n))]
    assert missing == []


def test_every_analysis_script_is_in_readme():
    scripts = [f for f in os.listdir(os.path.join(ROOT, "analysis"))
               if f.endswith(".py") and not f.startswith("_")]
    assert [s for s in sorted(scripts) if s not in README] == []


def test_readme_has_no_stale_claims():
    assert "BELL_Paper3" not in README
    assert "5-fold" not in README
    assert "PLOS ONE" not in README


def test_readme_does_not_point_real_data_at_tracked_files():
    section = README.split("## Running on real data", 1)[1].split("\n## ", 1)[0]
    assert "data/private/" in section
    assert "data/simulated_cohort_items.csv" not in section


def _has_docx():
    return importlib.util.find_spec("docx") is not None


@pytest.mark.skipif(not _has_docx(), reason="python-docx not installed")
def test_importing_build_submission_docs_creates_nothing(repo_copy):
    code = ("import sys; sys.path.insert(0, 'analysis'); import build_submission_docs")
    import subprocess
    import sys
    from conftest import step_env
    r = subprocess.run([sys.executable, "-c", code], cwd=repo_copy, env=step_env(),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not os.path.exists(os.path.join(repo_copy, "SUBMISSION"))


@pytest.mark.skipif(not _has_docx(), reason="python-docx not installed")
@pytest.mark.parametrize("script", ["build_docx.py", "build_submission_docs.py"])
def test_builders_explain_missing_manuscript(repo_copy, script):
    r = run_py(repo_copy, script)
    assert r.returncode != 0
    assert "Traceback" not in r.stderr
    assert "manuscript" in (r.stdout + r.stderr)
    assert not os.path.exists(os.path.join(repo_copy, "SUBMISSION"))


def _git_repo():
    import shutil
    return shutil.which("git") is not None and os.path.isdir(os.path.join(ROOT, ".git"))


@pytest.mark.skipif(not _git_repo(), reason="needs git and a git checkout")
@pytest.mark.parametrize("path", ["data/private/simulated_cohort_items.csv",
                                  "out/brm_figures/Figure1.tif", "SUBMISSION/S1_Protocol.docx",
                                  "data/realdata/isi.csv.part"])
def test_private_and_generated_paths_are_git_ignored(path):
    import subprocess
    r = subprocess.run(["git", "-C", ROOT, "check-ignore", "-q", path],
                       env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    assert r.returncode == 0, path

# -*- coding: utf-8 -*-
"""Shared test helpers.

Every test that runs a pipeline step works on a copy of the repository in a
pytest temporary folder, so the checkout itself is never written to.
"""
import hashlib
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = os.path.join(ROOT, "analysis")
if ANALYSIS not in sys.path:
    sys.path.insert(0, ANALYSIS)

# folders that are never copied (generated output, caches, private data)
_SKIP_ANYWHERE = {".git", "out", "SUBMISSION", "__pycache__", ".pytest_cache",
                  ".venv", "venv"}


def _ignore(dirpath, names):
    rel = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
    skip = {n for n in names if n in _SKIP_ANYWHERE or n.endswith((".pyc", ".part"))}
    if rel == "data":
        skip.add("private")
    if rel == "data/realdata":
        # downloaded open data and internal cohorts are not needed by the tests
        skip |= {n for n in names if n != "SOURCE.md"}
    return skip


def copy_repo(dest):
    shutil.copytree(ROOT, dest, ignore=_ignore)
    return str(dest)


@pytest.fixture
def repo_copy(tmp_path):
    """A fresh copy of the repository (code, committed data and results)."""
    return copy_repo(tmp_path / "repo")


def step_env():
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env.pop("CROSSDX_DATA_DIR", None)
    return env


def run_py(root, script, *args, timeout=900, env=None):
    """Run analysis/<script> of the copy at `root` in a fresh interpreter."""
    cmd = [sys.executable, os.path.join(root, "analysis", script)] + [str(a) for a in args]
    return subprocess.run(cmd, cwd=root, env=env or step_env(), capture_output=True,
                          text=True, timeout=timeout)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def tree_hashes(root, subdirs=("data", "results")):
    """SHA-256 of every file under the given subfolders (relative path: hash)."""
    out = {}
    for sub in subdirs:
        for dp, _dn, fn in os.walk(os.path.join(root, sub)):
            for f in fn:
                p = os.path.join(dp, f)
                out[os.path.relpath(p, root)] = sha256(p)
    return out

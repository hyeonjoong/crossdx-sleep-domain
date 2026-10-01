# -*- coding: utf-8 -*-
"""Citation metadata is complete and consistent (work order C9)."""
import json
import os

import pytest

from conftest import ROOT

REPO_URL = "https://github.com/hyeonjoong/crossdx-sleep-domain"


def _cff():
    yaml = pytest.importorskip("yaml")
    with open(os.path.join(ROOT, "CITATION.cff"), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _zenodo():
    with open(os.path.join(ROOT, ".zenodo.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_citation_cff_fields():
    c = _cff()
    assert c["cff-version"] == "1.2.0"
    for key in ("title", "version", "date-released", "repository-code", "authors"):
        assert c.get(key), key
    assert c["repository-code"] == REPO_URL
    assert str(c["version"]).count(".") == 2


def test_zenodo_json_parses_and_agrees_with_citation():
    z = _zenodo()
    assert z["upload_type"] == "software"
    if "version" in z:                       # Zenodo uses the release tag when absent
        assert z["version"] == str(_cff()["version"])
    c_names = ["%s, %s" % (a["family-names"], a["given-names"]) for a in _cff()["authors"]]
    assert [a["name"] for a in z["creators"]] == c_names

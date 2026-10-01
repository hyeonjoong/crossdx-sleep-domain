# -*- coding: utf-8 -*-
"""The committed simulated cohort is the canonical input of the paper."""
import os

import pandas as pd

import config as C
from conftest import ROOT, sha256

EXPECTED_SHA256 = {
    "simulated_cohort_items.csv": "dcfab0bff11eced9847838db051911cbde09a28340efbf970690bdc75e12c742",
    "simulated_cohort_meta.csv": "05e29ab06bbd539d98c6188837ba6a09d858fd30d602ff701ae4b7b81aca7da3",
    "item_dictionary.csv": "9c5733d20e2a84bf0a324447d368e9951ee0844766718ce0cb163aeb47e93858",
}


def test_committed_data_hashes():
    for name, digest in EXPECTED_SHA256.items():
        assert sha256(os.path.join(ROOT, "data", name)) == digest, name


def test_item_dictionary_matches_config():
    d = pd.read_csv(os.path.join(ROOT, "data", "item_dictionary.csv"))
    expected = [(code, dom, label, ncat, own)
                for dom in C.DOMAINS
                for (code, label, own, ncat, _cross) in C.ITEMS[dom]]
    got = list(zip(d["item"], d["domain"], d["label"], d["n_categories"], d["own_loading"]))
    assert got == expected


def test_cohort_columns_match_config():
    items = pd.read_csv(os.path.join(ROOT, "data", "simulated_cohort_items.csv"), nrows=5)
    codes = [it[0] for dom in C.DOMAINS for it in C.ITEMS[dom]]
    assert list(items.columns) == ["subject_id"] + codes
    meta = pd.read_csv(os.path.join(ROOT, "data", "simulated_cohort_meta.csv"), nrows=5)
    for dom in C.DOMAINS:
        assert "label_%s" % dom in meta.columns
    assert "label_sleep_sub" in meta.columns

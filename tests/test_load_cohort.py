# -*- coding: utf-8 -*-
"""Input checks in pipeline_core.load_cohort (work order C4) and the data
folder override (work order C5)."""
import os

import numpy as np
import pandas as pd
import pytest

import pipeline_core as P
from conftest import ROOT, run_py

ITEMS_CSV = os.path.join(ROOT, "data", "simulated_cohort_items.csv")
META_CSV = os.path.join(ROOT, "data", "simulated_cohort_meta.csv")


@pytest.fixture(scope="module")
def committed():
    return pd.read_csv(ITEMS_CSV), pd.read_csv(META_CSV)


def _write(d, items, meta, encoding="utf-8", lineterminator="\n"):
    os.makedirs(d, exist_ok=True)
    items.to_csv(os.path.join(d, "simulated_cohort_items.csv"), index=False,
                 encoding=encoding, lineterminator=lineterminator)
    meta.to_csv(os.path.join(d, "simulated_cohort_meta.csv"), index=False,
                encoding=encoding, lineterminator=lineterminator)
    return str(d)


def _load(monkeypatch, d):
    """Call load_cohort() with no arguments on folder d."""
    monkeypatch.delenv("CROSSDX_DATA_DIR", raising=False)
    monkeypatch.setattr(P, "DATA", str(d))
    return P.load_cohort()


def _input_error(monkeypatch, d):
    with pytest.raises(P.InputError) as e:
        _load(monkeypatch, d)
    msg = str(e.value)
    assert "\n" not in msg
    return msg


def test_committed_files_load_unchanged(committed, monkeypatch):
    items, meta = _load(monkeypatch, os.path.join(ROOT, "data"))
    assert items.equals(committed[0])
    assert meta.equals(committed[1])


def test_bom_and_crlf(committed, monkeypatch, tmp_path):
    d = _write(tmp_path, *committed, encoding="utf-8-sig", lineterminator="\r\n")
    items, meta = _load(monkeypatch, d)
    assert items.equals(committed[0])
    assert meta.equals(committed[1])


def test_meta_in_a_different_row_order_is_realigned(committed, monkeypatch, tmp_path):
    items0, meta0 = committed
    shuffled = meta0.assign(_k=meta0.subject_id.astype(str)).sort_values("_k") \
        .drop(columns="_k").reset_index(drop=True)
    assert not shuffled.subject_id.equals(meta0.subject_id)
    d = _write(tmp_path, items0, shuffled)
    items, meta = _load(monkeypatch, d)
    assert list(meta.subject_id) == list(items.subject_id)
    assert meta.equals(meta0)


def test_labels_coded_1_and_2_are_rejected(committed, monkeypatch, tmp_path):
    items0, meta0 = committed
    meta = meta0.copy()
    for c in [c for c in meta.columns if c.startswith("label_")]:
        meta[c] = np.where(meta[c] == 1, 1, 2)
    msg = _input_error(monkeypatch, _write(tmp_path, items0, meta))
    assert "label_dep" in msg
    assert "0 or 1" in msg


def test_blank_item_cells_are_rejected(committed, monkeypatch, tmp_path):
    items = committed[0].copy()
    items["ISI2"] = items["ISI2"].astype(float)
    items.loc[[3, 10, 20], "ISI2"] = np.nan
    msg = _input_error(monkeypatch, _write(tmp_path, items, committed[1]))
    assert "ISI2" in msg
    assert "5, 12, 22" in msg          # spreadsheet rows, header = row 1


def test_renamed_item_column_is_rejected(committed, monkeypatch, tmp_path):
    items = committed[0].rename(columns={"ISI2": "ISI_2"})
    msg = _input_error(monkeypatch, _write(tmp_path, items, committed[1]))
    assert "ISI_2" in msg and "ISI2" in msg


def test_out_of_range_item_value_is_rejected(committed, monkeypatch, tmp_path):
    items = committed[0].copy()
    items.loc[0, "PHQ1"] = 4            # PHQ-9 items are scored 0..3
    msg = _input_error(monkeypatch, _write(tmp_path, items, committed[1]))
    assert "PHQ1" in msg and "0..3" in msg


def test_meta_one_row_short_is_rejected(committed, monkeypatch, tmp_path):
    meta = committed[1].iloc[:-1]
    msg = _input_error(monkeypatch, _write(tmp_path, committed[0], meta))
    assert "subject_id" in msg


def test_duplicate_subject_id_is_rejected(committed, monkeypatch, tmp_path):
    items = committed[0].copy()
    items.loc[5, "subject_id"] = items.loc[4, "subject_id"]
    msg = _input_error(monkeypatch, _write(tmp_path, items, committed[1]))
    assert "subject_id" in msg


def test_label_with_one_class_is_rejected(committed, monkeypatch, tmp_path):
    meta = committed[1].copy()
    meta["label_panic"] = 0
    msg = _input_error(monkeypatch, _write(tmp_path, committed[0], meta))
    assert "label_panic" in msg


def _copy_with_extra_lines(d, items_tail=b"", meta_tail=b"", meta_insert=None):
    """Byte copies of the committed CSVs with lines added at the end (or, for
    meta_insert=(line number, text), inside the label file)."""
    os.makedirs(d, exist_ok=True)
    for src, tail, name in ((ITEMS_CSV, items_tail, "simulated_cohort_items.csv"),
                            (META_CSV, meta_tail, "simulated_cohort_meta.csv")):
        with open(src, "rb") as fh:
            data = fh.read()
        if name.endswith("meta.csv") and meta_insert:
            lines = data.split(b"\n")
            lines.insert(meta_insert[0], meta_insert[1])
            data = b"\n".join(lines)
        with open(os.path.join(d, name), "wb") as fh:
            fh.write(data + tail)
    return str(d)


def _commas(path):
    with open(path, encoding="utf-8") as fh:
        return ("," * (len(fh.readline().split(",")) - 1) + "\n").encode()


@pytest.mark.parametrize("case", ["blank_line_items", "comma_row_items", "both",
                                  "blank_line_inside_meta"])
def test_empty_rows_in_one_or_both_files(committed, monkeypatch, tmp_path, capsys, case):
    # An empty row makes pandas read integer columns (subject_id too) as decimals;
    # the loader must still match 0 with 0.0 and return the committed tables.
    tails = {"blank_line_items": dict(items_tail=b"\n"),
             "comma_row_items": dict(items_tail=_commas(ITEMS_CSV)),
             "both": dict(items_tail=_commas(ITEMS_CSV), meta_tail=_commas(META_CSV)),
             "blank_line_inside_meta": dict(meta_insert=(10, b""))}[case]
    d = _copy_with_extra_lines(tmp_path, **tails)
    items, meta = _load(monkeypatch, d)
    assert items.equals(committed[0])
    assert meta.equals(committed[1])
    assert items.subject_id.dtype == "int64" and meta.subject_id.dtype == "int64"
    assert "empty row" in capsys.readouterr().out


def test_whole_number_ids_written_as_decimals_still_match(committed, monkeypatch, tmp_path):
    meta = committed[1].assign(subject_id=committed[1].subject_id.astype(float))
    d = _write(tmp_path, committed[0], meta)          # 0.0, 1.0, ... in the label file
    items, meta_l = _load(monkeypatch, d)
    assert list(meta_l.label_dep) == list(committed[1].label_dep)


@pytest.mark.parametrize("column", ["PHQ9총점", "GAD7점수", "PHQ-9", "PCL-5", "PHQ9_total",
                                    "ISI총점"])
def test_total_score_columns_are_ignored(committed, monkeypatch, tmp_path, capsys, column):
    # Korean exports often carry instrument totals next to the items. They must not
    # be taken for a renamed item (renaming PHQ-9 to PHQ9 would duplicate PHQ9).
    items = committed[0].copy()
    items[column] = items[["PHQ%d" % k for k in range(1, 10)]].sum(axis=1)
    items_l, meta_l = _load(monkeypatch, _write(tmp_path, items, committed[1]))
    assert items_l.equals(committed[0])
    out = capsys.readouterr().out
    assert column in out
    assert "rename" not in out


def test_korean_suffix_on_a_missing_item_gets_a_rename_hint(committed, monkeypatch, tmp_path):
    items = committed[0].rename(columns={"ISI2": "ISI2번"})
    msg = _input_error(monkeypatch, _write(tmp_path, items, committed[1]))
    assert "ISI2번" in msg and "rename it to ISI2" in msg


def test_total_column_without_the_item_reports_the_missing_item(committed, monkeypatch,
                                                                tmp_path):
    items = committed[0].rename(columns={"PHQ9": "PHQ9총점"})
    msg = _input_error(monkeypatch, _write(tmp_path, items, committed[1]))
    assert "missing" in msg and "PHQ9" in msg
    assert "rename" not in msg


def test_cp949_file_with_extra_korean_column(committed, monkeypatch, tmp_path, capsys):
    items0, meta0 = committed
    items = items0.copy()
    items["비고"] = "메모"
    d = _write(tmp_path, items, meta0, encoding="cp949")
    items_l, meta_l = _load(monkeypatch, d)
    assert items_l.equals(items0)                # extra column dropped
    assert meta_l.equals(meta0)
    assert "비고" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# data folder override (C5)
# ---------------------------------------------------------------------------
def _marked_copy(committed, d):
    """Copy of the committed cohort whose meta lacks sex_male, so it is recognisable."""
    return _write(d, committed[0], committed[1].drop(columns="sex_male"))


def test_data_dir_argument(committed, monkeypatch, tmp_path):
    monkeypatch.delenv("CROSSDX_DATA_DIR", raising=False)
    d = _marked_copy(committed, tmp_path / "x")
    _items, meta = P.load_cohort(data_dir=d)
    assert "sex_male" not in meta.columns


def test_data_dir_env_var(committed, monkeypatch, tmp_path):
    d = _marked_copy(committed, tmp_path / "x")
    monkeypatch.setenv("CROSSDX_DATA_DIR", d)
    _items, meta = P.load_cohort()
    assert "sex_male" not in meta.columns


def test_argument_wins_over_env_var(committed, monkeypatch, tmp_path):
    monkeypatch.setenv("CROSSDX_DATA_DIR", str(tmp_path / "does-not-exist"))
    _items, meta = P.load_cohort(data_dir=os.path.join(ROOT, "data"))
    assert meta.equals(committed[1])


def test_default_reads_committed_data(committed, monkeypatch):
    monkeypatch.delenv("CROSSDX_DATA_DIR", raising=False)
    items, meta = P.load_cohort()
    assert items.equals(committed[0]) and meta.equals(committed[1])


def test_missing_folder_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("CROSSDX_DATA_DIR", raising=False)
    with pytest.raises(P.InputError) as e:
        P.load_cohort(data_dir=str(tmp_path / "nope"))
    assert "simulated_cohort_items.csv" in str(e.value)


@pytest.mark.slow
def test_step_reports_bad_input_without_traceback(committed, repo_copy, tmp_path):
    meta = committed[1].copy()
    meta["label_dep"] = np.where(meta["label_dep"] == 1, 1, 2)
    bad = _write(tmp_path / "bad", committed[0], meta)
    r = run_py(repo_copy, "09_deconfounding.py", "--data-dir", bad)
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    assert "label_dep" in r.stdout + r.stderr


@pytest.mark.slow
def test_step_reads_data_dir(committed, repo_copy, tmp_path):
    good = _write(tmp_path / "good", *committed)
    r = run_py(repo_copy, "09_deconfounding.py", "--data-dir", good)
    assert r.returncode == 0, r.stdout + r.stderr

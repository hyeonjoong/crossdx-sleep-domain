# -*- coding: utf-8 -*-
"""
pipeline_core.py — re-implementation of the Lim et al. (2026) cross-diagnostic
item-selection framework, extended to a 7th (sleep) domain.

Core pieces (shared by 02_optimize.py, 03_network_contrib.py, 05 and 09):
  - load_cohort()           load item-level data + labels, with input checks
  - make_split()            strict 15% lockbox vs 85% development
  - univariate_utility()    U(j,k) = mean(AUROC, AUPRC) of item k for domain j
  - cosine_redundancy()     cosine similarity between standardized item vectors
  - optimize_panel()        joint objective: max Σ U − λ Σ cos, 1 item per domain
  - evaluate_panel()        lockbox AUROC/AUPRC for panel vs anchor-only

The objective reproduces:
    max  Σ_j U(j, k_j)  −  λ Σ_{j<j'} cos( x_{k_j}, x_{k_j'} )
    s.t. exactly one item k_j per domain j, chosen from domain j's own instrument
"""
import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

ITEMS_FILE = "simulated_cohort_items.csv"
META_FILE = "simulated_cohort_meta.csv"
DATA_DIR_ENV = "CROSSDX_DATA_DIR"
LABEL_COLUMNS = ["label_%s" % d for d in C.DOMAINS] + ["label_sleep_sub"]
_ITEM_LIKE = re.compile(r"^(PHQ|GAD|PCL|PDSS|DSI|AUDIT|ISI)\d+$")
# words that mark a total or scale score in a column name (English and Korean)
_TOTAL_WORDS = ("total", "sum", "score", "총점", "합계", "총합", "점수")


class InputError(ValueError):
    """The cohort files do not have the layout the pipeline needs."""


def resolve_data_dir(data_dir=None):
    """Folder with the two cohort CSVs: the argument, else $CROSSDX_DATA_DIR,
    else data/ of this repository (the committed simulated cohort)."""
    if data_dir:
        return os.path.abspath(data_dir)
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return os.path.abspath(env)
    return DATA


def load_cohort(data_dir=None):
    """Read and check the item file and the label file (see check_input)."""
    folder = resolve_data_dir(data_dir)
    items = _read_csv(os.path.join(folder, ITEMS_FILE))
    meta = _read_csv(os.path.join(folder, META_FILE))
    return check_input(items, meta, ITEMS_FILE, META_FILE)


def _read_csv(path):
    """UTF-8 (with or without BOM) first, then CP949 (Korean Excel)."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        raise InputError("%s not found in %s / %s 파일이 %s 에 없습니다"
                         % (name, os.path.dirname(path), name, os.path.dirname(path)))
    try:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", skip_blank_lines=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949", skip_blank_lines=False)
    except UnicodeDecodeError:
        raise InputError("%s: text encoding is neither UTF-8 nor CP949; save it as "
                         "CSV UTF-8 / %s: 인코딩을 읽을 수 없습니다. CSV UTF-8로 저장해 주세요"
                         % (name, name)) from None
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise InputError("%s: not a readable CSV table (%s) / %s: CSV 표로 읽을 수 없습니다"
                         % (name, str(e).strip().replace("\n", " "), name)) from None
    return df


def _rows(index):
    """Spreadsheet row numbers (header = row 1) of the first offending rows."""
    rows = [str(int(i) + 2) for i in list(index)[:5]]
    return ", ".join(rows) + (", ..." if len(index) > 5 else "")


def _fail(name, column, en, ko, index=None):
    if index is not None and len(index):
        r = _rows(index)
        en = "%s (rows %s; row 1 = header)" % (en, r)
        ko = "%s (행 %s, 머리글 = 1행)" % (ko, r)
    raise InputError("%s, column %s: %s / %s 열: %s" % (name, column, en, column, ko))


def _fmt(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return repr(v) if isinstance(v, str) else str(v)


def _check_values(df, name, column, allowed_max, what_en, what_ko):
    """Raise InputError for blank, non-numeric, non-integer or out-of-range cells.
    Returns the column as int64, or None when it already is an integer column."""
    col = df[column]
    num = pd.to_numeric(col, errors="coerce")
    blank = col.isna()
    if blank.any():
        _fail(name, column, "%d blank cell(s); every row needs a value" % blank.sum(),
              "빈 칸 %d개; 모든 행에 값이 필요합니다" % blank.sum(), col.index[blank])
    bad = num.isna()
    if bad.any():
        v = _fmt(col[bad].iloc[0])
        _fail(name, column, "non-numeric value %s" % v, "숫자가 아닌 값 %s" % v,
              col.index[bad])
    numf = num.astype(float)
    bad = (numf != np.floor(numf)) | (numf < 0) | (numf > allowed_max)
    if bad.any():
        v = _fmt(num[bad].iloc[0])
        _fail(name, column, "%s; found %s" % (what_en, v), "%s (값 %s)" % (what_ko, v),
              col.index[bad])
    if pd.api.types.is_integer_dtype(col):
        return None
    return numf.astype("int64")


def check_input(items, meta, iname=ITEMS_FILE, mname=META_FILE):
    """Check the two cohort tables and return them aligned by subject_id.

    - fully empty rows (blank lines, ',,,,' lines) are dropped with a note.
    - subject_id: present in both files, no blanks, unique, same set of IDs
      (17 and 17.0 count as the same ID). The label rows are matched to the
      item rows by subject_id (not by row position).
    - the 64 item codes of config.ITEMS: present, whole numbers in
      0..n_categories-1, no blanks. A column that looks like a renamed item
      code (ISI_2, isi2, PHQ10) is an error; other extra columns are dropped
      with a note. That includes total scores such as PHQ9_total, ISI총점, and
      PHQ-9 or PHQ9총점 when the PHQ9 item column is also present.
    - label_<domain> for every domain and label_sleep_sub: only 0 or 1, no
      blanks, at least one 1 and one 0.
    Problems raise InputError with a one-line message (English / Korean).
    Input that passes every check is returned as it is (same objects, no copy).
    iname and mname are the file names used in messages.
    """
    items = _strip_headers(items, iname)
    meta = _strip_headers(meta, mname)
    items = _drop_empty_rows(items, iname)
    meta = _drop_empty_rows(meta, mname)

    # ---- subject_id ----
    ki = _subject_keys(items, iname)
    km = _subject_keys(meta, mname)
    only_i = ki[~ki.isin(set(km))]
    only_m = km[~km.isin(set(ki))]
    if len(only_i) or len(only_m):
        raise InputError(
            "subject_id differs between the files: %d ID(s) only in %s (first: %s), "
            "%d only in %s (first: %s) / 두 파일의 subject_id가 일치하지 않습니다"
            % (len(only_i), iname, ", ".join(only_i.iloc[:5]) or "-",
               len(only_m), mname, ", ".join(only_m.iloc[:5]) or "-"))

    # ---- item columns ----
    ncat = {it[0]: it[3] for d in C.DOMAINS for it in C.ITEMS[d]}
    extra = [c for c in items.columns if c != "subject_id" and c not in ncat]
    copies = {}
    for c in extra:
        guess = _item_guess(c)
        if guess is None:
            continue
        if guess in ncat and guess in items.columns:
            # PHQ-9, PHQ9총점, GAD7점수 next to the real PHQ9/GAD7 column: a total
            # score or a copy, not a renamed item
            copies[c] = guess
            continue
        hint_en = (" (rename it to %s)" % guess) if guess in ncat else ""
        hint_ko = (" (%s 로 바꿔 주세요)" % guess) if guess in ncat else ""
        raise InputError(
            "%s: column %s is not one of the 64 item codes%s; the codes are listed "
            "in data/item_dictionary.csv / %s 열은 문항 코드가 아닙니다%s"
            % (iname, c, hint_en, c, hint_ko))
    missing = [c for c in ncat if c not in items.columns]
    if missing:
        raise InputError("%s: %d item column(s) missing: %s / 문항 열 %d개가 없습니다: %s"
                         % (iname, len(missing), ", ".join(missing[:8]), len(missing),
                            ", ".join(missing[:8])))
    if extra:
        print("[input] %s: ignoring column(s) that are not item codes: %s"
              % (iname, ", ".join(extra)))
        for c, code in copies.items():
            print("[input] %s: column %s looks like a total score or a copy of %s, not "
                  "item %s itself; it is not used / %s 열은 총점이나 %s 의 복사본으로 "
                  "보여 사용하지 않습니다" % (iname, c, code, code, c, code))
        items = items.drop(columns=extra)
    conv = {}
    for code, k in ncat.items():
        new = _check_values(items, iname, code, k - 1,
                            "item scores must be whole numbers 0..%d" % (k - 1),
                            "문항 점수는 0..%d 정수여야 합니다" % (k - 1))
        if new is not None:
            conv[code] = new
    if conv:
        items = items.assign(**conv)

    # ---- label columns ----
    missing = [c for c in LABEL_COLUMNS if c not in meta.columns]
    if missing:
        raise InputError("%s: label column(s) missing: %s / 라벨 열이 없습니다: %s"
                         % (mname, ", ".join(missing), ", ".join(missing)))
    conv = {}
    for c in LABEL_COLUMNS:
        new = _check_values(meta, mname, c, 1,
                            "labels must be 0 or 1 (recode yes/no or 1/2 as 1/0)",
                            "라벨은 0 또는 1이어야 합니다 (예/아니오, 1/2 코딩은 1/0으로)")
        vals = meta[c] if new is None else new
        if new is not None:
            conv[c] = new
        pos = int((vals == 1).sum())
        if pos == 0 or pos == len(vals):
            v = 1 if pos else 0
            raise InputError(
                "%s, column %s: all %d rows are %d; each label needs at least one 1 and "
                "one 0 / %s 열: 모든 행이 %d입니다 (1과 0이 모두 있어야 합니다)"
                % (mname, c, len(vals), v, c, v))
    if conv:
        meta = meta.assign(**conv)

    # ---- match label rows to item rows by subject_id ----
    if not np.array_equal(ki.values, km.values):
        where = pd.Series(np.arange(len(km)), index=km.values)
        meta = meta.iloc[where.loc[ki.values].values]
        print("[input] %s: rows matched to %s by subject_id (the row order differed)"
              % (mname, iname))
    return _plain_index(items), _plain_index(meta)


def _strip_headers(df, name):
    stripped = [str(c).strip() for c in df.columns]
    if stripped == list(df.columns):
        return df
    if len(set(stripped)) != len(stripped):
        raise InputError("%s: duplicate column names after removing spaces / "
                         "%s: 중복된 열 이름이 있습니다" % (name, name))
    return df.set_axis(stripped, axis=1)


def _item_guess(column):
    """The item code an extra column name seems to stand for (ISI_2, isi2 and
    ISI2번 give ISI2), or None. A name with a total-score word (PHQ9_total,
    PHQ9총점, GAD7점수) never stands for an item."""
    name = str(column)
    if any(w in name.lower() for w in _TOTAL_WORDS):
        return None
    guess = re.sub(r"[^0-9A-Za-z]", "", name).upper()
    return guess if _ITEM_LIKE.match(guess) else None


def _is_whole(values):
    """True when a float array holds only finite whole numbers that fit int64 exactly."""
    v = np.asarray(values, dtype=float)
    return bool(len(v)) and bool(np.isfinite(v).all()) and bool((v == np.floor(v)).all()) \
        and bool((np.abs(v) < 2.0 ** 53).all())


def _drop_empty_rows(df, name):
    """Rows with no value at all (a blank line, or a ',,,,' line left by a
    spreadsheet). pandas reads every column of such a file as decimals, so after
    the rows are removed the columns holding only whole numbers are made integer
    again: the result is then the same as reading the file without those rows."""
    empty = df.isna().all(axis=1)
    if not empty.any():
        return df
    print("[input] %s: ignoring %d empty row(s)" % (name, int(empty.sum())))
    df = df.loc[~empty]
    whole = [c for c in df.columns
             if not isinstance(df[c], pd.DataFrame)
             and pd.api.types.is_float_dtype(df[c]) and _is_whole(df[c])]
    if whole:
        df = df.astype({c: "int64" for c in whole})
    return df


def _subject_keys(df, name):
    if "subject_id" not in df.columns:
        raise InputError("%s: no subject_id column; both files need one to match items "
                         "to labels / %s: subject_id 열이 없습니다" % (name, name))
    sid = df["subject_id"]
    if sid.isna().any():
        _fail(name, "subject_id", "blank subject_id", "빈 subject_id", sid.index[sid.isna()])
    if pd.api.types.is_float_dtype(sid) and _is_whole(sid):
        # 17.0 in one file and 17 in the other are the same participant
        sid = sid.astype("int64")
    k = sid.astype(str).str.strip()
    dup = k.duplicated(keep=False)
    if dup.any():
        _fail(name, "subject_id", "duplicate subject_id %r" % k[dup].iloc[0],
              "중복된 subject_id %r" % k[dup].iloc[0], k.index[dup])
    return k


def _plain_index(df):
    idx = df.index
    if isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1:
        return df
    return df.reset_index(drop=True)


def step_main(main, doc=None, argv=None):
    """Command line shared by the steps that read the cohort (02, 03, 05, 09):
    --data-dir, and input problems reported in one line with exit status 2."""
    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="folder with %s and %s (default: $%s, else data/)"
                         % (ITEMS_FILE, META_FILE, DATA_DIR_ENV))
    args = ap.parse_args(argv)
    try:
        main(data_dir=args.data_dir)
    except InputError as e:
        sys.stdout.flush()
        print("[input error] %s" % e, file=sys.stderr)
        sys.exit(2)


def domain_pool(domain):
    """Item codes belonging to a domain's own instrument."""
    return [it[0] for it in C.ITEMS[domain]]


def make_split(n, lockbox_frac=None, seed=None):
    lockbox_frac = C.LOCKBOX_FRAC if lockbox_frac is None else lockbox_frac
    seed = C.SEED if seed is None else seed
    rng = np.random.default_rng(seed + 7)
    idx = rng.permutation(n)
    n_lb = int(round(n * lockbox_frac))
    lockbox = np.sort(idx[:n_lb])
    dev = np.sort(idx[n_lb:])
    return dev, lockbox


def univariate_utility(items, meta, idx, domains=None):
    """U[domain][item_code] = mean(AUROC, AUPRC) using the single item as a
    ranking score for that domain's binary label. Rank-based => no model fit,
    no overfitting; equivalent to CV mean for a univariate predictor."""
    domains = C.DOMAINS if domains is None else domains
    U = {}
    for d in domains:
        y = meta.loc[idx, f"label_{d}"].values
        U[d] = {}
        if y.sum() == 0 or y.sum() == len(y):
            for code in domain_pool(d):
                U[d][code] = 0.0
            continue
        for code in domain_pool(d):
            x = items.loc[idx, code].values.astype(float)
            auroc = roc_auc_score(y, x)
            auprc = average_precision_score(y, x)
            U[d][code] = float(0.5 * (auroc + auprc))
    return U


def cosine_redundancy(items, idx, codes):
    """Cosine similarity matrix between standardized item response vectors."""
    X = items.loc[idx, codes].values.astype(float)
    X = StandardScaler().fit_transform(X)
    norms = np.linalg.norm(X, axis=0)
    norms[norms == 0] = 1e-9
    S = (X.T @ X) / np.outer(norms, norms)
    return pd.DataFrame(S, index=codes, columns=codes)


def optimize_panel(U, cos_df, lam=None, domains=None, n_restarts=60, seed=0):
    """Coordinate-ascent with random restarts for:
        max Σ_j U[j][k_j] − λ Σ_{j<j'} cos[k_j, k_{j'}]
    Returns dict domain->item_code and the objective value."""
    lam = C.REDUNDANCY_LAMBDA if lam is None else lam
    domains = C.DOMAINS if domains is None else domains
    rng = np.random.default_rng(seed)
    pools = {d: domain_pool(d) for d in domains}

    def objective(sel):
        val = sum(U[d][sel[d]] for d in domains)
        pen = 0.0
        for i, a in enumerate(domains):
            for b in domains[i + 1:]:
                pen += cos_df.loc[sel[a], sel[b]]
        return val - lam * pen

    best_sel, best_val = None, -np.inf
    for r in range(n_restarts):
        # random init
        sel = {d: pools[d][rng.integers(len(pools[d]))] for d in domains}
        improved = True
        while improved:
            improved = False
            for d in domains:
                cur = sel[d]
                best_k, best_local = cur, -np.inf
                others = [sel[o] for o in domains if o != d]
                for k in pools[d]:
                    score = U[d][k] - lam * sum(cos_df.loc[k, o] for o in others)
                    if score > best_local:
                        best_local, best_k = score, k
                if best_k != cur:
                    sel[d] = best_k
                    improved = True
        val = objective(sel)
        if val > best_val:
            best_val, best_sel = val, dict(sel)
    return best_sel, float(best_val)


def _fit_eval(items, meta, dev, lb, feat_codes, domain):
    """Train logistic on dev with feat_codes, evaluate on lockbox for domain label."""
    Xtr = items.loc[dev, feat_codes].values.astype(float)
    Xte = items.loc[lb, feat_codes].values.astype(float)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    ytr = meta.loc[dev, f"label_{domain}"].values
    yte = meta.loc[lb, f"label_{domain}"].values
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), float(average_precision_score(yte, p)), clf, sc


def evaluate_panel(items, meta, dev, lb, panel, domains=None):
    """For each domain: lockbox AUROC/AUPRC using full panel vs anchor-only,
    and the cross-diagnostic gain (panel − anchor)."""
    domains = C.DOMAINS if domains is None else domains
    panel_codes = [panel[d] for d in domains]
    rows = []
    for d in domains:
        a_roc, a_prc, _, _ = _fit_eval(items, meta, dev, lb, [panel[d]], d)
        p_roc, p_prc, _, _ = _fit_eval(items, meta, dev, lb, panel_codes, d)
        rows.append({
            "domain": d, "label": C.DOMAIN_LABEL[d], "anchor_item": panel[d],
            "anchor_auroc": round(a_roc, 3), "anchor_auprc": round(a_prc, 3),
            "panel_auroc": round(p_roc, 3), "panel_auprc": round(p_prc, 3),
            "crossdx_gain_auroc": round(p_roc - a_roc, 3),
        })
    return pd.DataFrame(rows)

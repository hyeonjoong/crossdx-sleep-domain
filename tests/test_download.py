# -*- coding: utf-8 -*-
"""Verified downloads of the open datasets (work order C6). No network is used:
urlopen is replaced by a fake response."""
import hashlib
import http.client
import os
import urllib.error

import pytest

import _download as D

PAYLOAD = b"export_id,question1\n1,2\n2,3\n" * 50
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
URL = "https://example.invalid/file.csv"


class FakeResponse:
    def __init__(self, body, length=None, fail_after=None):
        self.body = body
        self.pos = 0
        self.headers = {} if length is None else {"Content-Length": str(length)}
        self.fail_after = fail_after

    def read(self, n=-1):
        if self.fail_after is not None and self.pos >= self.fail_after:
            raise http.client.IncompleteRead(b"", 10)
        if n is None or n < 0:
            n = len(self.body) - self.pos
        chunk = self.body[self.pos:self.pos + n]
        self.pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, make):
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append((url, timeout))
        return make()

    monkeypatch.setattr(D, "urlopen", fake_urlopen)
    return calls


def _leftovers(path):
    return [p for p in (path, path + ".part") if os.path.exists(p)]


def test_correct_payload_lands(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, lambda: FakeResponse(PAYLOAD, len(PAYLOAD)))
    path = str(tmp_path / "sub" / "isi.csv")
    D.fetch_verified(URL, path, DIGEST)
    assert open(path, "rb").read() == PAYLOAD
    assert not os.path.exists(path + ".part")
    assert calls and calls[0][1] == 60


def test_truncated_payload_raises_and_leaves_nothing(monkeypatch, tmp_path):
    _serve(monkeypatch, lambda: FakeResponse(PAYLOAD[:500], len(PAYLOAD)))
    path = str(tmp_path / "isi.csv")
    with pytest.raises(D.DownloadError):
        D.fetch_verified(URL, path, DIGEST)
    assert _leftovers(path) == []


def test_wrong_content_raises_and_leaves_nothing(monkeypatch, tmp_path):
    _serve(monkeypatch, lambda: FakeResponse(PAYLOAD[:-5] + b"9,9\n\n"))
    path = str(tmp_path / "isi.csv")
    with pytest.raises(D.DownloadError) as e:
        D.fetch_verified(URL, path, DIGEST)
    assert "SHA-256" in str(e.value)
    assert _leftovers(path) == []


def test_connection_error_raises_and_leaves_nothing(monkeypatch, tmp_path):
    _serve(monkeypatch, lambda: FakeResponse(PAYLOAD, len(PAYLOAD), fail_after=100))
    path = str(tmp_path / "isi.csv")
    with pytest.raises(D.DownloadError):
        D.fetch_verified(URL, path, DIGEST)
    assert _leftovers(path) == []


def test_unreachable_server(monkeypatch, tmp_path):
    def boom(url, timeout=None):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(D, "urlopen", boom)
    path = str(tmp_path / "isi.csv")
    with pytest.raises(D.DownloadError) as e:
        D.fetch_verified(URL, path, DIGEST)
    assert URL in str(e.value)
    assert _leftovers(path) == []


def test_existing_good_file_is_reused(monkeypatch, tmp_path):
    calls = _serve(monkeypatch, lambda: FakeResponse(PAYLOAD))
    path = str(tmp_path / "isi.csv")
    with open(path, "wb") as fh:
        fh.write(PAYLOAD)
    D.fetch_verified(URL, path, DIGEST)
    assert calls == []


def test_existing_bad_file_is_fetched_again(monkeypatch, tmp_path, capsys):
    calls = _serve(monkeypatch, lambda: FakeResponse(PAYLOAD, len(PAYLOAD)))
    path = str(tmp_path / "isi.csv")
    with open(path, "wb") as fh:
        fh.write(PAYLOAD[:800])                  # an interrupted earlier download
    D.fetch_verified(URL, path, DIGEST)
    assert len(calls) == 1
    assert open(path, "rb").read() == PAYLOAD
    assert "isi.csv" in capsys.readouterr().out


def test_manifest_covers_the_published_inputs():
    assert set(D.MANIFEST) == {
        "isi.csv", "phq9.csv", "gad7.csv", "pss.csv", "demographic.csv",
        "cohortB_sri/sri_insomnia_items.csv", "cohortC_uk/uk_akram.xlsx"}
    assert all(len(v) == 64 for v in D.MANIFEST.values())


def test_source_md_lists_the_same_checksums():
    import re
    from conftest import ROOT
    with open(os.path.join(ROOT, "data", "realdata", "SOURCE.md"), encoding="utf-8") as fh:
        rows = dict(re.findall(r"^\| (\S+) \| ([0-9a-f]{64}) \|$", fh.read(), re.M))
    assert rows == D.MANIFEST

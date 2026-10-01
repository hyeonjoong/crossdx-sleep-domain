# -*- coding: utf-8 -*-
"""
_download.py - fetch the open datasets used by 06_realdata.py and 07_multicohort.py
and check them against the SHA-256 of the exact files behind the published
results (see data/realdata/SOURCE.md).

A file that is already on disk is reused only when its hash matches. A download
goes to <file>.part first and is moved into place only after the length and the
hash have been checked, so an interrupted download can never be mistaken for
the real file.
"""
import hashlib
import os
from urllib.request import urlopen

# SHA-256 of the files that reproduce RT1-RT7 (paths relative to data/realdata/)
MANIFEST = {
    "isi.csv": "3e748c0fd7827a271bcd7347a5ca442445596be9c6dbe51743661111ad0d6889",
    "phq9.csv": "0e837b6cf289c4083ebd131e6bc7a52d63eefae3113a8738b57eb8ef51b9aece",
    "gad7.csv": "e47ec29e61f30cadee85ed66d39092fa47f41fecb47544f30626fc33b021218f",
    "pss.csv": "ad844acb6c4239be4daf2400bb101dda770b41989523c5c7185c6c468da46397",
    "demographic.csv": "33247e3a6f1a8d9b64da54ab8f482b1bf26decfa3a825398d8c5d4d7564af693",
    "cohortB_sri/sri_insomnia_items.csv":
        "a68e167d8ad8d01b7a3f92b924ba109ae54e8e1c060b049ca586adb6ae9ad415",
    "cohortC_uk/uk_akram.xlsx":
        "e04b2a75dd4c46021c8520ba013fada3db73d03a694fea6afc06769ca0ae4e64",
}

TIMEOUT_S = 60
CHUNK = 1 << 16


class DownloadError(RuntimeError):
    """A download failed or did not match the expected file."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch_verified(url, path, sha256, tag="[download]"):
    """Make sure `path` holds the file with the given SHA-256, downloading it
    from `url` when it is missing or different. Raises DownloadError otherwise."""
    name = os.path.basename(path)
    if os.path.exists(path):
        if sha256_file(path) == sha256:
            return path
        print("%s %s on disk does not match the expected SHA-256 (incomplete or "
              "different file); downloading it again" % (tag, name))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    part = path + ".part"
    print("%s downloading %s from %s" % (tag, name, url))
    try:
        h = hashlib.sha256()
        n = 0
        with urlopen(url, timeout=TIMEOUT_S) as resp, open(part, "wb") as out:
            expected_len = resp.headers.get("Content-Length")
            while True:
                block = resp.read(CHUNK)
                if not block:
                    break
                out.write(block)
                h.update(block)
                n += len(block)
        if expected_len is not None and n != int(expected_len):
            raise DownloadError("%s: received %d of %s bytes from %s"
                                % (name, n, expected_len, url))
        if h.hexdigest() != sha256:
            raise DownloadError(
                "%s: SHA-256 of the download (%s) is not the expected %s. The file at "
                "%s may have changed; the published results were computed from the "
                "expected file (see data/realdata/SOURCE.md)." % (name, h.hexdigest(), sha256, url))
        os.replace(part, path)
    except DownloadError:
        raise
    except Exception as e:      # network, HTTP or disk errors
        raise DownloadError("%s: download from %s failed (%s: %s). Check the internet "
                            "connection and run the step again."
                            % (name, url, type(e).__name__, e)) from e
    finally:
        if os.path.exists(part):
            os.remove(part)
    return path

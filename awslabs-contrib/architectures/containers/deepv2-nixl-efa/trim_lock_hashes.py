#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Add sha256 hashes to a `name==version` lock, for the image's target only.

Target: CPython 3.12, manylinux x86_64 (the Ubuntu 22.04 image).
Input:  a lock with one `name==version` per line (comments/hashes ignored).
Output: the same pins, each followed by the sha256 of every file pip may pick
        for the target, from PyPI AND the PyTorch cu130 index. Both indexes are
        given to pip, and a few names (triton) are published on both with
        different bytes, so either choice must be covered.
A pin with no compatible file on either index fails loudly.

Usage (needs `packaging`):
    uv run --no-project --with packaging==25.0 python trim_lock_hashes.py pins.txt hashed.txt
"""
import json
import re
import sys
import urllib.request

from packaging.tags import cpython_tags, compatible_tags
from packaging.utils import canonicalize_name, parse_wheel_filename, parse_sdist_filename

PLATS = [f"manylinux_2_{m}_x86_64" for m in range(35, 4, -1)] + [
    "manylinux2014_x86_64", "manylinux2010_x86_64", "manylinux1_x86_64", "linux_x86_64"]
TAGS = set(cpython_tags((3, 12), platforms=PLATS)) | set(
    compatible_tags((3, 12), "cp312", platforms=PLATS))
INDEXES = ["https://pypi.org/simple", "https://download.pytorch.org/whl/cu130"]


def files(index, name):
    req = urllib.request.Request(
        f"{index}/{name}/", headers={"Accept": "application/vnd.pypi.simple.v1+json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
            ctype = r.headers.get("Content-Type", "")
    except Exception:
        return []
    if "json" in ctype:
        return [(f["filename"], f["hashes"].get("sha256")) for f in json.loads(body)["files"]]
    # PEP 503 HTML: <a href="...file#sha256=...">file</a>
    return [(m.group(2), m.group(1)) for m in re.finditer(
        r'href="[^"#]*#sha256=([0-9a-f]{64})"[^>]*>([^<]+)</a>', body)]


def compatible(fn, name, ver):
    if fn.endswith(".whl"):
        n, v, _, tags = parse_wheel_filename(fn)
        return n == name and str(v) == ver and bool(tags & TAGS), True
    if fn.endswith((".tar.gz", ".zip")):
        try:
            n, v = parse_sdist_filename(fn)
        except Exception:
            return False, False
        return n == name and str(v) == ver, False
    return False, False


out, missing = [], []
for line in open(sys.argv[1]):
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", line)
    if not m:
        continue
    name, ver = canonicalize_name(m.group(1)), m.group(2)
    wheels, sdists = [], []
    for idx in INDEXES:
        for fn, h in files(idx, name):
            ok, is_whl = compatible(fn, name, ver)
            if ok and h:
                (wheels if is_whl else sdists).append(h)
        # Both indexes are passed to pip, and a few names (e.g. triton) are
        # published on both with DIFFERENT bytes. List every compatible file
        # from both so either choice pip makes is still hash-checked.
    hashes = sorted(set(wheels or sdists))
    if not hashes:
        missing.append(f"{name}=={ver}")
        continue
    out.append(f"{m.group(1)}=={ver} \\\n" + " \\\n".join(f"    --hash=sha256:{h}" for h in hashes))

open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(f"pins={len(out)} hashes={sum(b.count('--hash') for b in out)} missing={missing}")
sys.exit(1 if missing else 0)

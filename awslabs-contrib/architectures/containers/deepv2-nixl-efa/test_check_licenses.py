#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Tests for check_licenses.py and sbom_emit.py (stdlib unittest + PyYAML).

Run:  python3 -m unittest test_check_licenses.py -v
"""
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_licenses as cl  # noqa: E402

ALLOW = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
ALIASES = {"MIT License": "MIT", "Expat (MIT/X11)": "MIT"}
MPL_EXC = {"id": "weak", "package_type": "pypi", "packages": ["orjson"],
           "licenses": ["MPL-2.0"], "pending_approval": True}


def verdict(name, lic, exceptions=(), allow_exc=frozenset(), ptype="pypi"):
    return cl.classify(name, "1", ptype, lic, ALLOW, True, list(exceptions),
                       {}, ALIASES, allow_exc)[0]


class Expressions(unittest.TestCase):
    def test_or_needs_one_side(self):
        # blake3: "CC0-1.0 OR Apache-2.0" was rewritten to AND and blocked
        self.assertEqual(verdict("blake3", "CC0-1.0 OR Apache-2.0"), "ALLOW")

    def test_and_needs_both_sides(self):
        self.assertEqual(verdict("x", "MIT AND GPL-3.0-only"), "BLOCK")

    def test_precedence_and_binds_tighter_than_or(self):
        # GPL-3.0-only OR (MIT AND Apache-2.0) -> the MIT AND Apache-2.0 side passes
        self.assertEqual(verdict("x", "GPL-3.0-only OR MIT AND Apache-2.0"), "ALLOW")
        # (GPL-3.0-only OR MIT) AND LGPL-2.1-only -> LGPL still required
        self.assertEqual(verdict("x", "(GPL-3.0-only OR MIT) AND LGPL-2.1-only"), "BLOCK")

    def test_with_needs_listed_exception(self):
        lic = "Apache-2.0 WITH LLVM-exception"
        self.assertEqual(verdict("nvtx", lic), "BLOCK")
        self.assertEqual(verdict("nvtx", lic, allow_exc={"LLVM-exception"}), "ALLOW")
        # an allowed exception does not rescue a non-allowed license
        self.assertEqual(verdict("x", "GPL-2.0-only WITH LLVM-exception",
                                 allow_exc={"LLVM-exception"}), "BLOCK")

    def test_whole_string_alias_with_parentheses(self):
        self.assertEqual(verdict("dbus-python", "Expat (MIT/X11)"), "ALLOW")

    def test_atom_alias_inside_expression(self):
        self.assertEqual(verdict("x", "MIT License AND Apache-2.0"), "ALLOW")

    def test_unparseable_is_one_opaque_atom(self):
        self.assertIsNone(cl.parse_license("MIT AND (Apache-2.0"))
        self.assertEqual(verdict("x", "MIT AND (Apache-2.0"), "BLOCK")


class Exceptions(unittest.TestCase):
    def test_exception_combines_with_allow_list(self):
        lic = "MPL-2.0 AND (Apache-2.0 OR MIT)"
        self.assertEqual(verdict("orjson", lic, [MPL_EXC]), "EXCEPTION:weak")

    def test_exception_is_scoped_to_named_packages(self):
        self.assertEqual(verdict("some-new-pkg", "MPL-2.0", [MPL_EXC]), "BLOCK")

    def test_exception_is_scoped_to_listed_licenses(self):
        self.assertEqual(verdict("orjson", "MPL-2.0 AND GPL-3.0-only", [MPL_EXC]), "BLOCK")

    def test_exception_is_scoped_to_package_type(self):
        self.assertEqual(verdict("orjson", "MPL-2.0", [MPL_EXC], ptype="deb"), "BLOCK")


class EndToEnd(unittest.TestCase):
    def run_gate(self, rows, pending):
        gate = textwrap.dedent(f"""
            license_policy:
              allow: [MIT, Apache-2.0]
              unknown_blocks: true
              exceptions:
                - id: weak
                  pending_approval: {str(pending).lower()}
                  package_type: pypi
                  packages: [certifi]
                  licenses: [MPL-2.0]
            """)
        with tempfile.TemporaryDirectory() as d:
            g, t = os.path.join(d, "gate.yaml"), os.path.join(d, "l.tsv")
            open(g, "w").write(gate)
            open(t, "w").write("".join("\t".join(r) + "\n" for r in rows))
            p = subprocess.run([sys.executable, os.path.join(HERE, "check_licenses.py"),
                                "--gate", g, t], capture_output=True, text=True)
            return p.returncode, p.stdout

    def test_pending_exception_is_not_sign_off(self):
        rc, out = self.run_gate([("certifi", "1", "pypi", "MPL-2.0")], pending=True)
        self.assertEqual(rc, 2, out)
        self.assertIn("LICENSE GATE: PENDING-APPROVAL", out)

    def test_approved_exception_signs_off(self):
        rc, out = self.run_gate([("certifi", "1", "pypi", "MPL-2.0")], pending=False)
        self.assertEqual(rc, 0, out)
        self.assertIn("LICENSE GATE: SIGN-OFF", out)

    def test_block_wins_over_pending(self):
        rc, out = self.run_gate([("certifi", "1", "pypi", "MPL-2.0"),
                                 ("bad", "1", "pypi", "GPL-3.0-only")], pending=True)
        self.assertEqual(rc, 1, out)
        self.assertIn("LICENSE GATE: BLOCK", out)


class SbomEmit(unittest.TestCase):
    def test_compound_findings_keep_their_meaning(self):
        sbom = {"artifacts": [{"name": "p", "version": "1", "type": "python",
                               "licenses": [{"spdxExpression": "CC0-1.0 OR Apache-2.0"},
                                            {"spdxExpression": "MIT"}]}]}
        p = subprocess.run([sys.executable, os.path.join(HERE, "sbom_emit.py")],
                           input=json.dumps(sbom), capture_output=True, text=True, check=True)
        lic = p.stdout.rstrip("\n").split("\t")[3]
        self.assertEqual(lic, "(CC0-1.0 OR Apache-2.0) AND MIT")
        self.assertEqual(verdict("p", lic), "ALLOW")


if __name__ == "__main__":
    unittest.main()

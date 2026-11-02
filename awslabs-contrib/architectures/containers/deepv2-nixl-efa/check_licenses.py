#!/usr/bin/env python3
"""
License gate for the DeepEP V2 + NIXL + EFA container pipeline.

Reads an SBOM component list and the license policy from validation-gate.yaml,
then BLOCKS publish if any component's license is neither on the OSS allow-list
nor covered by a scoped, named exception (base-OS copyleft, NVIDIA proprietary).

Input TSV (one row per component), tab-separated, in this exact column order:
    name <TAB> version <TAB> type <TAB> license

  - `type` is syft's package type: `deb` for dpkg (OS) packages, `python` /
    `python-package` for pip wheels, etc. It is what lets the base-OS exception
    excuse a GPL *dpkg* package while still blocking a GPL *pip wheel*.
  - `license` is a single SPDX id or a syft compound string
    ("A AND B", "LicenseRef-<hash>", "NOASSERTION").

Produce the TSV with:
    syft "$IMG" -o spdx-json \\
      | jq -r '.packages[] | [
          .name, .versionInfo,
          ((.externalRefs[]? | select(.referenceType=="purl") | .referenceLocator)
             | capture("pkg:(?<t>[^/]+)/").t) // "unknown",
          (.licenseConcluded // .licenseDeclared // "NOASSERTION")
        ] | @tsv' > licenses.tsv
    # (type column falls back to "unknown" when syft emits no purl)

Exit status: 0 = all clear (SIGN-OFF), 1 = at least one component BLOCKS,
             2 = nothing blocks, but some component is covered only by an
                 exception marked `pending_approval: true` (PENDING-APPROVAL).

Usage:
    check_licenses.py --gate validation-gate.yaml licenses.tsv
    check_licenses.py --allowlist aws-oss-allowlist.txt licenses.tsv   # allow-list only, no exceptions
"""
import argparse
import fnmatch
import re
import sys

try:
    import yaml
except ImportError:
    yaml = None


def load_policy(gate_path, allowlist_path):
    """Return (allow:set, unknown_blocks:bool, exceptions:list) from the gate
    yaml, or a bare allow-list file if --allowlist was given instead."""
    if allowlist_path:
        with open(allowlist_path) as f:
            allow = {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        return allow, True, [], {}, None, {}, set()
    if yaml is None:
        sys.exit("PyYAML is required to read the gate file; `pip install pyyaml` "
                 "or pass --allowlist with a plain allow-list file.")
    with open(gate_path) as f:
        gate = yaml.safe_load(f)
    pol = gate.get("license_policy", {})
    allow = set(pol.get("allow", []))
    unknown_blocks = bool(pol.get("unknown_blocks", True))
    exceptions = pol.get("exceptions", []) or []
    known = pol.get("known_licenses", {}) or {}
    include_types = (pol.get("scan_scope", {}) or {}).get("include_types")
    aliases = pol.get("license_aliases", {}) or {}
    # SPDX exceptions ("A WITH <exc>") accepted on top of an allowed license A
    allow_exc = set(pol.get("allow_exceptions", []) or [])
    return allow, unknown_blocks, exceptions, known, include_types, aliases, allow_exc


# ---------------------------------------------------------------------------
# SPDX license expressions.
#
# A license string is parsed as an SPDX expression so each operator keeps its
# meaning:  "A OR B"   -> acceptable if EITHER side is (the licensee may choose),
#           "A AND B"  -> acceptable only if BOTH are,
#           "A WITH X" -> acceptable if A is AND the exception X is listed under
#                         allow_exceptions (an SPDX exception only ADDS permissions,
#                         e.g. LLVM-exception). Precedence: WITH > AND > OR.
# A string that does not parse (free text with stray parentheses) is one opaque
# atom: it passes only by a whole-string alias or allow-list match.
# Tree nodes: ("atom", id) | ("with", node, exc_id) | ("and"|"or", [nodes]).
# ---------------------------------------------------------------------------
_TOKEN = re.compile(r"\(|\)|\s+(?:AND|OR|WITH)\s+")


def _tokenize(lic):
    toks, pos = [], 0
    for m in _TOKEN.finditer(lic):
        text = lic[pos:m.start()].strip()
        if text:
            toks.append(("id", text))
        op = m.group(0).strip()
        toks.append(("op", op) if op in ("AND", "OR", "WITH") else (op, op))
        pos = m.end()
    tail = lic[pos:].strip()
    if tail:
        toks.append(("id", tail))
    return toks


def parse_license(lic):
    """Parse an SPDX-style expression; None if it does not parse."""
    toks = _tokenize(lic)
    i = 0

    def peek():
        return toks[i] if i < len(toks) else (None, None)

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def primary():
        kind, val = peek()
        if kind == "(":
            take()
            node = disj()
            if peek()[0] != ")":
                raise ValueError("unbalanced parenthesis")
            take()
            return node
        if kind == "id":
            take()
            return ("atom", val)
        raise ValueError(f"unexpected token {val!r}")

    def with_expr():
        node = primary()
        if peek() == ("op", "WITH"):
            take()
            kind, exc = take() if i < len(toks) else (None, None)
            if kind != "id":
                raise ValueError("WITH needs an exception id")
            node = ("with", node, exc)
        return node

    def chain(sub, op, tag):
        nodes = [sub()]
        while peek() == ("op", op):
            take()
            nodes.append(sub())
        return nodes[0] if len(nodes) == 1 else (tag, nodes)

    def conj():
        return chain(with_expr, "AND", "and")

    def disj():
        return chain(conj, "OR", "or")

    try:
        node = disj()
        if i != len(toks):
            raise ValueError("trailing tokens")
        return node
    except (ValueError, IndexError):
        return None


def render_license(node):
    kind = node[0]
    if kind == "atom":
        return node[1]
    if kind == "with":
        return f"{render_license(node[1])} WITH {node[2]}"
    sep = " AND " if kind == "and" else " OR "
    parts = []
    for child in node[1]:
        text = render_license(child)
        # parenthesise an OR inside an AND so precedence survives re-rendering
        if kind == "and" and child[0] == "or":
            text = f"({text})"
        parts.append(text)
    return sep.join(parts)


def _map_atoms(node, aliases):
    kind = node[0]
    if kind == "atom":
        return ("atom", aliases.get(node[1], node[1]))
    if kind == "with":
        return ("with", _map_atoms(node[1], aliases), node[2])
    return (kind, [_map_atoms(c, aliases) for c in node[1]])


def license_tree(lic, aliases):
    """Normalize free-text names to SPDX ids and return the expression tree.
    A whole-string alias wins (free text may itself contain parentheses);
    otherwise each atom of the parsed expression is looked up."""
    aliases = aliases or {}
    if lic in aliases:
        lic = aliases[lic]
    node = parse_license(lic)
    if node is None:
        return ("atom", lic)
    return _map_atoms(node, aliases)


def normalize_license(lic, aliases):
    """Normalized SPDX expression string (kept for callers and tests)."""
    return render_license(license_tree(lic, aliases))


def license_ok(node, ids, exceptions_ok=()):
    """True if expression `node` is satisfiable with license ids `ids`."""
    kind = node[0]
    if kind == "atom":
        return node[1] in ids
    if kind == "with":
        return license_ok(node[1], ids, exceptions_ok) and node[2] in exceptions_ok
    results = [license_ok(c, ids, exceptions_ok) for c in node[1]]
    return all(results) if kind == "and" else any(results)


def split_license(lic):
    """Every license id mentioned in `lic` (operators and parentheses removed)."""
    node = parse_license(lic)
    if node is None:
        return {lic.strip()} if lic.strip() else set()
    out = set()

    def walk(n):
        if n[0] == "atom":
            out.add(n[1])
        elif n[0] == "with":
            walk(n[1])
            out.add(n[2])
        else:
            for c in n[1]:
                walk(c)
    walk(node)
    return out


UNKNOWN = {"NOASSERTION", "NONE", "", "UNKNOWN"}


def is_unknown(lic):
    if lic.strip().upper() in UNKNOWN:
        return True
    # syft emits LicenseRef-<hash> when it cannot map to a real SPDX id
    return lic.startswith("LicenseRef-") and len(lic) > 20


def exception_covers(name, ptype, lic, exc, allow=frozenset(), allow_exc=frozenset(),
                     aliases=None):
    """True if exception `exc` excuses component (name, ptype) under license
    string `lic`.

    Default: requires BOTH a package-name glob match AND the license expression
    to be satisfiable with the allow-list PLUS the exception's own license ids
    (so "MPL-2.0 AND (Apache-2.0 OR MIT)" is covered by an MPL-2.0 exception).

    When the exception sets `excuse_compound: true` (base-OS), a name+type match
    is ENOUGH -- the whole compound string is excused regardless of its atoms.
    This is for distro copyright files that over-list every license of every
    file in the package (e.g. coreutils -> "BSD-3 AND GPL-3.0 AND ISC AND
    GFDL"); the package's effective redistribution is governed by the base-OS
    policy, not by per-atom matching. It still requires the package NAME and
    TYPE to match, so it is not a blanket "ignore this layer".
    """
    # optional package_type narrowing (e.g. base-os only excuses `deb`)
    want_type = exc.get("package_type")
    if want_type and ptype and ptype != want_type:
        return False
    # name must match one of the globs
    if not any(fnmatch.fnmatch(name, pat) for pat in exc.get("packages", [])):
        return False
    # base-OS: a name+type match excuses the whole compound string
    if exc.get("excuse_compound"):
        return True
    # otherwise the expression must be satisfiable with allow + exception ids
    exc_lic = set(exc.get("licenses", []))
    if lic in exc_lic:
        return True
    return license_ok(license_tree(lic, aliases), set(allow) | exc_lic, allow_exc)


def classify(name, version, ptype, lic, allow, unknown_blocks, exceptions,
             known_licenses=None, aliases=None, allow_exc=frozenset()):
    """Return (verdict, detail). verdict in {ALLOW, EXCEPTION:<id>, PIN, BLOCK}.

    `known_licenses` pins the license for source-built components no scanner can
    read (a `git clone && make` install leaves no package metadata). A pin
    OVERRIDES the SBOM's NOASSERTION, then the pinned license is checked against
    the allow-list -- a pin is a resolved fact, not a free pass.
    """
    # 0. source-built pin overrides the (missing) SBOM license
    if known_licenses:
        for pat, pinned in known_licenses.items():
            if fnmatch.fnmatch(name, pat):
                lic = pinned
                break
    # 0b. normalize free-text license strings to SPDX ids (after pin, so a pin
    #     value is also normalized if it uses a free-text form)
    tree = license_tree(lic, aliases)
    lic = render_license(tree)
    # 1. whole string, or the SPDX expression, satisfiable by the allow-list -> ALLOW
    if lic in allow or license_ok(tree, allow, allow_exc):
        return "ALLOW", lic
    # 2. a scoped exception?
    for exc in exceptions:
        if exception_covers(name, ptype, lic, exc, allow, allow_exc, aliases):
            return f"EXCEPTION:{exc.get('id')}", lic
    # 3. unknown/unmapped
    if is_unknown(lic):
        return ("BLOCK", "unknown/NOASSERTION") if unknown_blocks else ("ALLOW", "unknown-permitted")
    # 4. everything else blocks
    return "BLOCK", lic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", help="SBOM TSV: name<TAB>version<TAB>type<TAB>license")
    ap.add_argument("--gate", default="validation-gate.yaml",
                    help="gate yaml with license_policy (default: validation-gate.yaml)")
    ap.add_argument("--allowlist", help="plain allow-list file (one SPDX id/line); "
                    "disables exceptions")
    ap.add_argument("--quiet", action="store_true", help="only print the verdict + blockers")
    args = ap.parse_args()

    allow, unknown_blocks, exceptions, known_licenses, include_types, aliases, allow_exc = \
        load_policy(args.gate, args.allowlist)
    pending_ids = {f"EXCEPTION:{e.get('id')}" for e in exceptions if e.get("pending_approval")}

    seen = set()
    counts = {"ALLOW": 0, "BLOCK": 0}
    by_exc = {}
    skipped = 0
    blockers = []
    pending = []
    with open(args.tsv) as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            cols = raw.split("\t")
            # tolerate a 3-col (name,version,license) legacy TSV: type = unknown
            if len(cols) == 3:
                name, version, lic = cols
                ptype = "unknown"
            elif len(cols) >= 4:
                name, version, ptype, lic = cols[0], cols[1], cols[2], cols[3]
            else:
                continue
            key = (name, version, ptype, lic)
            if key in seen:
                continue
            seen.add(key)
            # scan scope: skip types outside include_types, UNLESS the component
            # is explicitly pinned (a source-built dep may carry an odd type)
            is_pinned = bool(known_licenses) and any(
                fnmatch.fnmatch(name, pat) for pat in known_licenses)
            if include_types and ptype not in include_types and not is_pinned:
                skipped += 1
                continue
            verdict, detail = classify(name, version, ptype, lic, allow,
                                       unknown_blocks, exceptions, known_licenses,
                                       aliases, allow_exc)
            if verdict == "ALLOW":
                counts["ALLOW"] += 1
            elif verdict.startswith("EXCEPTION:"):
                by_exc[verdict] = by_exc.get(verdict, 0) + 1
                if verdict in pending_ids:
                    pending.append((verdict[len("EXCEPTION:"):], name, version, ptype, detail))
            else:
                counts["BLOCK"] += 1
                # show the EFFECTIVE license (post pin + normalize), with the
                # raw SBOM value in parens when it differs, so the reason is clear
                eff = detail
                shown = eff if eff == lic else f"{eff} (sbom: {lic})"
                blockers.append((name, version, ptype, shown))

    total = counts["ALLOW"] + counts["BLOCK"] + sum(by_exc.values())
    if not args.quiet:
        print(f"components in scope (distinct): {total}   (skipped out-of-scope type: {skipped})")
        print(f"  allow-list clean   : {counts['ALLOW']}")
        for exc_id in sorted(by_exc):
            tag = "  (PENDING APPROVAL)" if exc_id in pending_ids else ""
            print(f"  {exc_id:19s}: {by_exc[exc_id]}{tag}")
        print(f"  BLOCK              : {counts['BLOCK']}")
    if blockers:
        print("\nBLOCKED components (license not on allow-list and not excepted):")
        for name, version, ptype, lic in sorted(blockers):
            print(f"  {name} {version} [{ptype}] -> {lic}")
        print("\nLICENSE GATE: BLOCK")
        return 1
    if pending:
        # Covered only by an exception that is not approved yet: not a BLOCK
        # (the license is identified and scoped), but not publishable either.
        print("\nCovered by an exception PENDING APPROVAL (set pending_approval: false"
              " in the gate once approved):")
        for exc_id, name, version, ptype, lic in sorted(pending):
            print(f"  [{exc_id}] {name} {version} [{ptype}] -> {lic}")
        print("\nLICENSE GATE: PENDING-APPROVAL")
        return 2
    print("\nLICENSE GATE: SIGN-OFF")
    return 0


if __name__ == "__main__":
    sys.exit(main())

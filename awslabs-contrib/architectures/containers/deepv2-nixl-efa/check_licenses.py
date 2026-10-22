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

Exit status: 0 = all clear (SIGN-OFF), 1 = at least one component BLOCKS.

Usage:
    check_licenses.py --gate validation-gate.yaml licenses.tsv
    check_licenses.py --allowlist aws-oss-allowlist.txt licenses.tsv   # allow-list only, no exceptions
"""
import argparse
import fnmatch
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
        return allow, True, [], {}, None, {}
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
    return allow, unknown_blocks, exceptions, known, include_types, aliases


def normalize_license(lic, aliases):
    """Map a free-text license string (and each atom of a compound) to its SPDX
    id via the aliases map. Whole-string match wins; otherwise each atom is
    looked up and re-joined with AND. Unmatched atoms pass through unchanged."""
    if not aliases:
        return lic
    if lic in aliases:
        return aliases[lic]
    atoms = split_license(lic)
    if not atoms:
        return lic
    mapped = [aliases.get(a, a) for a in atoms]
    # dedupe preserving order
    out = []
    for m in mapped:
        if m not in out:
            out.append(m)
    return " AND ".join(out)


def split_license(lic):
    """Split a syft compound license into atomic ids.
    'A AND B', 'A OR B', 'A WITH B' -> {A, B}. Parens/strip handled."""
    s = lic.replace("(", " ").replace(")", " ")
    for sep in (" AND ", " OR ", " WITH "):
        s = s.replace(sep, "\n")
    return {p.strip() for p in s.split("\n") if p.strip()}


UNKNOWN = {"NOASSERTION", "NONE", "", "UNKNOWN"}


def is_unknown(lic):
    if lic.strip().upper() in UNKNOWN:
        return True
    # syft emits LicenseRef-<hash> when it cannot map to a real SPDX id
    return lic.startswith("LicenseRef-") and len(lic) > 20


def exception_covers(name, ptype, lic, exc):
    """True if exception `exc` excuses component (name, ptype) under license
    string `lic`.

    Default: requires BOTH a package-name glob match AND every license atom to
    be in the exception's allowed ids.

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
    # otherwise the license string (whole, or every atom) must be listed
    exc_lic = set(exc.get("licenses", []))
    if lic in exc_lic:
        return True
    atoms = split_license(lic)
    return bool(atoms) and atoms.issubset(exc_lic)


def classify(name, version, ptype, lic, allow, unknown_blocks, exceptions,
             known_licenses=None, aliases=None):
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
    lic = normalize_license(lic, aliases)
    # 1. whole-string or all-atoms on the OSS allow-list -> ALLOW
    atoms = split_license(lic)
    if lic in allow or (atoms and atoms.issubset(allow)):
        return "ALLOW", lic
    # 2. a scoped exception?
    for exc in exceptions:
        if exception_covers(name, ptype, lic, exc):
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

    allow, unknown_blocks, exceptions, known_licenses, include_types, aliases = load_policy(
        args.gate, args.allowlist)

    seen = set()
    counts = {"ALLOW": 0, "BLOCK": 0}
    by_exc = {}
    skipped = 0
    blockers = []
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
                                       aliases)
            if verdict == "ALLOW":
                counts["ALLOW"] += 1
            elif verdict.startswith("EXCEPTION:"):
                by_exc[verdict] = by_exc.get(verdict, 0) + 1
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
            print(f"  {exc_id:19s}: {by_exc[exc_id]}")
        print(f"  BLOCK              : {counts['BLOCK']}")
    if blockers:
        print("\nBLOCKED components (license not on allow-list and not excepted):")
        for name, version, ptype, lic in sorted(blockers):
            print(f"  {name} {version} [{ptype}] -> {lic}")
        print("\nLICENSE GATE: BLOCK")
        return 1
    print("\nLICENSE GATE: SIGN-OFF")
    return 0


if __name__ == "__main__":
    sys.exit(main())

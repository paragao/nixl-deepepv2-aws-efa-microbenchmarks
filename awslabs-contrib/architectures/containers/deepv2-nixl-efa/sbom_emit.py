#!/usr/bin/env python3
# Read syft NATIVE json on stdin, emit name<TAB>version<TAB>type<TAB>license.
# syft json carries per-license spdxExpression (spdx-json collapses/mangles it).
import sys, json
d = json.load(sys.stdin)
for p in d.get("artifacts", []):
    name = p.get("name", "")
    version = p.get("version", "")
    ptype = p.get("type", "unknown")   # deb-package, python, go-module, ...
    # normalize type to a short tag matching the gate's package_type
    tmap = {"deb-package": "deb", "python": "pypi", "python-package": "pypi",
            "go-module": "golang", "rpm-package": "rpm", "java-archive": "maven"}
    ptype = tmap.get(ptype, ptype)
    lics = []
    for l in p.get("licenses", []) or []:
        spdx = l.get("spdxExpression") or ""
        val = l.get("value") or ""
        # prefer a real SPDX id; fall back to the raw value (may be sha256:<hash>)
        lics.append(spdx if spdx else val)
    if not lics:
        lic = "NOASSERTION"
    else:
        # join multiple license findings with AND (syft lists them separately)
        uniq = []
        for x in lics:
            if x and x not in uniq:
                uniq.append(x)
        lic = " AND ".join(uniq) if uniq else "NOASSERTION"
    # free-text copyright files syft couldn't classify come through as
    # "sha256:<hash>" -- mark them unknown so the gate treats them as such
    if lic.startswith("sha256:") or all(x.startswith("sha256:") for x in lics if x):
        lic = "NOASSERTION"
    print("%s\t%s\t%s\t%s" % (name, version, ptype, lic))

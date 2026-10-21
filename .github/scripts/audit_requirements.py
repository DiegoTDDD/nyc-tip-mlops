"""Fail when any pinned dependency has a known vulnerability WITH a published fix.
Vulnerabilities with no fix yet are listed but accepted (nothing can be done about them)."""
import glob, json, os, re, subprocess, sys
files = sorted(f for f in glob.glob("**/requirements*.txt", recursive=True) if "node_modules" not in f)
fixable, unfixable = [], []
def run_audit(path):
    return subprocess.run([sys.executable, "-m", "pip_audit", "-r", path, "--no-deps", "--disable-pip",
                           "--format", "json", "--progress-spinner", "off"], capture_output=True, text=True)

for f in files:
    r = run_audit(f)
    if not r.stdout.strip().startswith("{"):
        # Not fully pinned (e.g. "pandas>=2.0"): resolve to the exact versions pip would install, then audit those.
        resolved = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "resolved-" + f.replace("/", "_"))
        c = subprocess.run([sys.executable, "-m", "uv", "pip", "compile", f, "--python-version", "3.11", "-o", resolved, "-q", "--refresh"],
                           capture_output=True, text=True)
        if c.returncode != 0:
            print(f"::error::could not resolve {f}: {c.stderr[-300:]}"); sys.exit(2)
        r = run_audit(resolved)
        print(f"(resolved unpinned file {f} before auditing)")
    try:
        data = json.loads(r.stdout)
    except Exception:
        print(f"::error::pip-audit failed on {f}: {r.stderr[-300:]}"); sys.exit(2)
    for dep in data.get("dependencies", []):
        for v in dep.get("vulns", []):
            row = (f, dep["name"], dep["version"], v["id"], ",".join(v.get("fix_versions", [])))
            (fixable if v.get("fix_versions") else unfixable).append(row)
def _norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def _is_line_for(line, pkg):
    m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*(?:==|>=|<=|~=|!=|<|>|;|$|\s)", line)
    return bool(m) and _norm(m.group(1)) == _norm(pkg)


def installable(req_file, pkg, fix):
    """Can `pkg>=fix` be installed together with everything else in req_file?
    Pins of packages that get in the way (named by the resolver) are relaxed, up to 4 times, because
    the real fix may be 'upgrade X as well'. Returns (ok, relaxed_packages, reason)."""
    lines = open(req_file, encoding="utf-8").read().splitlines()
    relax = set()
    reason = ""
    for _ in range(5):
        kept = []
        for l in lines:
            if _is_line_for(l, pkg):
                continue
            if any(_is_line_for(l, r) for r in relax):
                # a blocking package may be UPGRADED (never downgraded): turn "x==1.2" into "x>=1.2"
                l = re.sub(r"^(\s*[A-Za-z0-9_.\-]+(?:\[[^\]]*\])?\s*)==", r"\1>=", l)
            kept.append(l)
        kept.append(f"{pkg}>={fix}")
        tmp = os.path.join(os.environ.get("RUNNER_TEMP", "/tmp"), "installable-test.txt")
        open(tmp, "w", encoding="utf-8").write("\n".join(kept) + "\n")
        c = subprocess.run([sys.executable, "-m", "uv", "pip", "compile", tmp, "--python-version", "3.11",
                            "-q", "-o", os.devnull, "--refresh"], capture_output=True, text=True)
        if c.returncode == 0:
            return True, sorted(relax), ""
        err = c.stderr
        m = re.search(r"Because[^\n]*(?:\n[^\n]*)*?(?:and you require|we can conclude)[^\n]*", err)
        reason = " ".join(x.strip() for x in err.splitlines() if x.strip() and not x.strip().startswith(("×", "╰")))[:260]
        blockers = [b for b in re.findall(r"([A-Za-z0-9_.\-]+)==[0-9][^\s,]* depends on", err)
                    if _norm(b) != _norm(pkg) and _norm(b) not in relax]
        if not blockers:
            return False, sorted(relax), reason
        relax.update(_norm(b) for b in blockers)
    return False, sorted(relax), reason


# Split "fix available" into: really installable (must be upgraded) and blocked (accepted until upstream allows it)
from packaging.version import InvalidVersion as _IV, Version as _V
blocked = []
if fixable:
    groups = {}
    for (f_, name_, ver_, vid_, fix_) in fixable:
        groups.setdefault((f_, name_, ver_), []).append((vid_, fix_))
    real = []
    for (f_, name_, ver_), items in groups.items():
        fixes = []
        for _, fx in items:
            for x in fx.split(","):
                try:
                    fixes.append(_V(x))
                except _IV:
                    pass
        need = str(max(fixes)) if fixes else items[0][1].split(",")[0]
        ok, relaxed, why = installable(f_, name_, need)
        if ok:
            extra = f" (also upgrade {', '.join(relaxed)})" if relaxed else ""
            for vid_, fx in items:
                real.append((f_, name_, ver_, vid_, fx + extra))
        else:
            blocked.append((f_, name_, ver_, need, [v for v, _ in items], why))
    fixable = real

print(f"files audited: {len(files)}")
import json
import re
import urllib.request

from packaging.version import InvalidVersion, Version


def pypi(path):
    with urllib.request.urlopen(f"https://pypi.org/pypi/{path}/json", timeout=30) as r:
        return json.load(r)


# A package installed from a pinned git commit is invisible to the PyPI audit above. To make sure it is
# never forgotten, such a line must be followed by a directive such as
#     # audit-git-pin: nltk newer-than 3.10.3
# and this check FAILS as soon as PyPI has a release newer than that version with no known vulnerability.
watch_errors = []
for f in files:
    lines = open(f, encoding="utf-8").read().splitlines()
    directives = {}
    for l in lines:
        m = re.match(r"#\s*audit-git-pin:\s*([A-Za-z0-9_.\-]+)\s+newer-than\s+(\S+)", l)
        if m:
            directives[m.group(1).lower()] = m.group(2)
    for l in lines:
        if l.lstrip().startswith("#") or " @ " not in l:
            continue
        name = l.split("@")[0].strip().split("[")[0]
        base = directives.get(name.lower())
        print(f"note: {f}: {name} is installed from a pinned git commit")
        if base is None:
            watch_errors.append(f"{f}: {name} is pinned to a git commit but has no '# audit-git-pin: {name} newer-than <version>' line, so nothing watches for a fixed release")
            continue
        try:
            releases = pypi(name)["releases"]
        except Exception as e:
            print(f"::warning::could not reach PyPI to check {name}: {e}")
            continue
        newer = []
        for v, rel_files in releases.items():
            try:
                pv = Version(v)
            except InvalidVersion:
                continue
            if pv.is_prerelease or pv.is_devrelease or not rel_files or all(x.get("yanked") for x in rel_files):
                continue
            if pv > Version(base):
                newer.append(pv)
        newer.sort()
        clean = []
        for pv in newer:
            try:
                if not pypi(f"{name}/{pv}")["vulnerabilities"]:
                    clean.append(pv)
            except Exception:
                continue
        if clean:
            watch_errors.append(f"{f}: {name} {clean[-1]} is now on PyPI with no known vulnerabilities. Replace the git pin with {name}=={clean[-1]} and delete the audit-git-pin line")
        elif newer:
            print(f"note: {name} has newer releases ({', '.join(str(v) for v in newer)}) but all still have known vulnerabilities; the git pin stays")
        else:
            print(f"note: no {name} release newer than {base} yet; the git pin stays")
print("\nNo fix published yet (accepted on purpose):")
for r in sorted(set((x[1], x[2], x[3]) for x in unfixable)): print("  ", *r)
print("\nFix exists but cannot be installed yet (accepted until a package allows it; re-checked every run):")
for (f_, name_, ver_, need, ids, why) in blocked:
    print(f"   {f_} {name_} {ver_} -> {need} ({len(ids)} advisories). Blocked: {why[:200]}")
    print(f"::warning::{f_}: {name_} {ver_} has a fix ({need}) that cannot be installed together with the other packages yet")
print("\nFix available (must be upgraded):")
for r in fixable: print("  ", *r)
for r in fixable:
    print(f"::error::{r[0]}: {r[1]} {r[2]} -> upgrade to {r[4]} ({r[3]})")
for e in watch_errors:
    print(f"::error::{e}")
if fixable or watch_errors:
    sys.exit(1)
print("\nOK: no vulnerability with an available fix, and no git pin can be replaced by a clean release.")

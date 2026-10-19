"""Fail when any pinned dependency has a known vulnerability WITH a published fix.
Vulnerabilities with no fix yet are listed but accepted (nothing can be done about them)."""
import glob, json, subprocess, sys, os
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
        c = subprocess.run([sys.executable, "-m", "uv", "pip", "compile", f, "--python-version", "3.11", "-o", resolved, "-q"],
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
print(f"files audited: {len(files)}")
for f in files:
    for line in open(f, encoding="utf-8"):
        if " @ " in line and not line.lstrip().startswith("#"):
            print(f"note: {f}: {line.split('@')[0].strip()} is installed from a pinned git commit (not covered by the PyPI audit)")
print("\nNo fix published yet (accepted on purpose):")
for r in sorted(set((x[1], x[2], x[3]) for x in unfixable)): print("  ", *r)
print("\nFix available (must be upgraded):")
for r in fixable: print("  ", *r)
if fixable:
    for r in fixable: print(f"::error::{r[0]}: {r[1]} {r[2]} -> upgrade to {r[4]} ({r[3]})")
    sys.exit(1)
print("\nOK: no vulnerability with an available fix.")

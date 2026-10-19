import json, re, subprocess, sys, platform, sklearn, glob
name = sys.argv[1] if len(sys.argv) > 1 else "paper_final"
# table rows are generated with a trailing row terminator; the template supplies it, so strip it here
for f in glob.glob("tables/tab_*.tex"):
    t = open(f, encoding="utf-8").read().rstrip()
    if t.endswith("\\\\"): t = t[:-2].rstrip()
    open(f, "w", encoding="utf-8").write(t)
M = {}
for mf in ("results/macros.json", "results/macros_pop.json", "results/macros_val.json", "results/macros_paired.json", "results/macros_robust.json", "results/macros_robust2.json"):
    try: M.update(json.load(open(mf)))
    except FileNotFoundError: pass
M["pyver"] = platform.python_version(); M["skver"] = sklearn.__version__
s = open(name + ".tex.in", encoding="utf-8").read()
miss = sorted(set(re.findall(r"@@([\w-]+)@@", s)) - set(M))
if miss: sys.exit("missing macros: %s" % miss)
open(name + ".tex", "w", encoding="utf-8").write(re.sub(r"@@([\w-]+)@@", lambda m: str(M[m.group(1)]), s))
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"], capture_output=True, text=True)
    if r.returncode: print(r.stdout[-3000:]); sys.exit(1)
print("ok")

"""Validation table, acceptance statistics and prior-range table for the population data set."""
import json, numpy as np
from pop_data_utils import *

V = json.load(open("results_validation.json")); M = {}
rows = []
for r in V["six"]:
    rows.append(f"{r['eos']} & {r['R']:.4f} & {r['R_ref']:.4f} & {r['Lam']:.1f} & {r['Lam_ref']:.1f} & {r['f']:.2f} & {r['f_ref']:.2f} & {100*(r['f']/r['f_ref']-1):+.4f} & {r['tau']:.4f} \\\\")
open("tables/tab_valid.tex", "w").write("\n".join(rows).rstrip()[:-2])
M["valFmax"] = f"{max(abs(100*(r['f']/r['f_ref']-1)) for r in V['six']):.3f}"
M["valRmax"] = f"{max(abs(r['R']-r['R_ref']) for r in V['six']):.4f}"
M["valLmax"] = f"{max(abs(100*(r['Lam']/r['Lam_ref']-1)) for r in V['six']):.2f}"
M["valPscale"] = f"{V['p_scale']:.3f}"
M["valDM100"] = f"{V['dm_shift_G100']:.1f}"; M["valDM100ref"] = f"{V['dm_shift_ref_G100']:.1f}"; M["valDM200"] = f"{V['dm_shift_G200']:.1f}"
M["valTab"] = f"{100*abs(V['tab_vs_analytic_f']/V['f0']-1):.3f}"
M["valTauMin"] = f"{min(r['tau'] for r in V['six']):.2f}"; M["valTauMax"] = f"{max(r['tau'] for r in V['six']):.2f}"

recs = load_all()
for k in range(4):
    r = [x for x in recs if x["cls"] == k]
    acc = 1 / np.mean([x["tries"] for x in r])
    M[f"acc{k}"] = f"{100*acc:.0f}"
    why = {}
    for x in r:
        for w, n in x["reject"].items(): why[w] = why.get(w, 0) + n
    M[f"why{k}"] = ", ".join(f"{w} {n}" for w, n in sorted(why.items()))
M["nModes"] = sum(len(x["masses"]) for x in recs)
M["secEoS"] = f"{np.mean([x['seconds'] for x in recs]):.0f}"
M["MmaxMax"] = f"{max(x['Mmax'] for x in recs):.2f}"
json.dump(M, open("macros_val.json", "w"), indent=1)
print(M)

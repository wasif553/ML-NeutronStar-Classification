"""Macros/tables for the analytic benchmarks, strange-star convergence, dark-matter prescription and prior-width analyses."""
import json, os, numpy as np
M = {}
if os.path.exists("results_analytic.json"):
    A = json.load(open("results_analytic.json")); p0 = A["points"][0]
    M["anaC"] = f"{p0['C']:.3f}"; M["anaDf0"] = f"{p0['df_rel']:.1e}"; M["anaDfInt"] = f"{abs(A['f_intercept']):.0e}"
    M["anaK"] = f"{p0['k2']:.3f}"; M["anaKunc"] = f"{p0['k2_uncorrected']:.4f}"; M["anaKint"] = f"{abs(A['k2_intercept']) * 100:.1f}"
    M["anaKslope"] = f"{A['k2_slope']:.1f}"
    M["anaTauInt"] = f"{abs(A['tau_intercept']):.0e}"; M["anaTauSlope"] = f"{A['tau_slope']:.1f}"
if os.path.exists("results_convergence_strange.json"):
    C = json.load(open("results_convergence_strange.json"))
    M["cvSf"] = f"{max(abs(v['df_rel']) for s in C.values() for v in s.values()):.0e}"; M["cvSt"] = f"{max(abs(v['dtau_rel']) for s in C.values() for v in s.values()):.0e}"
if os.path.exists("results_dm_prescription.json"):
    P = json.load(open("results_dm_prescription.json"))["summary"]
    for m0, tag in (("931.494", "a"), ("935.0", "b"), ("938.9", "c"), ("931.192", "d")):
        M[f"presMed_{tag}"] = f"{100*P[m0]['median']:.2f}"
        M[f"presLo_{tag}"] = f"{100*P[m0]['p5']:.2f}"; M[f"presHi_{tag}"] = f"{100*P[m0]['p95']:.2f}"
    M["presN"] = P["931.494"]["n"]; M["presRelMed"] = f"{100*P['rel_change_938.9']['median']:.0f}"; M["presRelMax"] = f"{100*P['rel_change_938.9']['max_abs']:.0f}"
    M["presRelMed_d"] = f"{100*float(np.median([abs((r['f_931.192'] - r['f_931.494']) / (r['f_931.494'] - r['f_base'])) for r in json.load(open('results_dm_prescription.json'))['rows']])):.1f}"; M["presRelMax_d"] = f"{100*P['rel_change_931.192']['max_abs']:.1f}"
if os.path.exists("wide_prior_results/results_dm_prescription.json"):
    Wr = json.load(open("wide_prior_results/results_dm_prescription.json"))["rows"]
    def _rel(r): return abs((r["f_938.9"] - r["f_931.494"]) / (r["f_931.494"] - r["f_base"]))
    ok = [r for r in Wr if r["f_938.9"] == r["f_938.9"]]; out_ = [r for r in ok if r["mchi"] > 938.783]; in_ = [r for r in ok if r["mchi"] <= 938.783]
    M["presWideN"] = len(ok); M["presWideOutN"] = len(out_)
    M["presWideOutMax"] = f"{100*max(_rel(r) for r in out_):.0f}"; M["presWideOutMed"] = f"{100*float(__import__('numpy').median([_rel(r) for r in out_])):.0f}"
    M["presWideInMax"] = f"{100*max(_rel(r) for r in in_):.0f}"
if os.path.exists("results_prior_width.json"):
    W = json.load(open("results_prior_width.json")); rows = []
    for sc in ("noise-free", "s=0.25", "s=1"):
        for w in (1.0, 0.9, 0.8, 0.7):
            r = [x for x in W if x["scenario"] == sc and x["width"] == w]
            if not r: continue
            r = r[0]; best = max(r["LogReg"], r["RandForest"])
            rows.append(f"{sc.replace('s=', '$s=$')} & {w:g} & {r['n_nuc']}+{r['n_dm']} & {100*r['LogReg']:.0f} & {100*r['RandForest']:.0f} \\\\")
            M[f"pw_{sc.replace('=', '').replace('.', 'p').replace('-', '')}_{int(w*100)}"] = f"{100*best:.0f}"
    open("tables/tab_prior_width.tex", "w").write("\n".join(rows).rstrip()[:-2])
json.dump(M, open("macros_robust2.json", "w"), indent=1); print(M)

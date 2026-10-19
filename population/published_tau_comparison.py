"""Comparison of the computed damping times at 1.4 Msun with the universal relation of Lioutas & Stergioulas (2018, Gen. Rel. Grav. 50, 12),
their Eq. (20) (fitted to full-GR damping times of nuclear EoS for M/R > 0.1, G = c = 1):  M/tau = 0.112 x^4 - 0.53 x^5 + 0.628 x^6,  x = M/R.
Also reports Andersson & Kokkotas (1998) in the form of their Eq. (18): M/tau = 0.0862 x^4 - 0.2672 x^5."""
import json, numpy as np
import pop_data_utils as U

MSUN_KM = 1.476625061; C_KMS = 299792.458
NAMES = {0: "nuc", 1: "str", 2: "dm", 3: "hyp"}


def rel(M, R_km, tau_s, form="LS18"):
    x = M * MSUN_KM / R_km
    fit = 0.112 * x ** 4 - 0.53 * x ** 5 + 0.628 * x ** 6 if form == "LS18" else 0.0862 * x ** 4 - 0.2672 * x ** 5
    tau_fit = M * MSUN_KM / (fit * C_KMS)                 # seconds
    return tau_s / tau_fit - 1, x


out = {"six": [], "pop": {}}
V = json.load(open("results_validation.json"))
for e in V["six"]:
    d, x = rel(1.4, e["R"], e["tau"]); d2, _ = rel(1.4, e["R"], e["tau"], "AK98")
    out["six"].append(dict(eos=e["eos"], C=x, d_ls18=d, d_ak98=d2))
for tag, path, npc in (("A", "pop_data", 90), ("B", "pop_data_B", 48)):
    U.N_PER_CLASS = npc; recs = U.load_all(path); cls, mg, obs, lam = U.tables(recs); j = int(np.argmin(abs(mg - 1.4)))
    d = {}
    for k, nm in NAMES.items():
        sel = np.where(cls == k)[0]
        vals = np.array([rel(1.4, obs[i, j, 1], obs[i, j, 3])[0] for i in sel])
        v2 = np.array([rel(1.4, obs[i, j, 1], obs[i, j, 3], "AK98")[0] for i in sel])
        d[nm] = dict(n=len(sel), med=float(np.median(vals)), p5=float(np.percentile(vals, 5)), p95=float(np.percentile(vals, 95)), med_ak98=float(np.median(v2)))
    out["pop"][tag] = d
json.dump(out, open("results_published_tau.json", "w"), indent=1)
M = {}
ds = [e["d_ls18"] for e in out["six"]]; M["pubSixMin"] = f"{100*min(ds):.1f}"; M["pubSixMax"] = f"{100*max(ds):.1f}"
M["pubSixCmin"] = f"{min(e['C'] for e in out['six']):.3f}"; M["pubSixCmax"] = f"{max(e['C'] for e in out['six']):.3f}"
ds2 = [e["d_ak98"] for e in out["six"]]; M["pubAKMin"] = f"{100*min(ds2):.0f}"; M["pubAKMax"] = f"{100*max(ds2):.0f}"
for tag in ("A", "B"):
    for nm, v in out["pop"][tag].items():
        M[f"pub{tag}_{nm}"] = f"{100*v['med']:+.0f}"; M[f"pub{tag}_{nm}_p5"] = f"{100*v['p5']:+.0f}"; M[f"pub{tag}_{nm}_p95"] = f"{100*v['p95']:+.0f}"
json.dump(M, open("macros_published.json", "w"), indent=1); print(json.dumps(out["six"], indent=1)); print(M)

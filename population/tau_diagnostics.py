"""Is the offset of the computed damping times from the Lioutas & Stergioulas (2018) relation caused by the crust treatment?
tau at 1.4 Msun for the six reference EoS with the calibrated crust pressure scale and with it changed by +-10%."""
import json, numpy as np
from families import *
import eos as E
from convergence_tests import f14
from test_tov import REF
from published_tau_comparison import rel

P0 = 1.1354474
out = {}
for name, (a, b, c, d, R, Lam, fref) in REF.items():
    row = {}
    for tag, ps in (("lo", 0.9 * P0), ("cal", P0), ("hi", 1.1 * P0)):
        E.CRUST["p_scale"] = ps
        base = read_pp(a, b, c, d); f, tau = f14(base, fref)
        s = solve_star(base, np.exp(pc_for_mass(base, 1.4, max_mass(base)[1])))
        row[tag] = dict(f=f, tau=tau, R=s["R"], d_ls18=rel(1.4, s["R"], tau)[0])
    E.CRUST["p_scale"] = P0
    out[name] = row
    print(name, {t: (round(v["tau"], 5), round(100 * v["d_ls18"], 2)) for t, v in row.items()}, flush=True)
dt = [abs(r["hi"]["tau"] / r["lo"]["tau"] - 1) / 0.2 for r in out.values()]
M = {"tauCrustMax": f"{100*max(abs(r['hi']['tau']/r['cal']['tau']-1) for r in out.values()):.2f}", "tauCrustMaxLo": f"{100*max(abs(r['lo']['tau']/r['cal']['tau']-1) for r in out.values()):.2f}"}
json.dump(dict(rows=out, macros=M), open("results_tau_crust.json", "w"), indent=1); json.dump(M, open("macros_taucrust.json", "w")); print(M)

# equivalent radius shift: the relation M/tau ~ x^n locally (n = d ln(M/tau)/d ln x), so an offset d in tau corresponds to a radius change d/n
P = json.load(open("results_published_tau.json"))["six"]
slope = lambda x: (4 * .112 - 5 * .53 * x + 6 * .628 * x * x) / (.112 - .53 * x + .628 * x * x)
eq = [e["d_ls18"] / slope(e["C"]) for e in P]
M["tauSlope"] = f"{slope(0.17):.1f}"; M["tauEqRmin"] = f"{100*min(eq):.1f}"; M["tauEqRmax"] = f"{100*max(eq):.1f}"
json.dump(M, open("macros_taucrust.json", "w")); print(M)

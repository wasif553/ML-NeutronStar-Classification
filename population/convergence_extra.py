"""Numerical-error budget for self-bound (finite surface density) stars: MIT bag (CSS) and a pQCD-form quark-matter EoS."""
import json, numpy as np
from joblib import Parallel, delayed
from families import *
import families_b as FB
import convergence_tests as CT

variants = {
    "default": {}, "interior rtol 1e-8": dict(rtol_int=1e-8), "interior rtol 1e-12": dict(rtol_int=1e-12),
    "exterior rtol 1e-13": dict(rtol_ext=1e-13), "TOV rtol 1e-12": dict(rtol_tov=1e-12),
    "surface cut 1e-4": dict(cut=1e-4), "surface cut 1e-6": dict(cut=1e-6),
    "centre r0 x0.5": dict(r0f=1e-3), "centre r0 x2.5": dict(r0f=5e-3), "r_max 1500 km": dict(rmax=1500.0), "r_max 8000 km": dict(rmax=8000.0),
}


def case(label, eos, guess, **kw):
    f, tau = CT.f14(eos, guess, **kw)
    return label, f, tau


if __name__ == "__main__":
    stars = {"bag B^1/4=145 MeV (c_s^2=1/3)": (CSSEOS(145.0, 1 / 3), 1755.0),
             "pQCD-form a4=0.8, sqrt(a2)=100 MeV, B^1/4=150 MeV": (FB.build(dict(kind="pqcd", a4=0.8, sa2=100.0, B14=150.0)), 1750.0)}
    out = {}
    for name, (e, g) in stars.items():
        try:
            res = Parallel(n_jobs=11)(delayed(case)(k, e, g, **kw) for k, kw in variants.items())
        except Exception as ex:
            print(name, "failed", ex); continue
        d0 = [r for r in res if r[0] == "default"][0]
        out[name] = {k: dict(f=f, tau=t, df_rel=f / d0[1] - 1, dtau_rel=t / d0[2] - 1) for k, f, t in res}
        print(name, "f(default)=%.2f Hz" % d0[1], "max |df/f|=%.1e  max |dtau/tau|=%.1e" % (max(abs(v["df_rel"]) for v in out[name].values()), max(abs(v["dtau_rel"]) for v in out[name].values())))
    json.dump(out, open("results_convergence_strange.json", "w"), indent=1)

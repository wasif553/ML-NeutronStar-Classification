"""Numerical-error budget of the f-mode solver, compared with the 0.56% median dark-matter frequency shift."""
import json, numpy as np
from scipy.optimize import brentq
from joblib import Parallel, delayed
from families import *
import fmode2 as F, tov, eos as E
from fmode import Background
from test_tov import REF


def f14(eos, guess, M=1.4, cut=1e-5, r0f=2e-3, rmax=None, rtol_int=1e-10, rtol_ext=1e-11, rtol_tov=1e-10):
    F.RTOL_INT, F.RTOL_EXT, tov.RTOL_TOV = rtol_int, rtol_ext, rtol_tov
    lm = max_mass(eos)[1]; lpc = pc_for_mass(eos, M, lm); bg = Background(eos, np.exp(lpc))
    k = 2 * np.pi / F.C_KMS
    g = lambda w: F.ingoing_amplitude(bg, w, cut=cut, rmax=rmax)[0]
    # r0 enters through interior(); patch via the default argument
    orig = F.interior
    F.interior = lambda bg_, om, r0=None, cut=cut: orig(bg_, om, r0=r0f * bg_.R, cut=cut)
    try:
        om = k * guess + 1j / (0.2 * F.C_KMS); w0, w1 = om, om * (1 + 1e-4); g0, g1 = g(w0), g(w1)
        for _ in range(40):
            w2 = w1 - g1 * (w1 - w0) / (g1 - g0); w0, g0 = w1, g1; w1, g1 = w2, g(w2)
            if abs(w1 - w0) < 1e-9 * abs(w1): break
    finally:
        F.interior = orig; F.RTOL_INT, F.RTOL_EXT, tov.RTOL_TOV = 1e-10, 1e-11, 1e-10
    return w1.real / k, 1 / (w1.imag * F.C_KMS)


def case(label, eos, guess, **kw):
    f, tau = f14(eos, guess, **kw)
    return label, f, tau


if __name__ == "__main__":
    a, b, c, d, R, Lam, fref = REF["SLy"]
    base = read_pp(a, b, c, d)
    eos_dm = dm_admixed(base, 100.0, 938.0)
    variants = {
        "default": {}, "interior rtol 1e-8": dict(rtol_int=1e-8), "interior rtol 1e-12": dict(rtol_int=1e-12),
        "exterior rtol 1e-9": dict(rtol_ext=1e-9), "exterior rtol 1e-13": dict(rtol_ext=1e-13),
        "TOV rtol 1e-12": dict(rtol_tov=1e-12), "surface cut 1e-4": dict(cut=1e-4), "surface cut 1e-6": dict(cut=1e-6),
        "centre r0 x0.5": dict(r0f=1e-3), "centre r0 x2.5": dict(r0f=5e-3), "r_max 1500 km": dict(rmax=1500.0), "r_max 8000 km": dict(rmax=8000.0),
    }
    out = {"variants": {}}
    for name, eos_, guess in (("SLy", base, 1935.0), ("SLy+DM(G=100)", eos_dm, 1960.0)):
        res = Parallel(n_jobs=12)(delayed(case)(k, eos_, guess, **kw) for k, kw in variants.items())
        d0 = [r for r in res if r[0] == "default"][0]
        out["variants"][name] = {k: dict(f=f, tau=t, df_rel=(f / d0[1] - 1), dtau_rel=(t / d0[2] - 1)) for k, f, t in res}
    # crust sensitivity: absolute pressure scale +-10 %, effect on f of the nuclear EoS and on the DM shift
    crust = {}
    for ps in (1.1354474 * 0.9, 1.1354474, 1.1354474 * 1.1):
        E.CRUST["p_scale"] = ps
        b2 = read_pp(a, b, c, d); e2 = dm_admixed(b2, 100.0, 938.0)
        r = Parallel(n_jobs=2)(delayed(case)(n, e_, g_) for n, e_, g_ in (("nuc", b2, 1935.0), ("dm", e2, 1960.0)))
        crust[f"{ps:.4f}"] = dict(f_nuc=r[0][1], f_dm=r[1][1], shift_rel=r[1][1] / r[0][1] - 1)
    E.CRUST["p_scale"] = 1.1354474
    out["crust"] = crust
    json.dump(out, open("results_convergence.json", "w"), indent=1)
    for name, v in out["variants"].items():
        print(name)
        for k, x in v.items(): print("  %-22s f=%.3f  df/f=%+.2e  dtau/tau=%+.2e" % (k, x["f"], x["df_rel"], x["dtau_rel"]))
    print(json.dumps(crust, indent=1))

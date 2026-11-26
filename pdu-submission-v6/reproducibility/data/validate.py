"""Solver validation against the independent reference values for six EoS fits at 1.4 Msun, plus consistency checks."""
import json, numpy as np
from scipy.optimize import brentq
from joblib import Parallel, delayed
from families import *
import fmode2 as F
from fmode import Background
from test_tov import REF


def one(nm):
    a, b, c, d, R, Lam, fref = REF[nm]
    eos = read_pp(a, b, c, d)
    lpc = brentq(lambda l: solve_star(eos, np.exp(l), tidal=False)["M"] - 1.4, np.log(5e-6), np.log(3e-3), xtol=1e-13)
    s = solve_star(eos, np.exp(lpc)); bg = Background(eos, np.exp(lpc))
    f, tau, w = F.find_mode(bg, fref * 0.998, 0.2)
    return dict(eos=nm, R=s["R"], R_ref=R, Lam=s["Lambda"], Lam_ref=Lam, f=f, f_ref=fref, tau=tau)


if __name__ == "__main__":
    res = Parallel(n_jobs=6)(delayed(one)(nm) for nm in REF)
    out = dict(six=res, p_scale=CRUST["p_scale"])
    # tabulated-vs-analytic consistency and dark-matter shifts (reference: 25.845 Hz for G = 100 fm^2 at 1.4 Msun, SLy)
    a, b, c, d, R, Lam, fref = REF["SLy"]
    base = read_pp(a, b, c, d)

    def mode14(e, guess):
        lm = max_mass(e)[1]; l14 = pc_for_mass(e, 1.4, lm); bg = Background(e, np.exp(l14))
        return F.find_mode(bg, guess, 0.2)[0]
    f0 = mode14(base, 1935.0)
    out["tab_vs_analytic_f"] = mode14(dm_admixed(base, 100.0, 1e5), 1935.0)
    out["f0"] = f0
    out["dm_shift_G100"] = mode14(dm_admixed(base, 100.0, 938.0), 1960.0) - f0
    out["dm_shift_G200"] = mode14(dm_admixed(base, 200.0, 938.0), 1950.0) - f0
    out["dm_shift_ref_G100"] = 1960.905 - 1935.060
    json.dump(out, open("results_validation.json", "w"), indent=1)
    print(json.dumps(out, indent=1))

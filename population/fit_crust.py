"""Calibrate the single crust parameter p_scale on (R, Lambda) of six reference EoS at 1.4 Msun; f is held out."""
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from test_tov import REF, star14
from eos import read_pp


def one(nm, ps):
    a, b, c, d, R, L, f = REF[nm]
    s = star14(read_pp(a, b, c, d, p_scale=ps))
    return s["R"] - R, 100 * (s["Lambda"] / L - 1)


def cost(ps):
    out = np.array(Parallel(n_jobs=6)(delayed(one)(nm, ps) for nm in REF))
    return float(np.sum((out[:, 0] / 0.05) ** 2 + (out[:, 1] / 2.0) ** 2))


if __name__ == "__main__":
    r = minimize_scalar(cost, bounds=(0.8, 1.6), method="bounded", options=dict(xatol=2e-3))
    print("p_scale =", r.x, "cost", r.fun)
    out = np.array(Parallel(n_jobs=6)(delayed(one)(nm, r.x) for nm in REF))
    for nm, o in zip(REF, out): print("  %-5s dR=%+.3f km  dLambda=%+.2f%%" % (nm, o[0], o[1]))

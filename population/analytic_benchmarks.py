"""Analytic benchmarks with a finite surface density (self-bound-like surface): the homogeneous incompressible star.

Newtonian results: Kelvin f-mode  omega^2 = 2 l (l-1)/(2 l + 1) G M / R^3  (= 0.8 G M / R^3 for l = 2),
tidal Love number k2 = 3/4.   The GR solver is run at small compactness C = M/R with a nearly incompressible, constant-density EoS
(eps = eps0 + p/cs2 with cs2 = 10^4); the deviations must vanish linearly as C -> 0.
"""
import json, numpy as np
from scipy.optimize import brentq
from families import CSSEOS
from tov import solve_star
import fmode2 as F
from fmode import Background

eos = CSSEOS(145.0, 1e4)
K = 2 * np.pi / F.C_KMS
out = []
for target_C in (2e-3, 5e-3, 1e-2, 2e-2, 4e-2):
    lpc = brentq(lambda l: solve_star(eos, np.exp(l), tidal=False)["Mkm"] / solve_star(eos, np.exp(l), tidal=False)["R"] - target_C,
                 np.log(1e-9), np.log(1e-3), xtol=1e-12)
    s = solve_star(eos, np.exp(lpc)); bg = Background(eos, np.exp(lpc))
    C = s["Mkm"] / s["R"]; om_K = np.sqrt(0.8 * s["Mkm"] / s["R"] ** 3)          # km^-1
    f_K = om_K / K
    f, tau, w = F.find_mode(bg, f_K * 0.99, 1e6)
    s0 = solve_star(eos, np.exp(lpc), surface_correction=False)
    tau_N = 19.53125 * s["R"] ** 4 / s["Mkm"] ** 3 / F.C_KMS          # Newtonian quadrupole-formula damping time of the Kelvin mode (uniform density)
    out.append(dict(tau=tau, tau_newton=tau_N, dtau_rel=tau / tau_N - 1, k2_uncorrected=s0["k2"], C=C, f=f, f_Kelvin=f_K, df_rel=f / f_K - 1, k2=s["k2"], k2_newton=0.75, dk2_rel=s["k2"] / 0.75 - 1))
    print("C=%.4f  tau/tau_N-1=%+.4f  f/f_Kelvin-1=%+.4f  k2=%.4f (Newtonian 0.75, %+.4f); without the finite-surface-density correction k2=%.4f" % (C, tau / tau_N - 1, f / f_K - 1, s["k2"], s["k2"] / 0.75 - 1, s0["k2"]), flush=True)
# linear extrapolation to C -> 0
C_ = np.array([o["C"] for o in out]); d = np.array([o["df_rel"] for o in out]); dk = np.array([o["dk2_rel"] for o in out])
pf = np.polyfit(C_, d, 1); pk = np.polyfit(C_, dk, 1); dt = np.array([o['dtau_rel'] for o in out]); pt = np.polyfit(C_, dt, 1)
res = dict(tau_intercept=float(pt[1]), tau_slope=float(pt[0]), points=out, f_intercept=float(pf[1]), f_slope=float(pf[0]), k2_intercept=float(pk[1]), k2_slope=float(pk[0]))
print("extrapolated C->0: tau/tau_N - 1 = %+.2e, slope %+.2f ;  f/f_Kelvin - 1 = %+.2e ;  k2/0.75 - 1 = %+.2e" % (pt[1], pt[0], pf[1], pk[1]))
json.dump(res, open("results_analytic.json", "w"), indent=1)

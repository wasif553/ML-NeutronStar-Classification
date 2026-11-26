"""Benchmark of the exterior complex-frequency machinery (outgoing-wave condition, damping time) on a case with a published answer:
the fundamental l=2 quasi-normal mode of a Schwarzschild black hole, M*omega = 0.37367 -/+ 0.08896 i (e^{-/+ i omega t} convention).
The Zerilli equation is integrated from the horizon (purely ingoing) to large radius and the ingoing amplitude at infinity is
extracted with the same asymptotic projection used for stars."""
import json, numpy as np
from scipy.integrate import solve_ivp
from fmode import zerilli_V, asym_Z, tortoise


def a_in(om, M=1.0, delta=1e-6, rmax=400.0):
    r0 = 2 * M * (1 + delta); f0 = 1 - 2 * M / r0
    Z0 = np.exp(1j * om * (tortoise(r0, M) - tortoise(r0, M)))          # = 1 (normalisation)
    Zr0 = 1j * om / f0 * Z0                                              # ingoing at the horizon: Z ~ exp(+i om r*) for exp(+i om t)
    om2 = om * om

    def fz(rr, s):
        f = 1 - 2 * M / rr
        return [s[1], (zerilli_V(rr, M) - om2) * s[0] / f ** 2 - (2 * M / rr ** 2) / f * s[1]]
    sol = solve_ivp(fz, (r0, rmax), [Z0, Zr0], method="DOP853", rtol=1e-12, atol=1e-30)
    Zn, Zrn = sol.y[0, -1], sol.y[1, -1]
    zo, dzo = asym_Z(rmax, M, om, +1, r0); zi, dzi = asym_Z(rmax, M, om, -1, r0)
    A = np.linalg.solve(np.array([[zi, zo], [dzi, dzo]], complex), np.array([Zn, Zrn], complex))
    return A[0]


def find(om0):
    w0, w1 = om0, om0 * (1 + 1e-3); g0, g1 = a_in(w0), a_in(w1)
    for _ in range(40):
        w2 = w1 - g1 * (w1 - w0) / (g1 - g0); w0, g0 = w1, g1; w1, g1 = w2, a_in(w2)
        if abs(w1 - w0) < 1e-10 * abs(w1): break
    return w1


if __name__ == "__main__":
    ref = 0.37367 + 0.08896j        # e^{+i omega t} convention (damped for Im > 0), M = 1 (Leaver 1985)
    out = {}
    for rmax in (40.0, 60.0, 80.0):  # larger radii lose the ingoing amplitude to round-off because the outgoing solution grows as exp(Im(w) r*)
        a_in.__defaults__ = (1.0, 1e-6, rmax)
        w = find(0.37 + 0.09j)
        out[f"rmax={rmax:g}"] = dict(re=w.real, im=w.imag, rel_err_re=w.real / ref.real - 1, rel_err_im=w.imag / ref.imag - 1)
        print(rmax, w, "rel. errors: Re %.1e  Im %.1e" % (w.real / ref.real - 1, w.imag / ref.imag - 1))
    json.dump(dict(reference=[ref.real, ref.imag], runs=out), open("results_bh_qnm.json", "w"), indent=1)

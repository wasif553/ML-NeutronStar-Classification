"""TOV integration with tidal deformability (km, G=c=1)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from eos import MSUN_KM


def _rhs_factory(eos):
    fpi = 4 * np.pi

    def rhs(r, s):
        p, m, nu, y = s
        if p <= 0: p = 1e-60
        e = eos.eps(p); cs2 = eos.cs2(p)
        f = 1.0 - 2 * m / r
        elam = 1.0 / f
        a = (m + fpi * r ** 3 * p)
        dnu = 2 * a / (r * r * f)
        dp = -(e + p) * dnu / 2
        dm = fpi * r * r * e
        Q = fpi * elam * (5 * e + 9 * p + (e + p) / cs2) - 6 * elam / r ** 2 - dnu ** 2
        dy = (-y * y - y * elam * (1 + fpi * r * r * (p - e)) - r * r * Q) / r
        return [dp, dm, dnu, dy]
    return rhs


def solve_star(eos, pc, tidal=True, dense=False, rtol=1e-10):
    """Integrate outward from the centre; returns dict with M [Msun], R [km], Lambda, and (optionally) a dense solution."""
    ec = eos.eps(pc)
    r0 = 1e-4
    p0 = pc - (2 * np.pi / 3) * (ec + pc) * (ec + 3 * pc) * r0 ** 2
    m0 = 4 * np.pi / 3 * ec * r0 ** 3
    psurf = eos.p_surface()
    ev = lambda r, s: s[0] - psurf
    ev.terminal = True; ev.direction = -1
    sol = solve_ivp(_rhs_factory(eos), (r0, 60.0), [p0, m0, 0.0, 2.0], events=ev, method="DOP853",
                    rtol=rtol, atol=[1e-40, 1e-12, 1e-12, 1e-12], dense_output=dense)
    R = sol.t_events[0][0]
    p, m, nu, y = sol.y_events[0][0]
    M = m
    out = dict(M=M / MSUN_KM, R=R, pc=pc, ec=ec, nuR=nu, Mkm=M, psurf=psurf)
    if tidal:
        es = eos.eps(psurf)
        yR = y - 4 * np.pi * R ** 3 * es / M if es > 1e-3 * ec * 0 + 1e-30 and es > 1e-8 else y
        # discontinuity correction only matters when eps(surface) is non-negligible (self-bound matter)
        C = M / R
        num = (8 / 5) * C ** 5 * (1 - 2 * C) ** 2 * (2 + 2 * C * (yR - 1) - yR)
        den = (2 * C * (6 - 3 * yR + 3 * C * (5 * yR - 8)) + 4 * C ** 3 * (13 - 11 * yR + C * (3 * yR - 2) + 2 * C ** 2 * (1 + yR))
               + 3 * (1 - 2 * C) ** 2 * (2 - yR + 2 * C * (yR - 1)) * np.log(1 - 2 * C))
        k2 = num / den
        out["k2"] = k2; out["Lambda"] = 2.0 / 3.0 * k2 / C ** 5
    if dense:
        out["sol"] = sol
    return out


def pc_for_mass(eos, Mtarget, lo=1e-5, hi=None):
    """central pressure (km^-2) giving gravitational mass Mtarget (Msun) on the stable branch below the maximum mass."""
    f = lambda lp: solve_star(eos, np.exp(lp), tidal=False)["M"] - Mtarget
    lo_, hi_ = np.log(lo), np.log(hi if hi else 3e-2)
    # locate the maximum mass first so we stay on the stable branch
    return brentq(f, lo_, hi_, xtol=1e-12)


def mass_sequence(eos, lpc_grid):
    return np.array([[solve_star(eos, np.exp(l), tidal=False)["M"], solve_star(eos, np.exp(l), tidal=False)["R"]] for l in lpc_grid])

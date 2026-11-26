"""Alternative EoS population ("family B"): different functional forms and priors from families.py, used for robustness tests.

  nucleonic:  single-polytrope outer core up to 1.3 n_sat, then a piecewise-linear sound-speed profile cs2(eps) (model-agnostic form)
  hyperonic:  same, with a pronounced dip of cs2 (softening) at 2.5 eps_x, then stiffening
  strange:    self-bound quark matter, P(mu) = 3 a4 mu^4/(4 pi^2) - 3 a2 mu^2/(4 pi^2) - B (pQCD-motivated, Alford et al. form)
  dark-matter admixed: nucleonic (sound-speed form) + the same neutron-decay dark Fermi gas in chemical equilibrium
"""
import numpy as np
from scipy.optimize import brentq
from families import *
import families as FA

RHO_X = 10 ** 14.55            # transition density (about 1.3 n_sat)
KNOT_FACTORS = np.array([1.0, 1.5, 2.5, 4.0, 7.0])


def _table(logP1, G1, knots):
    """returns arrays (P, E, N, CS2) in MeV/fm^3, MeV/fm^3, fm^-3, dimensionless for a sound-speed-profile EoS"""
    pp = read_pp(logP1, G1, G1, G1)
    rho = np.logspace(6, np.log10(RHO_X), 1200)
    p_cgs = np.array([pp.p_of_rho(r_) for r_ in rho]) / P2G
    e_cgs = np.array([pp.rho_eps_cgs(p_)[1] for p_ in p_cgs])
    cs2_lo = np.array([pp.cs2(p_ * P2G) for p_ in p_cgs])
    P_lo, E_lo, N_lo = p_cgs * ERG_CC_TO_MEV_FM3, e_cgs * ERG_CC_TO_MEV_FM3, rho / M0_G / 1e39
    ex, px, nx = E_lo[-1], P_lo[-1], N_lo[-1]
    e = ex * np.logspace(0, np.log10(80.0), 1500)[1:]
    ek = ex * KNOT_FACTORS
    cs2 = np.interp(np.log(e), np.log(ek), knots)
    P_hi = np.empty_like(e); N_hi = np.empty_like(e)
    p_prev, e_prev, c_prev, n_prev = px, ex, cs2_lo[-1], nx
    for i in range(len(e)):
        dp = 0.5 * (c_prev + cs2[i]) * (e[i] - e_prev)
        p_new = p_prev + dp
        n_new = n_prev * np.exp(0.5 * (1 / (e_prev + p_prev) + 1 / (e[i] + p_new)) * (e[i] - e_prev))
        P_hi[i], N_hi[i] = p_new, n_new
        p_prev, e_prev, c_prev, n_prev = p_new, e[i], cs2[i], n_new
    return (np.concatenate([P_lo, P_hi]), np.concatenate([E_lo, e]), np.concatenate([N_lo, N_hi]), np.concatenate([cs2_lo, cs2]))


def _to_tab(P, E, CS2):
    return TabEOS(P * MEV_FM3_GEO, E * MEV_FM3_GEO, np.clip(CS2, 1e-4, 1.0))


def _add_dark(P, E, N, G, mchi):
    mu = (E + P) / N
    nchi = np.zeros_like(P); P2, E2 = P.copy(), E.copy()
    for i, m in enumerate(mu):
        if m <= mchi: continue
        f = lambda k: np.sqrt(k * k + mchi ** 2) + HBARC * G * (k / HBARC) ** 3 / (3 * np.pi ** 2) - m
        k = brentq(f, 1e-9, np.sqrt(max(m * m - mchi ** 2, 1e-12)) + 1e-9)
        n = (k / HBARC) ** 3 / (3 * np.pi ** 2)
        ef, pf = FA._fermi(k, mchi); ef /= HBARC ** 3; pf /= HBARC ** 3
        E2[i] += ef + 0.5 * HBARC * G * n * n; P2[i] += pf + 0.5 * HBARC * G * n * n; nchi[i] = n
    return P2, E2


def sample_params(cls, rng):
    if cls in (0, 2):
        k = [rng.uniform(0.1, 0.6)] + list(rng.uniform(0.1, 1.0, 4))
        d = dict(kind="csk", logP1=rng.uniform(34.15, 34.55), G1=rng.uniform(2.6, 3.6), knots=[float(x) for x in k])
        if cls == 2: d.update(kind="csk_dm", G=rng.uniform(50, 500), mchi=rng.uniform(937.9, 938.78))
        return d
    if cls == 3:
        k = [rng.uniform(0.15, 0.6), rng.uniform(0.15, 0.6), rng.uniform(0.03, 0.30), rng.uniform(0.5, 1.0), rng.uniform(0.5, 1.0)]
        return dict(kind="csk", logP1=rng.uniform(34.15, 34.55), G1=rng.uniform(2.6, 3.6), knots=[float(x) for x in k])
    if cls == 1:
        return dict(kind="pqcd", a4=rng.uniform(0.55, 1.0), sa2=rng.uniform(0, 150), B14=rng.uniform(135, 165))
    raise ValueError(cls)


def _pqcd(par):
    a4, a2, B = par["a4"], par["sa2"] ** 2, par["B14"] ** 4
    P = lambda m: (3 * a4 / (4 * np.pi ** 2)) * m ** 4 - (3 * a2 / (4 * np.pi ** 2)) * m ** 2 - B
    mu0 = brentq(P, np.sqrt(a2 / (2 * a4)) * 1.0001 + 1e-6, 3000.0)
    mu = mu0 * (1 + np.logspace(-9, np.log10(0.9), 1500))
    p = P(mu); n = (3 * a4 / np.pi ** 2) * mu ** 3 - (3 * a2 / (2 * np.pi ** 2)) * mu
    e = mu * n - p
    return p / HBARC ** 3, e / HBARC ** 3


def build(par):
    if par["kind"] == "pqcd":
        p, e = _pqcd(par)
        de = np.gradient(e); cs2 = np.clip(np.gradient(p) / de, 1e-3, 1.0)
        o = np.argsort(p); p, e, cs2 = p[o], e[o], cs2[o]
        keep = p > 1e-9
        return TabEOS(p[keep] * MEV_FM3_GEO, e[keep] * MEV_FM3_GEO, cs2[keep])
    P, E, N, CS2 = _table(par["logP1"], par["G1"], np.array(par["knots"]))
    if par["kind"] == "csk_dm":
        Pt, Et = _add_dark(P, E, N, par["G"], par["mchi"])
        cs2 = np.clip(np.gradient(Pt) / np.gradient(Et), 1e-4, 5.0)
        return _to_tab(Pt, Et, np.minimum(cs2, 5.0))
    return _to_tab(P, E, CS2)


screen = FA.screen

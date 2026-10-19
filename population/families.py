"""EoS families for the population study (four composition scenarios, each a *population* of EoS)."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from eos import *
from tov import solve_star

HBARC = 197.3269804          # MeV fm
M0_G = 1.66053907e-24        # reference baryon mass (u), g
ERG_CC_TO_MEV_FM3 = 6.241509074e-34


class CSSEOS:
    """Constant-speed-of-sound EoS  eps = eps0 + p/cs2  (cs2 = 1/3 and eps0 = 4B is the MIT bag model)."""

    def __init__(self, B14_mev, cs2):
        B = B14_mev ** 4 / HBARC ** 3                       # MeV/fm^3
        self.e0 = 4 * B * MEV_FM3_GEO
        self.cs2c = cs2
        self.pmin = 1e-14

    def eps(self, p): return self.e0 + p / self.cs2c
    def cs2(self, p): return self.cs2c
    def p_surface(self): return self.pmin


class PPCSSEOS:
    """Piecewise-polytropic EoS up to a transition pressure, then a constant-speed-of-sound segment (causal stiffening)."""

    def __init__(self, pp, rho_t, cs2c):
        self.pp, self.cs2c = pp, cs2c
        self.p_t = pp.p_of_rho(rho_t)
        self.e_t = pp.eps(self.p_t)
        self.pmin = pp.pmin

    def eps(self, p): return self.pp.eps(p) if p < self.p_t else self.e_t + (p - self.p_t) / self.cs2c
    def cs2(self, p): return self.pp.cs2(p) if p < self.p_t else self.cs2c
    def p_surface(self): return self.pp.p_surface()


def _fermi(k, m):
    """free Fermi gas (natural units MeV^4): energy density (incl. rest mass) and pressure"""
    E = np.sqrt(k * k + m * m)
    L = np.log((k + E) / m)
    eps = (k * (2 * k * k + m * m) * E - m ** 4 * L) / (8 * np.pi ** 2)
    p = (k * (2 * k * k - 3 * m * m) * E + 3 * m ** 4 * L) / (24 * np.pi ** 2)
    return eps, p


def dm_admixed(base, G, mchi, rho_max=3.2e15, npts=3000, m0_mev=931.494):
    """Barotropic effective EoS of a baryon PP EoS plus a self-interacting dark Fermi gas in chemical equilibrium
    (mu_chi = mu_N): total p(mu) = p_N + p_chi, eps = eps_N + eps_chi (the 'effective one-fluid barotrope')."""
    rho = np.logspace(6, np.log10(rho_max), npts)
    pN = np.array([base.p_of_rho(r_) for r_ in rho]) / P2G                  # dyn/cm^2
    eN = np.array([base.rho_eps_cgs(p_)[1] for p_ in pN])                   # erg/cm^3
    nB = rho / (M0_G * m0_mev / 931.494) / 1e39                             # fm^-3 (m0_mev: baryon mass assigned to the fitted barotrope)
    muN = (eN + pN) * ERG_CC_TO_MEV_FM3 / nB                                # MeV
    p_tot = pN * ERG_CC_TO_MEV_FM3
    e_tot = eN * ERG_CC_TO_MEV_FM3
    nchi = np.zeros_like(rho)
    seg = np.array([base._seg(p_) for p_ in pN]); gam = np.array([base.Gam[j] for j in seg])
    dpN = gam * pN * ERG_CC_TO_MEV_FM3                  # dp_N / d ln n_B at fixed composition (MeV/fm^3)
    dpchi = np.zeros_like(rho); ech = np.zeros_like(rho)  # frozen-composition response and (eps+p) of the dark component
    for i, mu in enumerate(muN):
        if mu <= mchi: continue
        f = lambda k: np.sqrt(k * k + mchi ** 2) + HBARC * G * (k / HBARC) ** 3 / (3 * np.pi ** 2) - mu
        k = brentq(f, 1e-9, np.sqrt(max(mu * mu - mchi ** 2, 1e-12)) + 1e-9)
        n = (k / HBARC) ** 3 / (3 * np.pi ** 2)
        eps_f, p_f = _fermi(k, mchi)
        eps_f /= HBARC ** 3; p_f /= HBARC ** 3
        e_tot[i] += eps_f + 0.5 * HBARC * G * n * n
        p_tot[i] += p_f + 0.5 * HBARC * G * n * n
        nchi[i] = n
        Ek = np.sqrt(k * k + mchi ** 2)
        dpchi[i] = n * k * k / (3 * Ek) + HBARC * G * n * n
        ech[i] = (eps_f + p_f) + HBARC * G * n * n
    p_g, e_g = p_tot * MEV_FM3_GEO, e_tot * MEV_FM3_GEO
    de = np.gradient(e_g); dpp = np.gradient(p_g)
    cs2 = np.clip(dpp / de, 1e-4, 5.0)
    eos = TabEOS(p_g, e_g, cs2)
    # frozen-composition (no baryon <-> dark conversion) adiabatic response with a common compression of both components
    cs2f = np.clip((dpN + dpchi) / ((eN + pN) * ERG_CC_TO_MEV_FM3 + ech), 1e-4, 5.0)
    eos._lcf = PchipInterpolator(np.log(p_g), np.log(cs2f))
    eos.cs2_pert = lambda p, eq=eos: float(np.exp(eq._lcf(np.log(max(p, eq.pmin)))))
    eos.dark_fraction_central = (nchi[-1] * mchi) / (e_tot[-1]) if e_tot[-1] > 0 else 0.0
    return eos


def sample_params(cls, rng):
    if cls == 0:       # nucleonic
        return dict(kind="pp", logP1=rng.uniform(34.15, 34.60), G1=rng.uniform(2.4, 3.8), G2=rng.uniform(2.4, 3.9), G3=rng.uniform(2.2, 3.9))
    if cls == 3:       # hyperonic-like: soft segment above ~2 n0, stiffening above 1e15 g/cc
        return dict(kind="ppcss", logP1=rng.uniform(34.20, 34.60), G1=rng.uniform(2.6, 3.8), G2=rng.uniform(1.8, 2.6),
                    log_rho_t=rng.uniform(14.95, 15.25), cs2=rng.uniform(0.5, 1.0))
    if cls == 1:       # strange (self-bound) quark matter
        return dict(kind="css", B14=rng.uniform(130, 160), cs2=rng.uniform(1 / 3, 1.0))
    if cls == 2:       # neutron-decay dark-matter admixed nucleonic matter
        d = sample_params(0, rng)
        d.update(kind="dm", G=rng.uniform(50, 500), mchi=rng.uniform(937.9, 939.2))
        return d
    raise ValueError(cls)


def build(par):
    if par["kind"] == "css": return CSSEOS(par["B14"], par["cs2"])
    if par["kind"] == "ppcss":
        base = read_pp(par["logP1"], par["G1"], par["G2"], par["G2"], rho2=10 ** par["log_rho_t"])
        return PPCSSEOS(base, 10 ** par["log_rho_t"], par["cs2"])
    base = read_pp(par["logP1"], par["G1"], par["G2"], par["G3"])
    if par["kind"] == "pp": return base
    return dm_admixed(base, par["G"], par["mchi"])


def mass_of(eos, lpc): return solve_star(eos, np.exp(lpc), tidal=False)["M"]


def max_mass(eos, lo=np.log(2e-5), hi=np.log(8e-3)):
    r = minimize_scalar(lambda l: -mass_of(eos, l), bounds=(lo, hi), method="bounded", options=dict(xatol=2e-3))
    return -r.fun, r.x


def pc_for_mass(eos, M, lpc_max, lo=np.log(1e-5)):
    return brentq(lambda l: mass_of(eos, l) - M, lo, lpc_max, xtol=1e-12)


def screen(eos, Mmax_req=1.97):
    """Observational priors: M_max >= 1.97, causal, 70 <= Lambda_1.4 <= 800, 10 <= R_1.4 <= 14 km."""
    try:
        Mmax, lpcm = max_mass(eos)
    except Exception:
        return None
    if Mmax < Mmax_req: return dict(ok=False, why="Mmax", Mmax=Mmax)
    pcm = np.exp(lpcm)
    if eos.cs2(pcm) > 1.0: return dict(ok=False, why="causal", Mmax=Mmax)
    try:
        l14 = pc_for_mass(eos, 1.4, lpcm)
        s = solve_star(eos, np.exp(l14))
    except Exception:
        return dict(ok=False, why="solve", Mmax=Mmax)
    if not (70 <= s["Lambda"] <= 800 and 10.0 <= s["R"] <= 14.0): return dict(ok=False, why="LR", Mmax=Mmax)
    return dict(ok=True, Mmax=Mmax, lpcmax=lpcm, R14=s["R"], Lam14=s["Lambda"])

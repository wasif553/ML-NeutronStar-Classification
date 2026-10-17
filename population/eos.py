"""Equation-of-state building blocks (geometric units: lengths in km, p and eps in km^-2, G = c = 1)."""
import numpy as np
from bisect import bisect_right
from scipy.interpolate import PchipInterpolator

G_CGS, C_CGS, MSUN_G = 6.67430e-8, 2.99792458e10, 1.98847e33
P2G = G_CGS / C_CGS ** 4 * 1e10          # erg/cm^3 (or dyn/cm^2) -> km^-2
MSUN_KM = G_CGS * MSUN_G / C_CGS ** 2 / 1e5   # 1.4766 km
MEV_FM3_GEO = 1.602176634e-6 / 1e-39 * P2G   # MeV/fm^3 -> km^-2
ME_G, H_CGS, MU_G = 9.1093837e-28, 6.62607015e-27, 1.66053907e-24
RHO_E = MU_G * 8 * np.pi / 3 * (ME_G * C_CGS / H_CGS) ** 3     # density scale per unit mu_e (g/cc)


def p_chandrasekhar(rho, mue=2.154):
    """Degenerate electron-gas pressure (dyn/cm^2); the outer-crust baseline."""
    x = (rho / (mue * RHO_E)) ** (1 / 3)
    return (np.pi * ME_G ** 4 * C_CGS ** 5 / (3 * H_CGS ** 3)) * (x * (2 * x * x - 3) * np.sqrt(1 + x * x) + 3 * np.arcsinh(x))


class PPEOS:
    """Piecewise-polytropic EoS (Read et al. 2009 form): p = K_i rho^G_i, eps = (1+a_i) rho c^2 + p/(G_i-1)."""

    def __init__(self, rho_b, K, Gam):
        self.rho_b, self.K, self.Gam = list(rho_b), list(K), list(Gam)
        n = len(K)
        self.a = [0.0] * n
        self.p_b = [self.K[0] * self.rho_b[0] ** self.Gam[0]]
        for i in range(1, n):
            pi = self.K[i - 1] * self.rho_b[i] ** self.Gam[i - 1]
            self.a[i] = self.a[i - 1] + pi / (self.rho_b[i] * C_CGS ** 2) * (1 / (self.Gam[i - 1] - 1) - 1 / (self.Gam[i] - 1))
            self.p_b.append(pi)
        self.pmin = self.p_b[0]

    def _seg(self, p):
        return max(0, bisect_right(self.p_b, p) - 1)

    def rho_eps_cgs(self, p):
        i = self._seg(p)
        rho = (p / self.K[i]) ** (1 / self.Gam[i])
        return rho, (1 + self.a[i]) * rho * C_CGS ** 2 + p / (self.Gam[i] - 1)

    def eps(self, p):            # p in km^-2 -> eps in km^-2
        return self.rho_eps_cgs(p / P2G)[1] * P2G

    def cs2(self, p):
        pc = p / P2G
        i = self._seg(pc)
        e = self.rho_eps_cgs(pc)[1]
        return self.Gam[i] * pc / (e + pc)

    def p_of_rho(self, rho):     # g/cc -> km^-2
        i = max(0, bisect_right(self.rho_b, rho) - 1)
        return self.K[i] * rho ** self.Gam[i] * P2G

    def p_surface(self):
        return self.p_of_rho(1e6)


CRUST = dict(p_scale=1.1354474)   # calibrated on R and Lambda of six reference EoS in fit_crust.py


def crust_pieces(p_scale=None):
    """Four-piece SLy-like crust. Indices and boundaries follow the Read et al. (2009) structure
    (Gamma = 1.58425, 1.28733, 0.62223, 1.35692 with boundaries 2.44034e7, 3.78358e11, 2.62789e12 g/cc, ordered by increasing
    density); the absolute pressure is anchored on the degenerate electron-gas pressure at 2.44034e7 g/cc and the
    remaining constants follow from continuity."""
    p_scale = CRUST["p_scale"] if p_scale is None else p_scale
    rb = [0.0, 2.44034e7, 3.78358e11, 2.62789e12]
    Gam = [1.58425, 1.28733, 0.62223, 1.35692]
    K = [p_scale * p_chandrasekhar(rb[1]) / rb[1] ** Gam[0]]
    for i in range(1, 4):
        K.append(K[-1] * rb[i] ** (Gam[i - 1] - Gam[i]))
    return rb, K, Gam


def read_pp(logP1, G1, G2, G3, rho1=10 ** 14.7, rho2=1e15, **crust):
    rb, K, Gam = crust_pieces(**crust)
    K1 = 10 ** logP1 / rho1 ** G1
    rho0 = (K[-1] / K1) ** (1 / (G1 - Gam[-1]))      # crust-core intersection
    if not (rb[-1] < rho0 < rho1): raise ValueError('crust-core intersection outside (rho_mid, rho1)')
    K2 = 10 ** logP1 / rho1 ** G2
    K3 = K2 * rho2 ** G2 / rho2 ** G3
    return PPEOS(rb + [rho0, rho1, rho2], K + [K1, K2, K3], Gam + [G1, G2, G3])


class TabEOS:
    """EoS from arrays (p, eps) in km^-2, ascending in p; eps(0) may be nonzero (self-bound matter)."""

    def __init__(self, p, eps, cs2=None):
        p, eps = np.asarray(p, float), np.asarray(eps, float)
        o = np.argsort(p)
        self.p, self.e = p[o], eps[o]
        self.pmin = self.p[0]
        self._lp = np.log(self.p)
        self._f = PchipInterpolator(self._lp, np.log(self.e))
        if cs2 is None:
            cs2 = 1.0 / np.maximum(np.gradient(self.e, self.p), 1.0)
        self._lc = PchipInterpolator(self._lp, np.log(np.asarray(cs2)[o]))

    def eps(self, p):
        return float(np.exp(self._f(np.log(max(p, self.pmin)))))

    def cs2(self, p):
        return float(np.exp(self._lc(np.log(max(p, self.pmin)))))

    def p_surface(self):
        return self.pmin

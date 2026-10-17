"""Full-GR l=2 f-mode (quasi-normal mode) of a non-rotating barotropic star.

The interior perturbation equations were derived symbolically (derive_matter.py, elim.py) from the linearised Einstein
equations and the conservation law in the Regge-Wheeler gauge; the state vector is y = (H1, K, U_r, dp) with
exp(i omega t) time dependence (lengths in km, G = c = 1, omega in km^-1).  The exterior is matched to the Zerilli equation
and the quasi-normal-mode condition is that the ingoing amplitude vanishes.
"""
import numpy as np
from scipy.integrate import solve_ivp
from fmode import Background, tortoise, zerilli_V, alpha_K, asym_Z, LAM, PI4
from matter_rhs import coef

L = 2
C_KMS = 2.99792458e5


def bg_terms(bg, r):
    p, m, nu, e, cs2, f, A, nup = bg.at(r)
    lam = -np.log(f)
    lam_d = (2 / f) * (PI4 * r * e - m / r ** 2)
    p_d = -(e + p) * nup / 2
    eps_d = p_d / cs2
    return dict(r=r, nu=nu, lam=lam, eps=e, p=p, cs2=cs2, nu_d=nup, lam_d=lam_d, eps_d=eps_d, p_d=p_d), (p, m, nu, e, cs2, f)


def matrix(bg, r, om):
    t, _ = bg_terms(bg, r)
    return coef(t["r"], om, t["nu"], t["lam"], t["eps"], t["p"], t["cs2"], t["nu_d"], t["lam_d"], t["eps_d"], t["p_d"])


SCALE = np.array([3.0, 2.0, 1.0, 2.0])      # regular behaviour r^s of (H1, K, U_r, dp) at the centre


def regular_start(bg, om, r0):
    A, _, _ = matrix(bg, r0, om)
    # y = D yhat, yhat' = (D^-1 A D - S/r) yhat ; regular solutions: (r D^-1 A D - S) yhat0 = 0
    D = np.diag(r0 ** SCALE)
    B = r0 * np.linalg.solve(D, A @ D) - np.diag(SCALE)
    u, s, vh = np.linalg.svd(B)
    null = vh[-2:].conj().T              # two smallest singular values
    return null, s


def interior(bg, om, r0=None, cut=1e-5):
    r0 = r0 or 2e-3 * bg.R
    null, sv = regular_start(bg, om, r0)
    y0 = (np.diag(r0 ** SCALE) @ null)           # columns: two regular solutions
    rend = bg.R * (1 - cut)

    def rhs(r, Y):
        A = matrix(bg, r, om)[0]
        Y = Y.reshape(4, 2)
        return (A @ Y).reshape(-1)
    sol = solve_ivp(rhs, (r0, rend), y0.reshape(-1).astype(complex), method="DOP853", rtol=1e-10, atol=1e-30)
    Y = sol.y[:, -1].reshape(4, 2)
    # surface condition: Lagrangian pressure perturbation vanishes
    t, (p, m, nu, e, cs2, f) = bg_terms(bg, rend)

    def dlag(y):
        xi = np.exp(nu / 2 + np.log(f)) * (y[2] - np.exp(-nu / 2) * y[0]) / (1j * om)     # e^{-lambda} = f
        return y[3] + xi * t["p_d"]
    a, b = dlag(Y[:, 0]), dlag(Y[:, 1])
    y = Y[:, 0] - (a / b) * Y[:, 1]
    return y, rend, sv


def vac_constraint(r, M, om):
    f = 1 - 2 * M / r
    c_h0 = -(r - 2 * M) * (3 * M + 2 * r) / (16 * r ** 4)
    c_h1 = -1j * (om ** 2 * r ** 3 - 3 * M) * (r - 2 * M) / (16 * om * r ** 5)
    c_k = -(3 * M ** 2 + 3 * M * r + om ** 2 * r ** 4 - 2 * r ** 2) / (16 * r ** 4)
    return c_h0, c_h1, c_k


def KH1_from_Z(r, M, om, Z, Zr):
    f = 1 - 2 * M / r
    al = alpha_K(r, M)
    h = 1e-6 * r
    dal = (alpha_K(r + h, M) - alpha_K(r - h, M)) / (2 * h)
    om2 = om * om
    Zrr = (zerilli_V(r, M) - om2) * Z / f ** 2 - (2 * M / r ** 2) / f * Zr
    K = al * Z + f * Zr
    Kr = dal * Z + al * Zr + (2 * M / r ** 2) * Zr + f * Zrr
    c_h0, c_h1, c_k = vac_constraint(r, M, om)
    # K' = H0/r - (r-3M)/(r^2 f) K + 3 H1/(i om r^2),   H0 = -(c_h1 H1 + c_k K)/c_h0
    a1 = -c_h1 / (c_h0 * r) + 3 / (1j * om * r ** 2)
    a0 = -c_k / (c_h0 * r) - (r - 3 * M) / (r ** 2 * f)
    H1 = (Kr - a0 * K) / a1
    return K, H1


def Z_from_KH1(r, M, om, K, H1):
    K1, H11 = KH1_from_Z(r, M, om, 1.0, 0.0)
    K2, H12 = KH1_from_Z(r, M, om, 0.0, 1.0)
    return np.linalg.solve(np.array([[K1, K2], [H11, H12]], complex), np.array([K, H1], complex))


def ingoing_amplitude(bg, om, rmax=None, cut=1e-5):
    M = bg.M
    y, rend, sv = interior(bg, om, cut=cut)
    y = y / y[1]
    Z, Zr = Z_from_KH1(rend, M, om, y[1], y[0])
    om2 = om * om
    rmax = rmax or max(4000.0, 25 * 2 * np.pi / abs(om.real))

    def fz(rr, s):
        f = 1 - 2 * M / rr
        return [s[1], (zerilli_V(rr, M) - om2) * s[0] / f ** 2 - (2 * M / rr ** 2) / f * s[1]]
    sol = solve_ivp(fz, (rend, rmax), [Z, Zr], method="DOP853", rtol=1e-11, atol=1e-30)
    Zn, Zrn = sol.y[0, -1], sol.y[1, -1]
    zo, dzo = asym_Z(rmax, M, om, +1, rend)
    zi, dzi = asym_Z(rmax, M, om, -1, rend)
    A = np.linalg.solve(np.array([[zi, zo], [dzi, dzo]], complex), np.array([Zn, Zrn], complex))
    return A[0], A[1], sv


def to_km(f_hz):
    return 2 * np.pi * f_hz / (C_KMS * 1e5) * 1e5 / 1.0 * 1.0 if False else 2 * np.pi * f_hz / (C_KMS * 1e3)


def find_mode(bg, f_guess_hz, tau_guess=0.2, cut=1e-5, tol=1e-9, maxit=40):
    k = 2 * np.pi / (C_KMS * 1e0)           # Hz -> km^-1 : omega = 2 pi f / c,  c in km/s
    om = k * f_guess_hz + 1j / (tau_guess * C_KMS)
    g = lambda w: ingoing_amplitude(bg, w, cut=cut)[0]
    w0, w1 = om, om * (1 + 1e-4)
    g0, g1 = g(w0), g(w1)
    for it in range(maxit):
        w2 = w1 - g1 * (w1 - w0) / (g1 - g0)
        w0, g0 = w1, g1
        w1, g1 = w2, g(w2)
        if abs(w1 - w0) < tol * abs(w1): break
    return w1.real / k, 1.0 / (w1.imag * C_KMS), w1

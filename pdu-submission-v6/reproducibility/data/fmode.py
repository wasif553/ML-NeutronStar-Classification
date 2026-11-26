"""Full-GR l=2 f-mode of a non-rotating barotropic star (Detweiler & Lindblom 1985 formulation).

Units: km, G=c=1.  omega is the complex angular frequency in km^-1 (time dependence exp(i omega t)).
Interior: H1, K, W, X equations; two regular solutions at the centre, combined to satisfy X=0 at the surface.
Exterior: Zerilli equation integrated outward and projected onto the asymptotic ingoing/outgoing solutions;
the quasi-normal-mode condition is that the ingoing amplitude vanishes.
"""
import numpy as np
from scipy.integrate import solve_ivp
from eos import MSUN_KM, P2G
from tov import solve_star

L = 2
LAM = (L - 1) * (L + 2) / 2.0          # Zerilli lambda
PI4 = 4 * np.pi


class Background:
    def __init__(self, eos, pc):
        s = solve_star(eos, pc, tidal=False, dense=True)
        self.eos, self.s = eos, s
        self.sol = s["sol"]
        self.R, self.M = s["R"], s["Mkm"]
        self.nu_shift = np.log(1 - 2 * self.M / self.R) - s["nuR"]
        self.psurf = s["psurf"]

    def at(self, r):
        p, m, nu, _ = self.sol.sol(r)
        nu = nu + self.nu_shift
        p = max(p, 1e-60)
        e = self.eos.eps(p); cs2 = self.eos.cs2(p)
        f = 1 - 2 * m / r
        A = m + PI4 * r ** 3 * p
        nup = 2 * A / (r * r * f)
        return p, m, nu, e, cs2, f, A, nup

    def Qp(self, r, p, m, e, f, A, nup):
        """(e^{-lambda/2} nu'/r^2)' evaluated analytically."""
        q = f ** -0.5
        dA = PI4 * r * r * e + 3 * PI4 * r * r * p + PI4 * r ** 3 * (-(e + p) * nup / 2)
        dq = q * (1 / f) * (PI4 * r * e - m / r ** 2)
        return 2 * (dA * q / r ** 2 + A * dq / r ** 2 - 2 * A * q / r ** 3)


def algebra(bg_at, r, om2, y):
    """H0 and V from (H1, K, W, X)."""
    p, m, nu, e, cs2, f, A, nup = bg_at
    H1, K, W, X = y
    elam = 1 / f
    enu = np.exp(nu)
    ex = om2 * r ** 3 * np.exp(-(nu + np.log(elam)))     # omega^2 r^3 e^{-(lambda+nu)}
    exK = om2 * r ** 3 * np.exp(-nu)                      # omega^2 r^3 e^{-nu}  (checked against the vacuum Ricci derivation)
    den = 3 * m + 0.5 * (L + 2) * (L - 1) * r + PI4 * r ** 3 * p
    H0 = (2 * PI4 * r ** 3 * np.exp(-nu / 2) * X - (0.5 * L * (L + 1) * A - ex) * H1
          + (0.5 * (L + 2) * (L - 1) * r - exK - (elam / r) * A * (3 * m - r + PI4 * r ** 3 * p)) * K) / den
    dp = -(e + p) * nup / 2
    if e + p > 0:
        V = np.exp(nu / 2) * (X + (dp / r) * np.exp((nu - np.log(elam)) / 2) * W - 0.5 * (e + p) * np.exp(nu / 2) * H0) / (om2 * (e + p))
    else:
        V = 0.0
    return H0, V


def rhs_one(bg, r, om2, y):
    at = bg.at(r)
    p, m, nu, e, cs2, f, A, nup = at
    H1, K, W, X = y
    H0, V = algebra(at, r, om2, y)
    elam = 1 / f; el2 = f ** -0.5
    dH1 = -(1 / r) * (L + 1 + 2 * m * elam / r + PI4 * r * r * elam * (p - e)) * H1 + (elam / r) * (H0 + K - 4 * PI4 * (e + p) * V)
    dK = H0 / r + (L * (L + 1) / (2 * r)) * H1 - ((L + 1) / r - nup / 2) * K - 2 * PI4 * (e + p) * el2 * W / r
    ew = np.exp(-nu / 2) * X / ((e + p) * cs2)
    dW = -((L + 1) / r) * W + r * el2 * (ew - (L * (L + 1) / r ** 2) * V + 0.5 * (H0 + K))
    Qp = bg.Qp(r, p, m, e, f, A, nup)
    br = PI4 * (e + p) * el2 + om2 * el2 * np.exp(-nu) - 0.5 * r * r * Qp
    dX = -(L / r) * X + (e + p) * np.exp(nu / 2) * (0.5 * (1 / r - nup / 2) * H0 + 0.5 * (r * om2 * np.exp(-nu) + L * (L + 1) / (2 * r)) * H1
                                                     + 0.5 * (1.5 * nup - 1 / r) * K - (L * (L + 1) / (2 * r * r)) * nup * V - (1 / r) * br * W)
    return np.array([dH1, dK, dW, dX])


def start_vectors(bg, om2, r0):
    """Two regular solutions at r0 (K0, W0) = (1, 0) and (0, 1); X0 fixed by the vanishing of the 1/r residual."""
    at = bg.at(r0)
    p, m, nu, e, cs2, f, A, nup = at
    out = []
    for K0, W0 in ((1.0, 0.0), (0.0, 1.0)):
        H10 = (2 * L * K0 + 4 * PI4 * (e + p) * W0) / (L * (L + 1))
        g0 = r0 * rhs_one(bg, r0, om2, np.array([H10, K0, W0, 0.0 + 0j]))[3]
        g1 = r0 * rhs_one(bg, r0, om2, np.array([H10, K0, W0, 1.0 + 0j]))[3]
        X0 = -g0 / (g1 - g0)
        out.append(np.array([H10, K0, W0, X0], complex))
    return out


def interior(bg, om, r0=None, cut=1e-5):
    om2 = om * om
    r0 = r0 or 1e-3 * bg.R
    ya, yb = start_vectors(bg, om2, r0)
    rend = bg.R * (1 - cut)

    def f(r, Y):
        return np.concatenate([rhs_one(bg, r, om2, Y[:4]), rhs_one(bg, r, om2, Y[4:])])
    sol = solve_ivp(f, (r0, rend), np.concatenate([ya, yb]), method="DOP853", rtol=1e-10, atol=1e-14)
    Y = sol.y[:, -1]
    a, b = Y[:4], Y[4:]
    y = a - (a[3] / b[3]) * b                      # enforce X = 0 (Lagrangian pressure vanishes at the surface)
    return y, rend


def vacuum_H0(r, M, om2, H1, K):
    f = 1 - 2 * M / r
    ex = om2 * r ** 3                                  # e^{-(lambda+nu)} = 1 outside
    exK = om2 * r ** 3 / f                             # e^{-nu} = 1/f outside
    den = 3 * M + 0.5 * (L + 2) * (L - 1) * r
    return (-(0.5 * L * (L + 1) * M - ex) * H1 + (0.5 * (L + 2) * (L - 1) * r - exK - (1 / (r * f)) * M * (3 * M - r)) * K) / den


# --- exterior: Zerilli function -------------------------------------------------------------
def zerilli_V(r, M):
    num = LAM ** 2 * (LAM + 1) * r ** 3 + 3 * LAM ** 2 * M * r ** 2 + 9 * LAM * M ** 2 * r + 9 * M ** 3
    return 2 * (r - 2 * M) / (r ** 4 * (LAM * r + 3 * M) ** 2) * num


def alpha_K(r, M):
    return (LAM * (LAM + 1) * r ** 2 + 3 * LAM * M * r + 6 * M ** 2) / (r ** 2 * (LAM * r + 3 * M))


def H1_from_ZZp(r, M, om2, Z, Zr):
    """Given Z and dZ/dr, return (K, H1) in the exterior.  K_RW = alpha Z + dZ/dr* (Regge-Wheeler gauge); the DL variables
    are K = K_RW / r^L and the (K') equation then fixes H1."""
    f = 1 - 2 * M / r
    al = alpha_K(r, M)
    KRW = al * Z + f * Zr
    h = 1e-6 * r
    dal = (alpha_K(r + h, M) - alpha_K(r - h, M)) / (2 * h)
    Zrr = (zerilli_V(r, M) - om2) * Z / f ** 2 - (2 * M / r ** 2) / f * Zr
    KRWr = dal * Z + al * Zr + (2 * M / r ** 2) * Zr + f * Zrr
    K = KRW / r ** L
    Kr = KRWr / r ** L - L * K / r
    nup = 2 * M / (r * r * f)
    ex = om2 * r ** 3
    exK = om2 * r ** 3 / f
    den = 3 * M + 0.5 * (L + 2) * (L - 1) * r
    c1 = -(0.5 * L * (L + 1) * M - ex) / den
    c2 = (0.5 * (L + 2) * (L - 1) * r - exK - (1 / (r * f)) * M * (3 * M - r)) / den
    H1 = (Kr - c2 * K / r + ((L + 1) / r - nup / 2) * K) / (c1 / r + L * (L + 1) / (2 * r))
    return K, H1


def ZZp_from_KH1(r, M, om2, K, H1):
    """Invert the 2x2 linear map (Z, Zr) -> (K, H1)."""
    K1, H11 = H1_from_ZZp(r, M, om2, 1.0, 0.0)
    K2, H12 = H1_from_ZZp(r, M, om2, 0.0, 1.0)
    Mx = np.array([[K1, K2], [H11, H12]], complex)
    return np.linalg.solve(Mx, np.array([K, H1], complex))


def asym_coeffs(M, om, nmax=8):
    """Series u(r) = sum a_n r^-n for Z = exp(-i om r*) u(r) (outgoing for exp(i om t))."""
    # W(r) = V/f expanded in 1/r:  W = 2 num / (r^3 (lam r + 3M)^2)
    from numpy.polynomial import polynomial as Pn
    N = nmax + 4
    # 1/(lam r + 3M)^2 = (lam r)^-2 (1 + 3M/(lam r))^-2 = (lam r)^-2 sum (k+1)(-3M/(lam r))^k
    inv = np.array([(k + 1) * (-3 * M / LAM) ** k for k in range(N)]) / LAM ** 2     # coefficients of r^-(k+2)
    num = np.array([9 * M ** 3, 9 * LAM * M ** 2, 3 * LAM ** 2 * M, LAM ** 2 * (LAM + 1)])   # num = sum num[j] r^j (j=0..3)
    # W = 2 * sum_j num[j] r^(j-3) * sum_k inv[k] r^-(k+2) -> r^-(k+5-j)
    w = np.zeros(N + 6)
    for j in range(4):
        for k in range(N):
            w[k + 5 - j] += 2 * num[j] * inv[k]
    a = np.zeros(nmax + 1, complex); a[0] = 1.0
    for m_ in range(1, nmax + 1):
        s = -(m_ - 1) * m_ * a[m_ - 1]
        if m_ >= 2: s += 2 * M * (m_ - 2) * m_ * a[m_ - 2]
        for k in range(2, m_ + 2):
            if m_ + 1 - k >= 0: s += w[k] * a[m_ + 1 - k]
        a[m_] = s / (2j * om * m_)
    return a


def tortoise(r, M):
    return r + 2 * M * np.log(r / (2 * M) - 1)


def asym_Z(r, M, om, sign, rs):
    """outgoing (sign=+1) or ingoing (sign=-1) asymptotic solution and its r-derivative."""
    w = sign * om
    a = asym_coeffs(M, w)
    n = np.arange(len(a))
    u = np.sum(a * r ** (-n)); du = np.sum(-n * a * r ** (-n - 1.0))
    f = 1 - 2 * M / r
    ph = np.exp(-1j * w * (tortoise(r, M) - tortoise(rs, M)))
    return ph * u, ph * (du - 1j * w * u / f)


def ingoing_amplitude(bg, om, rmax=None, cut=1e-5):
    M = bg.M
    y, rend = interior(bg, om, cut=cut)
    om2 = om * om
    r = rend
    # fluid variables at r=rend with X=0; vacuum matching (the surface layer r in [rend, R] is negligible)
    y = y / y[1]                                     # analytic normalisation (K = 1 at the surface)
    Z, Zr = ZZp_from_KH1(r, M, om2, y[1], y[0])
    rmax = rmax or max(4000.0, 25 * 2 * np.pi / abs(om.real))

    def fz(rr, s):
        f = 1 - 2 * M / rr
        return [s[1], (zerilli_V(rr, M) - om2) * s[0] / f ** 2 - (2 * M / rr ** 2) / f * s[1]]
    sol = solve_ivp(fz, (r, rmax), [Z, Zr], method="DOP853", rtol=1e-11, atol=1e-30)
    Zn, Zrn = sol.y[0, -1], sol.y[1, -1]
    zo, dzo = asym_Z(rmax, M, om, +1, r)
    zi, dzi = asym_Z(rmax, M, om, -1, r)
    A = np.linalg.solve(np.array([[zi, zo], [dzi, dzo]], complex), np.array([Zn, Zrn], complex))
    return A[0], y, A[1]


def find_mode(bg, f_guess_hz, tau_guess=0.2, cut=1e-5, tol=1e-10, maxit=30):
    kmps = 1e5 / 2.99792458e10          # not used; conversion below
    to_km = 2 * np.pi / 2.99792458e5    # Hz -> km^-1
    om = to_km * f_guess_hz + 1j / (tau_guess * 2.99792458e5)
    g = lambda w: ingoing_amplitude(bg, w, cut=cut)[0]
    w0, w1 = om, om * (1 + 1e-3)
    g0, g1 = g(w0), g(w1)
    for it in range(maxit):
        w2 = w1 - g1 * (w1 - w0) / (g1 - g0)
        w0, g0 = w1, g1
        w1, g1 = w2, g(w2)
        if abs(w1 - w0) < tol * abs(w1): break
    f_hz = w1.real / to_km
    tau = 1.0 / (w1.imag * 2.99792458e5)
    return f_hz, tau, w1

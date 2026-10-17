"""Derive, from the linearised Einstein equations and the conservation law, the l=2 even-parity perturbation equations of a
barotropic perfect-fluid star (Regge-Wheeler gauge, exp(i omega t) time dependence).  Output: a pickle of coefficient
expressions for the linear system used by fmode2.py."""
import sympy as sp, pickle, time

t0 = time.time()
t, r, th, ph, w = sp.symbols("t r theta phi omega")
x = [t, r, th, ph]
E = sp.exp(sp.I * w * t)
nu, lam, eps0, p0 = [sp.Function(n)(r) for n in ("nu", "lam", "eps0", "p0")]
H0, H1, K, dp, Ur, Uth = [sp.Function(n)(r) for n in ("H0", "H1", "K", "dp", "Ur", "Uth")]
Y = (3 * sp.cos(th) ** 2 - 1) / 2
th0 = sp.pi / 3
cs2 = sp.Symbol("cs2")

g0 = sp.diag(-sp.exp(nu), sp.exp(lam), r ** 2, r ** 2 * sp.sin(th) ** 2)
gi0 = sp.diag(-sp.exp(-nu), sp.exp(-lam), 1 / r ** 2, 1 / (r ** 2 * sp.sin(th) ** 2))
h = sp.zeros(4, 4)
h[0, 0] = sp.exp(nu) * H0 * Y * E
h[0, 1] = h[1, 0] = H1 * Y * E
h[1, 1] = sp.exp(lam) * H0 * Y * E           # H2 = H0
h[2, 2] = r ** 2 * K * Y * E
h[3, 3] = r ** 2 * sp.sin(th) ** 2 * K * Y * E

G0 = [[[sum(gi0[a, d] * (sp.diff(g0[d, b], x[c]) + sp.diff(g0[d, c], x[b]) - sp.diff(g0[b, c], x[d])) for d in range(4)) / 2
        for c in range(4)] for b in range(4)] for a in range(4)]
G0 = [[[sp.simplify(G0[a][b][c]) for c in range(4)] for b in range(4)] for a in range(4)]


def cov(hh, d, b, c):    # nabla_b h_{dc}
    s = sp.diff(hh[d, c], x[b])
    for e in range(4):
        s -= G0[e][b][d] * hh[e, c] + G0[e][b][c] * hh[d, e]
    return s


G1 = [[[sp.expand(sum(gi0[a, d] * (cov(h, d, b, c) + cov(h, d, c, b) - cov(h, b, d, c)) for d in range(4)) / 2) for c in range(4)] for b in range(4)] for a in range(4)]


def ricci0():
    R = sp.zeros(4, 4)
    for b in range(4):
        for c in range(4):
            s = 0
            for a in range(4):
                s += sp.diff(G0[a][b][c], x[a]) - sp.diff(G0[a][b][a], x[c])
                for d in range(4):
                    s += G0[a][a][d] * G0[d][b][c] - G0[a][c][d] * G0[d][b][a]
            R[b, c] = sp.simplify(s)
    return R


R0 = ricci0()
R1 = sp.zeros(4, 4)
for b in range(4):
    for c in range(b, 4):
        s = 0
        for a in range(4):
            s += sp.diff(G1[a][b][c], x[a]) - sp.diff(G1[a][b][a], x[c])
            for d in range(4):
                s += G1[a][a][d] * G0[d][b][c] + G0[a][a][d] * G1[d][b][c] - G1[a][c][d] * G0[d][b][a] - G0[a][c][d] * G1[d][b][a]
        R1[b, c] = R1[c, b] = s
print("Ricci done", time.time() - t0, flush=True)
Rs0 = sum(gi0[a, a] * R0[a, a] for a in range(4))
hup = gi0 * h * gi0
dRs = sum(-hup[a, b] * R0[a, b] + gi0[a, b] * R1[a, b] for a in range(4) for b in range(4))
dG = sp.zeros(4, 4)
for a in range(4):
    for b in range(a, 4):
        dG[a, b] = dG[b, a] = R1[a, b] - sp.Rational(1, 2) * h[a, b] * Rs0 - sp.Rational(1, 2) * g0[a, b] * dRs

# fluid perturbation
u0 = [-sp.exp(nu / 2), 0, 0, 0]
du = [sp.exp(nu / 2) * H0 * Y * E / 2, Ur * Y * E, Uth * sp.diff(Y, th) * E, 0]
deps = dp * Y * E / cs2
dpp = dp * Y * E
dT = sp.zeros(4, 4)
for a in range(4):
    for b in range(a, 4):
        dT[a, b] = dT[b, a] = ((deps + dpp) * u0[a] * u0[b] + (eps0 + p0) * (du[a] * u0[b] + u0[a] * du[b]) + dpp * g0[a, b] + p0 * h[a, b])
Eq = {}
for (a, b) in [(0, 1), (0, 2), (1, 2), (1, 1)]:
    Eq[(a, b)] = sp.expand((dG[a, b] - 8 * sp.pi * dT[a, b]) / E).subs(th, th0).doit()
print("Einstein done", time.time() - t0, flush=True)

# conservation: nabla_mu T^{mu nu} = 0 at first order (contravariant components)
uup0 = [sp.exp(-nu / 2), 0, 0, 0]
duu = [gi0[a, a] * du[a] - hup[a, 0] * u0[0] for a in range(4)]       # delta u^mu = g0^{mu a} du_a - h^{mu a} u0_a
dTup = sp.zeros(4, 4)
for a in range(4):
    for b in range(4):
        dTup[a, b] = ((deps + dpp) * uup0[a] * uup0[b] + (eps0 + p0) * (duu[a] * uup0[b] + uup0[a] * duu[b]) + dpp * gi0[a, b] - p0 * hup[a, b])
T0 = sp.diag(eps0 * sp.exp(-nu), p0 * sp.exp(-lam), p0 / r ** 2, p0 / (r ** 2 * sp.sin(th) ** 2))
cons = {}
for nuidx in (0, 1, 2):
    s_ = 0
    for mu in range(4):
        s_ += sp.diff(dTup[mu, nuidx], x[mu])
        for lm in range(4):
            s_ += G0[mu][mu][lm] * dTup[lm, nuidx] + G1[mu][mu][lm] * T0[lm, nuidx] + G0[nuidx][mu][lm] * dTup[mu, lm]
        s_ += G1[nuidx][mu][mu] * T0[mu, mu]
    cons[nuidx] = sp.expand(s_ / E).subs(th, th0).doit()
print("conservation done", time.time() - t0, flush=True)
pickle.dump({"Eq": Eq, "cons": cons}, open("matter_eqs.pkl", "wb"))
print("saved")

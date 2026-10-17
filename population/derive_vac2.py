"""Linearised Ricci tensor for even-parity l=2, m=0 perturbations of Schwarzschild (Regge-Wheeler gauge)."""
import sympy as sp, pickle

t, r, th, ph, M, w = sp.symbols("t r theta phi M omega")
f = 1 - 2 * M / r
H0, H1, H2, K = [sp.Function(n)(r) for n in ("H0", "H1", "H2", "K")]
Y = (3 * sp.cos(th) ** 2 - 1) / 2
x = [t, r, th, ph]
E = sp.exp(sp.I * w * t)
g0 = sp.diag(-f, 1 / f, r ** 2, r ** 2 * sp.sin(th) ** 2)
h = sp.zeros(4, 4)
h[0, 0] = f * H0 * Y * E
h[0, 1] = h[1, 0] = H1 * Y * E
h[1, 1] = H2 * Y * E / f
h[2, 2] = r ** 2 * K * Y * E
h[3, 3] = r ** 2 * sp.sin(th) ** 2 * K * Y * E
gi0 = g0.inv()
G0 = [[[sum(gi0[a, d] * (sp.diff(g0[d, b], x[c]) + sp.diff(g0[d, c], x[b]) - sp.diff(g0[b, c], x[d])) for d in range(4)) / 2
        for c in range(4)] for b in range(4)] for a in range(4)]
G0 = [[[sp.simplify(G0[a][b][c]) for c in range(4)] for b in range(4)] for a in range(4)]
th0 = sp.pi/3
# first-order Christoffel: dG^a_bc = 1/2 g0^{ad} (nabla_b h_dc + nabla_c h_db - nabla_d h_bc)
def cov(hh, d, b, c):   # nabla_b h_{dc}
    s = sp.diff(hh[d, c], x[b])
    for e in range(4):
        s -= G0[e][b][d] * hh[e, c] + G0[e][b][c] * hh[d, e]
    return s
G1 = [[[sp.expand(sum(gi0[a, d] * (cov(h, d, b, c) + cov(h, d, c, b) - cov(h, b, d, c)) for d in range(4)) / 2) for c in range(4)] for b in range(4)] for a in range(4)]
R1 = sp.zeros(4, 4)
for b in range(4):
    for c in range(b, 4):
        s = 0
        for a in range(4):
            s += sp.diff(G1[a][b][c], x[a]) - sp.diff(G1[a][b][a], x[c])
            for d in range(4):
                s += G1[a][a][d] * G0[d][b][c] + G0[a][a][d] * G1[d][b][c] - G1[a][c][d] * G0[d][b][a] - G0[a][c][d] * G1[d][b][a]
        R1[b, c] = R1[c, b] = sp.simplify(sp.expand(s / E).subs(th, th0).doit())
for (a, b) in [(0, 1), (0, 2), (1, 2), (2, 3), (0, 0), (1, 1), (2, 2), (3, 3)]:
    print((a, b), R1[a, b], flush=True)
pickle.dump({k: R1[k] for k in [(0, 1), (0, 2), (1, 2), (2, 3), (0, 0), (1, 1), (2, 2), (3, 3)]}, open("vac_ricci.pkl", "wb"))

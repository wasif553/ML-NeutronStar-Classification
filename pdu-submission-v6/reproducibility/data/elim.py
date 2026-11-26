import sympy as sp, pickle
S = pickle.load(open("matter_sym.pkl", "rb"))
sym = lambda n: sp.Symbol(n)
h0, h1, k, dp, ur, ut = map(sym, "h0 h1 k dp ur ut".split())
h0d, h1d, kd, dpd, urd = map(sym, "h0_d h1_d k_d dp_d ur_d".split())
tr, tth, rth, rr, c0, c1, c2 = S[(0, 1)], S[(0, 2)], S[(1, 2)], S[(1, 1)], S[("c", 0)], S[("c", 1)], S[("c", 2)]
sol = {}
sol[ut] = sp.solve(c2, ut)[0]
sol[kd] = sp.solve(tr, kd)[0]
sol[h0d] = sp.solve(rth.subs(kd, sol[kd]), h0d)[0]
sol[h1d] = sp.solve(tth.subs(ut, sol[ut]), h1d)[0]
rr2 = rr.subs({h0d: sol[h0d], kd: sol[kd]})
sol[h0] = sp.solve(rr2, h0)[0]
H0e = sol[h0]
def full(e):
    e = e.subs(sol[ut].free_symbols and {ut: sol[ut]}) if False else e
    return e
dp_d = sp.solve(c1.subs(h0d, sol[h0d]), dpd)[0]
ur_d = sp.solve(c0.subs({h1d: sol[h1d], ut: sol[ut]}), urd)[0]
# assemble: substitute H0 = H0e (a function of state) everywhere
state = [h1, k, ur, dp]
rhs = [sol[h1d].subs(ut, sol[ut]), sol[kd], ur_d, dp_d]
rhs = [e.subs(h0, H0e) for e in rhs]
A = sp.Matrix(4, 4, lambda i, j: sp.diff(rhs[i], state[j]))
# check linearity: residual with state set to zero
for e in rhs: assert sp.simplify(e.subs({h1: 0, k: 0, ur: 0, dp: 0})) == 0
bg = sp.symbols("r omega nu lam eps p cs2 nu_d lam_d eps_d p_d")
print("free symbols:", sorted(str(s_) for s_ in A.free_symbols))
H0c = [sp.diff(H0e, s_) for s_ in state]
UTc = [sp.diff(sol[ut].subs(h0, H0e), s_) for s_ in state]
exprs = list(A) + H0c + UTc
reps, red = sp.cse(exprs, optimizations="basic")
from sympy.printing.numpy import NumPyPrinter
pr = NumPyPrinter()
lines = ["import numpy as np", "from numpy import exp, sqrt, pi, sin, cos", "", "def coef(r, omega, nu, lam, eps, p, cs2, nu_d, lam_d, eps_d, p_d):"]
for a, b in reps: lines.append("    %s = %s" % (a, pr.doprint(b)))
lines.append("    A = np.array([%s], dtype=complex).reshape(4, 4)" % ", ".join(pr.doprint(e) for e in red[:16]))
lines.append("    H0c = np.array([%s], dtype=complex)" % ", ".join(pr.doprint(e) for e in red[16:20]))
lines.append("    UTc = np.array([%s], dtype=complex)" % ", ".join(pr.doprint(e) for e in red[20:24]))
lines.append("    return A, H0c, UTc")
open("matter_rhs.py", "w").write(chr(10).join(lines))
print("written", len(lines), "lines")

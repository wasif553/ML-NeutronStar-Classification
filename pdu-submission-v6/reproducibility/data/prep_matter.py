import sympy as sp, pickle
r, w = sp.symbols("r omega")
nu, lam, eps0, p0 = [sp.Function(n)(r) for n in ("nu", "lam", "eps0", "p0")]
H0, H1, K, dp, Ur, Uth = [sp.Function(n)(r) for n in ("H0", "H1", "K", "dp", "Ur", "Uth")]
d = pickle.load(open("matter_eqs.pkl", "rb"))
names = dict(H0="h0", H1="h1", K="k", dp="dp", Ur="ur", Uth="ut")
funcs = dict(H0=H0, H1=H1, K=K, dp=dp, Ur=Ur, Uth=Uth)
S = {}
def rep(e):
    # unknown functions: highest derivatives first
    for nm, F in funcs.items():
        for order in (2, 1):
            e = e.subs(sp.Derivative(F, (r, order)), sp.Symbol(names[nm] + "_" + "d" * order))
        e = e.subs(F, sp.Symbol(names[nm]))
    # background functions
    for nm, F, sy in (("nu", nu, "nu"), ("lam", lam, "lam"), ("eps0", eps0, "eps"), ("p0", p0, "p")):
        for order in (2, 1):
            e = e.subs(sp.Derivative(F, (r, order)), sp.Symbol(sy + "_" + "d" * order))
        e = e.subs(F, sp.Symbol(sy))
    return e
out = {}
for key, e in list(d["Eq"].items()) + [(("c", k), v) for k, v in d["cons"].items()]:
    e = rep(e)
    syms = sorted(str(s) for s in e.free_symbols)
    out[key] = e
    print(key, syms)
pickle.dump(out, open("matter_sym.pkl", "wb"))

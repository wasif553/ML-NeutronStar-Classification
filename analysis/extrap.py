"""E5: extrapolation to an unseen mass window (top 25% of each class) -- accuracy, epistemic uncertainty
and conformal coverage when exchangeability is violated."""
import numpy as np, pandas as pd, warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from common import *
from main import make_mlp, metrics, ent, prep, NENS, AUG, NEVAL
warnings.filterwarnings("ignore")
D = load()

def task(regime_name, rep, s, fs, seed, top=0.25):
    rng = np.random.default_rng(seed); d = regime(D, regime_name); cols = FEATS[fs]; classes = np.array(sorted(d.y.unique()))
    ext = np.zeros(len(d), bool)
    for k in classes:
        idx = np.where(d.y.values == k)[0]; ext[idx[int(len(idx) * (1 - top)):]] = True
    din, dex = d[~ext].reset_index(drop=True), d[ext].reset_index(drop=True)
    role = block_split(din, rng)
    tr, ca, te = (din[role == r] for r in (0, 1, 2))
    Xtr, ytr, _ = prep(tr, cols, s, AUG, rng, classes); Xca, yca, _ = prep(ca, cols, s, AUG, rng, classes)
    Xte, yte, _ = prep(te, cols, s, NEVAL, rng, classes); Xex, yex, _ = prep(dex, cols, s, NEVAL, rng, classes)
    sc = StandardScaler().fit(logfeat(Xtr)); T = lambda X: sc.transform(logfeat(X))
    ens = [make_mlp(seed * 10 + j).fit(T(Xtr), ytr) for j in range(NENS)]
    P = lambda X: np.stack([m.predict_proba(T(X)) for m in ens]); pc = P(Xca).mean(0)
    rows = []
    for nm, X, y in (("in-distribution", Xte, yte), ("unseen mass window", Xex, yex)):
        Pm = P(X); p = Pm.mean(0); r = metrics(p, y, pc, yca, len(ca)); r["MI"] = float((ent(p) - ent(Pm).mean(0)).mean())
        rows.append(dict(regime=regime_name, rep=rep, scale=s, feats=fs, test=nm, model="MLP-ens", **r))
    grid, gy = d[cols].values.astype(float), d.y.values
    pe = np.exp(bayes_logpost(Xex, grid, gy, cols, s, classes)); pcc = np.exp(bayes_logpost(Xca, grid, gy, cols, s, classes))
    rows.append(dict(regime=regime_name, rep=rep, scale=s, feats=fs, test="unseen mass window", model="Bayes-oracle", **metrics(pe, yex, pcc, yca, len(ca))))
    return rows

if __name__ == "__main__":
    jobs = [(rg, rep, s, fs, 500 * rep + 3) for rg in ("B",) for rep in range(10) for s in (1.0,) for fs in FEATS]
    out = Parallel(n_jobs=3, verbose=3)(delayed(task)(*j) for j in jobs)
    pd.DataFrame([r for o in out for r in o]).to_csv("results/extrap.csv", index=False)

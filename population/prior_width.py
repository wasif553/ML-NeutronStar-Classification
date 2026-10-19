"""Sensitivity of the nucleonic-versus-dark-matter accuracy to the width of the nuclear prior (subsets of the main population)."""
import json, numpy as np, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pop_data_utils import load_all, tables, NM
warnings.filterwarnings("ignore")
RECS = load_all(); CLS, MG, OBS, LAM = tables(RECS)
RANGES = dict(logP1=(34.15, 34.60), G1=(2.4, 3.8), G2=(2.4, 3.9), G3=(2.2, 3.9))
SCEN = {"noise-free": [0.0] * 4, "s=0.25": list(0.25 * np.array([0.05, 0.05, 0.02, 0.20])), "s=1": list(np.array([0.05, 0.05, 0.02, 0.20]))}


def inside(par, w):
    return all(abs((par[k] - (lo + hi) / 2) / (hi - lo)) <= w / 2 for k, (lo, hi) in RANGES.items())


def draw(idx, sig, rng, nd):
    X = OBS[np.ix_(idx, np.arange(0, NM, 2))].reshape(-1, 4); X = np.repeat(X, nd, 0)
    y = np.repeat(np.repeat(CLS[idx], len(np.arange(0, NM, 2))), nd); g = np.repeat(np.repeat(idx, len(np.arange(0, NM, 2))), nd)
    sg = np.array(sig); e = rng.standard_normal(X.shape); bad = (1 + sg * e) <= 0
    while bad.any():
        e[bad] = rng.standard_normal(int(bad.sum())); bad = (1 + sg * e) <= 0
    return X * (1 + sg * e), y, g


def run(w, scen, sig, seed):
    rng = np.random.default_rng(seed)
    sel = np.array([i for i, r in enumerate(RECS) if CLS[i] in (0, 2) and inside(r["par"], w)])
    n0, n2 = int((CLS[sel] == 0).sum()), int((CLS[sel] == 2).sum()); accs = {"LogReg": [], "RandForest": []}
    for rep in range(10):
        order = rng.permutation(len(sel)); fold = np.zeros(len(sel), int); fold[order] = np.arange(len(sel)) % 5
        for k in range(5):
            te, tr = sel[fold == k], sel[fold != k]
            Xtr, ytr, _ = draw(tr, sig, rng, 1 if max(sig) == 0 else 3); Xte, yte, _ = draw(te, sig, rng, 1 if max(sig) == 0 else 3)
            sc = StandardScaler().fit(np.log(Xtr)); A, B = sc.transform(np.log(Xtr)), sc.transform(np.log(Xte))
            accs["LogReg"].append((LogisticRegression(max_iter=2000).fit(A, ytr).predict(B) == yte).mean())
            accs["RandForest"].append((RandomForestClassifier(100, min_samples_leaf=3, n_jobs=1, random_state=rep).fit(A, ytr).predict(B) == yte).mean())
    return dict(width=w, n_nuc=n0, n_dm=n2, scenario=scen, **{m: float(np.mean(v)) for m, v in accs.items()})


if __name__ == "__main__":
    from joblib import Parallel, delayed
    jobs = [(w, sc, sg, 11 + j) for j, (w, (sc, sg)) in enumerate([(w, x) for w in (1.0, 0.9, 0.8, 0.7) for x in SCEN.items()])]
    out = Parallel(n_jobs=12, verbose=3)(delayed(run)(*j) for j in jobs)
    for r in out: print(r)
    json.dump(out, open("results_prior_width.json", "w"), indent=1)

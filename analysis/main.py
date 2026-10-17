"""Noise-robustness, calibration and conformal-uncertainty study (E2-E4)."""
import numpy as np, pandas as pd, json, warnings, sys, time
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from scipy.special import softmax, log_softmax
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from common import *
warnings.filterwarnings("ignore")
D = load()
SCALES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
NREP, AUG, NEVAL, NENS, ALPHA = 10, 20, 30, 3, 0.10


def ece(p, y, nb=15):
    conf, pred = p.max(1), p.argmax(1); acc = (pred == y).astype(float); e = 0.
    edges = np.linspace(0, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e

def reliab(p, y, nb=10):
    conf, ok = p.max(1), (p.argmax(1) == y).astype(float); edges = np.linspace(0, 1, nb + 1)
    return [[int(((conf > lo) & (conf <= hi)).sum()), float(conf[(conf > lo) & (conf <= hi)].sum()), float(ok[(conf > lo) & (conf <= hi)].sum())]
            for lo, hi in zip(edges[:-1], edges[1:])]

def nll(p, y): return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
def brier(p, y): oh = np.eye(p.shape[1])[y]; return float(((p - oh) ** 2).sum(1).mean())
def ent(p): return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)

def fit_T(logp, y):
    f = lambda t: nll(softmax(logp / np.exp(t), 1), y)
    return float(np.exp(minimize_scalar(f, bounds=(-2, 3), method="bounded").x))

def conformal(pc, yc, pt, yt, alpha=ALPHA, n_eff=None):
    """Split conformal (LAC score). Noisy replicates of one grid point are not exchangeable draws, so the
    finite-sample level correction uses the number of distinct calibration grid points (n_eff)."""
    s = 1 - pc[np.arange(len(yc)), yc]; n = n_eff or len(s)
    q = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = pt >= 1 - q
    return float(sets[np.arange(len(yt)), yt].mean()), float(sets.sum(1).mean())

def metrics(p, y, pc=None, yc=None, n_eff=None):
    out = dict(acc=float((p.argmax(1) == y).mean()), nll=nll(p, y), brier=brier(p, y), ece=float(ece(p, y)), H=float(ent(p).mean()))
    if pc is not None:
        out["cov"], out["setsize"] = conformal(pc, yc, p, y, n_eff=n_eff)
    return out

def make_mlp(seed): return MLPClassifier((64, 64), alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=200,
                                         early_stopping=True, n_iter_no_change=15, random_state=seed)

def prep(d, cols, s, reps, rng, classes):
    X, y, i = noisy_copies(d, cols, s, reps, rng)
    return X, np.searchsorted(classes, y), i

def task(regime_name, rep, s, fs, seed):
    t0 = time.time(); rng = np.random.default_rng(seed)
    d = regime(D, regime_name); cols = FEATS[fs]; classes = np.array(sorted(d.y.unique()))
    role = block_split(d, rng)
    tr, ca, te = (d[role == r] for r in (0, 1, 2))
    reps_tr = 1 if s == 0 else AUG
    Xtr, ytr, _ = prep(tr, cols, s, reps_tr, rng, classes)
    Xca, yca, _ = prep(ca, cols, s, 1 if s == 0 else AUG, rng, classes)
    Xte, yte, _ = prep(te, cols, s, 1 if s == 0 else NEVAL, rng, classes)
    sc = StandardScaler().fit(logfeat(Xtr)); T = lambda X: sc.transform(logfeat(X))
    ens = [make_mlp(seed * 10 + j).fit(T(Xtr), ytr) for j in range(NENS)]
    P = lambda X: np.stack([m.predict_proba(T(X)) for m in ens])       # (members, n, K)
    Pc, Pt = P(Xca), P(Xte); pc, pt = Pc.mean(0), Pt.mean(0)
    rows = []
    base = dict(regime=regime_name, rep=rep, scale=s, feats=fs, n_classes=len(classes))
    NE = len(ca); r = metrics(pt, yte, pc, yca, NE); r["MI"] = float((ent(pt) - ent(Pt).mean(0)).mean())
    Tt = fit_T(np.log(np.clip(pc, 1e-12, None)), yca)
    ptT, pcT = softmax(np.log(np.clip(pt, 1e-12, None)) / Tt, 1), softmax(np.log(np.clip(pc, 1e-12, None)) / Tt, 1)
    rT = metrics(ptT, yte, pcT, yca, NE)
    rows.append({**base, "model": "MLP-ens", **r})
    rows.append({**base, "model": "MLP-ens+T", **rT, "T": Tt})
    # simple baselines on identical data
    lr = LogisticRegression(max_iter=2000).fit(T(Xtr), ytr); rows.append({**base, "model": "LogReg", **metrics(lr.predict_proba(T(Xte)), yte)})
    rf = RandomForestClassifier(150, min_samples_leaf=3, n_jobs=1, random_state=seed).fit(T(Xtr), ytr)
    rows.append({**base, "model": "RandForest", **metrics(rf.predict_proba(T(Xte)), yte)})
    rel = {"MLP-ens": reliab(pt, yte), "MLP-ens+T": reliab(ptT, yte)}
    cm = np.zeros((len(classes),) * 2, int)
    for a, b in zip(yte, pt.argmax(1)): cm[a, b] += 1
    # oracle Bayes classifier (knows the model curves; noise model exact)
    if s > 0:
        grid, gy = d[cols].values.astype(float), d.y.values
        lpt = bayes_logpost(Xte, grid, gy, cols, s, classes); lpc = bayes_logpost(Xca, grid, gy, cols, s, classes)
        rows.append({**base, "model": "Bayes-oracle", **metrics(np.exp(lpt), yte, np.exp(lpc), yca, NE)})
        rel["Bayes-oracle"] = reliab(np.exp(lpt), yte)
    return rows, dict(regime=regime_name, scale=s, feats=fs, cm=cm.tolist(), classes=classes.tolist(), rel=rel), time.time() - t0


if __name__ == "__main__":
    import os
    only_s = [float(x) for x in os.environ.get("ONLY_SCALES", "").split(",") if x]; only_f = [x for x in os.environ.get("ONLY_FS", "").split(",") if x]
    jobs = [(rg, rep, s, fs, 1000 * rep + 7) for rg in REGIMES for rep in range(NREP) for s in SCALES for fs in FEATS
            if (not only_s or s in only_s) and (not only_f or fs in only_f)]
    print(len(jobs), "tasks", flush=True)
    out = Parallel(n_jobs=6, verbose=5)(delayed(task)(*j) for j in jobs)
    rows = [r for o in out for r in o[0]]
    pd.DataFrame(rows).to_csv(os.environ.get("MAIN_OUT", "results/main.csv"), index=False)
    json.dump([o[1] for o in out], open(os.environ.get("CONF_OUT", "results/confusions.json"), "w"))
    print("done; mean task time", np.mean([o[2] for o in out]))

"""Out-of-family test: train on one EoS population (priors/functional forms A), test on the independently constructed population B, and vice versa."""
import numpy as np, json, sys, warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import pop_data_utils as U
from pop_experiment import SCEN, mlp, bayes_kernel as _bk
from scipy.special import logsumexp
warnings.filterwarnings("ignore")

U.N_PER_CLASS = 90; RA = U.load_all("pop_data")
U.N_PER_CLASS = 70; RB = U.load_all("pop_data_B")
TA, TB = U.tables(RA), U.tables(RB)         # (cls, mg, obs, lam)
NM = U.NM
SC = ["ALL@0", "NICER+f20", "SO@1.0", "SO@0.25"]


def draw(T, idx, mass_idx, nd, cols, sig, rng):
    cls, mg, obs, lam = T
    X = obs[np.ix_(idx, mass_idx)][:, :, cols].reshape(-1, len(cols))
    X = np.repeat(X, nd, 0); y = np.repeat(np.repeat(cls[idx], len(mass_idx)), nd)
    sg = np.array(sig); eps = rng.standard_normal(X.shape); bad = (1 + sg * eps) <= 0
    while bad.any():
        eps[bad] = rng.standard_normal(int(bad.sum())); bad = (1 + sg * eps) <= 0
    return X * (1 + sg * eps), y


def kernel_bayes(Xobs, Tref, ref_idx, cols, sig, classes, chunk=300):
    cls, mg, obs, lam = Tref
    R = obs[np.ix_(ref_idx, np.arange(0, NM, 3))][:, :, cols].reshape(-1, len(cols)); ry = np.repeat(cls[ref_idx], len(np.arange(0, NM, 3)))
    sd = R * np.maximum(np.array(sig), 1e-4); ln = -np.log(sd).sum(1); out = np.zeros((len(Xobs), len(classes)))
    for s in range(0, len(Xobs), chunk):
        ll = (-0.5 * ((Xobs[s:s + chunk][:, None, :] - R[None]) / sd[None]) ** 2).sum(-1) + ln[None]
        for j, k in enumerate(classes): out[s:s + chunk, j] = logsumexp(ll[:, ry == k], axis=1) - np.log((ry == k).sum())
    return out


def task(direction, scen, seed):
    Ts, Tt = (TA, TB) if direction == "A->B" else (TB, TA)
    cols, sig = SCEN[scen]; nf = max(sig) == 0; rng = np.random.default_rng([99, seed, sum(map(ord, scen)), len(direction) + ord(direction[0])])
    classes = np.arange(4)
    itr, ite = np.arange(len(Ts[0])), np.arange(len(Tt[0]))
    Xtr, ytr = draw(Ts, itr, np.arange(0, NM, 2), 1 if nf else 3, cols, sig, rng)
    Xte, yte = draw(Tt, ite, np.arange(0, NM, 2), 1 if nf else 3, cols, sig, rng)
    sc = StandardScaler().fit(np.log(Xtr)); T = lambda X: sc.transform(np.log(X))
    pr = {}
    pr["LogReg"] = LogisticRegression(max_iter=2000).fit(T(Xtr), ytr).predict(T(Xte))
    pr["RandForest"] = RandomForestClassifier(200, min_samples_leaf=5, n_jobs=1, random_state=seed).fit(T(Xtr), ytr).predict(T(Xte))
    pr["HistGB"] = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(T(Xtr), ytr).predict(T(Xte))
    ens = [mlp(1000 * seed + j).fit(T(Xtr), ytr) for j in range(3)]
    pr["MLP-ens"] = classes[np.mean([e.predict_proba(T(Xte)) for e in ens], 0).argmax(1)]
    if not nf: pr["Bayes-kernel"] = classes[kernel_bayes(Xte, Ts, itr, cols, sig, classes).argmax(1)]
    rows = []
    for m, p in pr.items():
        cm = np.zeros((4, 4), int)
        for a, b in zip(yte, p): cm[a, b] += 1
        rows.append(dict(direction=direction, scen=scen, seed=seed, model=m, acc=float((p == yte).mean()), cm=cm.tolist()))
    return rows


if __name__ == "__main__":
    print("A:", len(RA), "EoS; B:", len(RB), "EoS", flush=True)
    jobs = [(d, s, sd) for d in ("A->B", "B->A") for s in SC for sd in range(3)]
    out = Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 14, verbose=5)(delayed(task)(*j) for j in jobs)
    json.dump([r for o in out for r in o], open("results_cross_family.json", "w"))
    print("done")

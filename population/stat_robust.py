"""EoS-level statistical robustness: cluster (EoS-level) bootstrap intervals and learning curves versus the number of training EoS."""
import numpy as np, json, sys, warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pop_experiment as PE
from pop_experiment import SCEN, CLS, NM, folds, draw, bayes_kernel, mlp, K_FOLDS
warnings.filterwarnings("ignore")
SCENS = ["ALL@0", "NICER+f20", "SO@1.0", "SO@0.25"]
FRACS = [0.25, 0.5, 0.75, 1.0]


def task(scen, rep, fold, frac, seed0=4242):
    cols, sig = SCEN[scen]; noise_free = max(sig) == 0
    rng = np.random.default_rng([seed0, rep, fold, int(frac * 100), sum(map(ord, scen))])
    part = folds(CLS, np.random.default_rng([777, rep]))              # same partitions as the main experiment
    te = np.where(part == fold)[0]
    pool = np.where((part != fold) & (part != (fold + 1) % K_FOLDS))[0]
    tr = np.sort(rng.choice(pool, int(round(frac * len(pool))), replace=False))
    classes = np.unique(CLS)
    Xtr, ytr, _ = draw(tr, np.arange(0, NM, 2), 1 if noise_free else 3, cols, sig, rng)
    Xte, yte, gte = draw(te, np.arange(0, NM, 2), 1 if noise_free else 3, cols, sig, rng)
    sc = StandardScaler().fit(np.log(Xtr)); T = lambda X: sc.transform(np.log(X))
    preds = {}
    m = LogisticRegression(max_iter=2000).fit(T(Xtr), ytr); preds["LogReg"] = m.predict(T(Xte))
    m = RandomForestClassifier(200, min_samples_leaf=5, n_jobs=1, random_state=seed0).fit(T(Xtr), ytr); preds["RandForest"] = m.predict(T(Xte))
    if not noise_free:
        lp = bayes_kernel(Xte, tr, np.arange(0, NM, 3), cols, sig, classes); preds["Bayes-kernel"] = classes[lp.argmax(1)]
    if frac in (0.5, 1.0):
        ens = [mlp(seed0 + 10 * j + fold).fit(T(Xtr), ytr) for j in range(3)]
        preds["MLP-ens"] = classes[np.mean([e.predict_proba(T(Xte)) for e in ens], 0).argmax(1)]
    rows = []
    for mname, p in preds.items():
        for g in np.unique(gte):
            s = gte == g
            rows.append(dict(scen=scen, rep=rep, fold=fold, frac=frac, model=mname, eos=int(g), cls=int(CLS[g]), n=int(s.sum()),
                             correct=int((p[s] == yte[s]).sum()), pred=[int((p[s] == k).sum()) for k in range(4)]))
    return rows


if __name__ == "__main__":
    jobs = [(s, r, f, fr) for s in SCENS for r in range(PE.N_REP) for f in range(K_FOLDS) for fr in FRACS]
    out = Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 14, verbose=5)(delayed(task)(*j) for j in jobs)
    rows = [r for o in out for r in o]
    json.dump(rows, open("results_stat_robust.json", "w"))
    print("done", len(rows))

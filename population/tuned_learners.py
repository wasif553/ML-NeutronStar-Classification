"""Does tuning the learners, or giving them physically motivated features, break the ~65% plateau?
Whole-EoS hold-out as in the main experiment (5 folds: test / calibration / train by EoS). For each learner a small hyper-parameter grid is
fitted on the training EoS and the configuration with the best accuracy on the *calibration* EoS is applied to the test EoS ('selected');
the best test accuracy over the whole grid ('best on test', optimistic because it is chosen on the test fold) is reported as an upper bound
on what tuning could give. Two feature sets: raw log(M,R,f,tau) and 'engineered' features (ln M, ln R, ln[f/sqrt(M/R^3)], ln[tau M^3/R^4])
that remove the leading Newtonian scalings. Scenarios: noise-free and errors 0.25 x (5,5,2,20)%."""
import json, sys, warnings, zlib, numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from pop_data_utils import load_all, tables, NM
warnings.filterwarnings("ignore")
RECS = load_all(); CLS, MG, OBS, LAM = tables(RECS)
MASS_IDX = np.arange(0, NM, 2)
SCEN = {"noise-free": np.zeros(4), "s=0.25": 0.25 * np.array([0.05, 0.05, 0.02, 0.20])}

GRID = {
    "LogReg": [dict(C=c) for c in (0.1, 1, 10, 100)],
    "kNN": [dict(k=k) for k in (5, 25, 100)],
    "RandForest": [dict(leaf=l, mf=m) for l in (1, 5, 20) for m in ("sqrt", None)],
    "HistGB": [dict(depth=d, lr=lr, it=it) for d in (3, None) for lr in (0.05, 0.1) for it in (150, 400)],
    "SVM": [dict(C=c, g=g) for c in (1, 10, 100) for g in (0.1, 0.5, "scale")],
    "MLP": [dict(h=h, a=a) for h in ((64, 64), (128, 128, 128), (256, 256)) for a in (1e-4, 1e-2)],
}


def make(name, p, seed):
    if name == "LogReg": return LogisticRegression(C=p["C"], max_iter=3000)
    if name == "kNN": return KNeighborsClassifier(p["k"])
    if name == "RandForest": return RandomForestClassifier(200, min_samples_leaf=p["leaf"], max_features=p["mf"], n_jobs=1, random_state=seed)
    if name == "HistGB": return HistGradientBoostingClassifier(max_depth=p["depth"], learning_rate=p["lr"], max_iter=p["it"], random_state=seed)
    if name == "SVM": return SVC(C=p["C"], gamma=p["g"])
    if name == "MLP": return MLPClassifier(p["h"], alpha=p["a"], early_stopping=True, max_iter=200, random_state=seed)


def stars(idx, sig, rng, nd):
    X = OBS[np.ix_(idx, MASS_IDX)].reshape(-1, 4); y = np.repeat(CLS[idx], len(MASS_IDX))
    X = np.repeat(X, nd, 0); y = np.repeat(y, nd)
    if sig.max() > 0:
        e = rng.standard_normal(X.shape); bad = (1 + sig * e) <= 0
        while bad.any():
            e[bad] = rng.standard_normal(int(bad.sum())); bad = (1 + sig * e) <= 0
        X = X * (1 + sig * e)
    return X, y


def feats(X, kind):
    M, R, f, t = X.T
    if kind == "raw": return np.log(X)
    return np.column_stack([np.log(M), np.log(R), np.log(f / np.sqrt(M / R ** 3)), np.log(t * M ** 3 / R ** 4)])


def one(scen, kind, rep, fold):
    rng = np.random.default_rng([zlib.crc32(scen.encode()), rep, 7]); sig = SCEN[scen]
    part = np.zeros(len(CLS), int)
    for k in range(4):
        ii = np.where(CLS == k)[0]; part[rng.permutation(ii)] = np.arange(len(ii)) % 5
    te, ca = np.where(part == fold)[0], np.where(part == (fold + 1) % 5)[0]; tr = np.where((part != fold) & (part != (fold + 1) % 5))[0]
    nd = 1 if sig.max() == 0 else 3
    Xtr, ytr = stars(tr, sig, rng, nd); Xca, yca = stars(ca, sig, rng, 1); Xte, yte = stars(te, sig, rng, 1)
    sc = StandardScaler().fit(feats(Xtr, kind)); A, B, C = (sc.transform(feats(x, kind)) for x in (Xtr, Xca, Xte))
    if len(A) > 6000: sub = rng.choice(len(A), 6000, replace=False)
    out = {}
    for name, grid in GRID.items():
        res = []
        for p in grid:
            Atr, ytr_ = (A[sub], ytr[sub]) if (name == "SVM" and len(A) > 6000) else (A, ytr)
            m = make(name, p, rep).fit(Atr, ytr_)
            res.append(((m.predict(B) == yca).mean(), m.predict(C)))
        j = int(np.argmax([r[0] for r in res]))
        accs = [(r[1] == yte).mean() for r in res]
        pred = res[j][1]; msk = np.isin(yte, (0, 2)) & np.isin(pred, (0, 2)); pair = float((pred[msk] == yte[msk]).mean())
        out[name] = dict(sel=float(accs[j]), best=float(max(accs)), default=float(accs[0]), pair=pair, cfg=str(grid[j]))
    return dict(scen=scen, kind=kind, rep=rep, fold=fold, res=out)


if __name__ == "__main__":
    nj = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [(s, k, r, f) for s in SCEN for k in ("raw", "eng") for r in range(2) for f in range(5)]
    out = Parallel(n_jobs=nj, verbose=5)(delayed(one)(*j) for j in jobs)
    json.dump(out, open("results_tuned.json", "w"), indent=1)
    for s in SCEN:
        for k in ("raw", "eng"):
            R = [o for o in out if o["scen"] == s and o["kind"] == k]
            print(s, k, {n: (round(100 * np.mean([o["res"][n]["sel"] for o in R]), 1), round(100 * np.mean([o["res"][n]["best"] for o in R]), 1), round(100 * np.mean([o["res"][n]["pair"] for o in R]), 1)) for n in GRID})

"""Nucleonic-versus-dark-matter accuracy as the nuclear EoS is assumed to be known more precisely (EoS drawn from windows about the SLy fit, gen_narrow.py).
Reports, for each window fraction w: number of EoS, spread of f at 1.4 Msun among the nucleonic EoS, and the balanced accuracy of two-class classifiers
(whole-EoS 5-fold CV, 10 repetitions) for noise-free data and for errors s x (5,5,2,20)%."""
import json, sys, warnings, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pop_data_utils as U
warnings.filterwarnings("ignore")
NPC = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SCEN = {"noise-free": np.zeros(4), "s=0.25": 0.25 * np.array([0.05, 0.05, 0.02, 0.20]), "s=1": np.array([0.05, 0.05, 0.02, 0.20])}
MODELS = {"LogReg": lambda s: LogisticRegression(max_iter=3000), "RandForest": lambda s: RandomForestClassifier(200, min_samples_leaf=3, n_jobs=2, random_state=s),
          "HistGB": lambda s: HistGradientBoostingClassifier(random_state=s)}
out = {}
for w in (0.4, 0.2, 0.1, 0.05, 0.02):
    if not __import__('os').path.isdir(f'pop_narrow/w{int(round(100*w)):03d}'): continue
    U.N_PER_CLASS = NPC; recs = U.load_all(f"pop_narrow/w{int(round(100*w)):03d}")
    cls, mg, obs, lam = U.tables(recs); nm = len(mg); MI = np.arange(0, nm, 2); j14 = int(np.argmin(abs(mg - 1.4)))
    n0, n2 = int((cls == 0).sum()), int((cls == 2).sum()); f14 = obs[cls == 0, j14, 2]
    row = dict(w=w, n_nuc=n0, n_dm=n2, sigma_f14=float(np.std(f14) / np.mean(f14)), acc={})
    for sc, sig in SCEN.items():
        accs = {m: [] for m in MODELS}
        for rep in range(10):
            rng = np.random.default_rng([int(100 * w), rep]); fold = np.zeros(len(cls), int)
            for c in (0, 2):
                ii = np.where(cls == c)[0]; fold[rng.permutation(ii)] = np.arange(len(ii)) % 5
            for k in range(5):
                te, tr = np.where(fold == k)[0], np.where(fold != k)[0]
                def draw(idx, nd):
                    X = np.repeat(obs[np.ix_(idx, MI)].reshape(-1, 4), nd, 0); y = np.repeat(np.repeat(cls[idx], len(MI)), nd)
                    if sig.max() > 0:
                        e = rng.standard_normal(X.shape); bad = (1 + sig * e) <= 0
                        while bad.any(): e[bad] = rng.standard_normal(int(bad.sum())); bad = (1 + sig * e) <= 0
                        X = X * (1 + sig * e)
                    return X, y
                Xtr, ytr = draw(tr, 1 if sig.max() == 0 else 3); Xte, yte = draw(te, 1)
                sc_ = StandardScaler().fit(np.log(Xtr)); A, B = sc_.transform(np.log(Xtr)), sc_.transform(np.log(Xte))
                for m, mk in MODELS.items():
                    p = mk(rep).fit(A, ytr).predict(B); accs[m].append(np.mean([np.mean(p[yte == c] == c) for c in (0, 2)]))
        row["acc"][sc] = {m: float(np.mean(v)) for m, v in accs.items()}
    out[str(w)] = row; print(w, n0, n2, round(100 * row["sigma_f14"], 2), {s: {m: round(100 * a, 0) for m, a in v.items()} for s, v in row["acc"].items()}, flush=True)
# main population for reference (w = 1)
U.N_PER_CLASS = 90; recs = U.load_all("pop_data"); cls, mg, obs, lam = U.tables(recs); j14 = int(np.argmin(abs(mg - 1.4)))
out["main"] = dict(sigma_f14=float(np.std(obs[cls == 0, j14, 2]) / np.mean(obs[cls == 0, j14, 2])))
json.dump(out, open("results_narrow.json", "w"), indent=1)

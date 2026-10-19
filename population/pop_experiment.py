"""Population experiment: classification of EoS *populations* with whole-EoS hold-out (grouped cross-validation)."""
import numpy as np, pandas as pd, json, warnings, sys, time, zlib, os
from joblib import Parallel, delayed
from scipy.special import logsumexp, softmax
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from pop_data_utils import *
from unc import *
from group_conformal import grouped_mondrian
warnings.filterwarnings("ignore")

RECS = load_all(); CLS, MG, OBS, LAM = tables(RECS)
K_FOLDS, N_REP = 5, 3
# observational-error scenarios: (name, feature columns [M, R, f, tau], fractional 1-sigma errors)
BASE = np.array([0.05, 0.05, 0.02, 0.20])
SCEN = {"NICER": ([0, 1], [0.10, 0.09]), "NICER+f20": ([0, 1, 2], [0.10, 0.09, 0.20])}
for s in (0.25, 0.5, 1.0, 2.0, 4.0):
    for nm, cols in (("S", [0, 1]), ("SF", [0, 1, 2]), ("SO", [0, 1, 2, 3])):
        if nm != "SO" and s in (0.5, 2.0): continue          # full dial only for the complete observable set
        SCEN[f"{nm}@{s}"] = (cols, list(s * BASE[cols]))
for nm, cols in (("S", [0, 1]), ("SF", [0, 1, 2]), ("SO", [0, 1, 2, 3])):
    SCEN[f"{nm}@0"] = (cols, [0.0] * len(cols))
SCEN["ALL@0"] = ([0, 1, 2, 3], [0.0] * 4)


def folds(cls, rng):
    f = np.zeros(len(cls), int)
    for k in np.unique(cls):
        idx = rng.permutation(np.where(cls == k)[0]); f[idx] = np.arange(len(idx)) % K_FOLDS
    return f


def draw(idx_eos, mass_idx, ndraw, cols, sig, rng):
    """observations (rows) for the given EoS: all listed mass indices x ndraw noise draws"""
    X = OBS[np.ix_(idx_eos, mass_idx)][:, :, cols]                       # (n_eos, n_m, d)
    X = np.repeat(X.reshape(-1, len(cols)), ndraw, 0)
    y = np.repeat(np.repeat(CLS[idx_eos], len(mass_idx)), ndraw)
    g = np.repeat(np.repeat(idx_eos, len(mass_idx)), ndraw)
    sg = np.array(sig); eps = rng.standard_normal(X.shape); bad = (1 + sg * eps) <= 0
    while bad.any():                       # redraw non-positive factors (truncated Gaussian); no-op unless sigma is large
        eps[bad] = rng.standard_normal(int(bad.sum())); bad = (1 + sg * eps) <= 0
    X = X * (1 + sg * eps)
    return X, y, g


def bayes_kernel(Xobs, ref_eos, mass_idx, cols, sig, classes, chunk=400):
    """Kernel-Bayes classifier: class-conditional density = mean over (EoS, mass) reference points of the Gaussian noise model."""
    R = OBS[np.ix_(ref_eos, mass_idx)][:, :, cols].reshape(-1, len(cols))
    ry = np.repeat(CLS[ref_eos], len(mass_idx))
    sig = np.maximum(np.array(sig), 1e-4)
    out = np.zeros((len(Xobs), len(classes)))
    sd = R * sig
    lognorm = -np.log(sd).sum(1)
    for s in range(0, len(Xobs), chunk):
        xo = Xobs[s:s + chunk][:, None, :]
        ll = (-0.5 * ((xo - R[None]) / sd[None]) ** 2).sum(-1) + lognorm[None]
        for j, k in enumerate(classes):
            m = ry == k
            out[s:s + chunk, j] = logsumexp(ll[:, m], axis=1) - np.log(m.sum())
    out -= logsumexp(out, axis=1, keepdims=True)
    return out


def mlp(seed): return MLPClassifier((64, 64), alpha=1e-4, batch_size=128, learning_rate_init=1e-3, max_iter=120, early_stopping=True,
                                    n_iter_no_change=10, random_state=seed)


def task(scen, rep, fold, seed0=777):
    cols, sig = SCEN[scen]
    rng = np.random.default_rng([seed0, rep, fold, zlib.crc32(scen.encode()) % 10_000])
    part = folds(CLS, np.random.default_rng([seed0, rep]))
    te = np.where(part == fold)[0]; ca = np.where(part == (fold + 1) % K_FOLDS)[0]
    tr = np.where((part != fold) & (part != (fold + 1) % K_FOLDS))[0]
    classes = np.unique(CLS)
    noise_free = max(sig) == 0
    nd_tr = 1 if noise_free else 3
    Xtr, ytr, _ = draw(tr, np.arange(0, NM, 2), nd_tr, cols, sig, rng)
    Xca, yca, gca = draw(ca, np.arange(0, NM, 2), 1 if noise_free else 2, cols, sig, rng)
    Xte, yte, gte = draw(te, np.arange(0, NM, 2), 1 if noise_free else 3, cols, sig, rng)
    sc = StandardScaler().fit(np.log(Xtr)); T = lambda X: sc.transform(np.log(X))
    rows = []
    base = dict(scen=scen, rep=rep, fold=fold)
    # probabilistic MLP ensemble
    ens = [mlp(seed0 + 10 * j + fold).fit(T(Xtr), ytr) for j in range(3)]
    P = lambda X: np.stack([m.predict_proba(T(X)) for m in ens])
    Pc, Pt = P(Xca), P(Xte); pc, pt = Pc.mean(0), Pt.mean(0)
    NE = len(ca)
    r = metrics(pt, yte, pc, yca, NE); r["MI"] = float((ent(pt) - ent(Pt).mean(0)).mean())
    rows.append({**base, "model": "MLP-ens", **r})
    Tt = fit_T(np.log(np.clip(pc, 1e-12, None)), yca)
    ptT = softmax(np.log(np.clip(pt, 1e-12, None)) / Tt, 1); pcT = softmax(np.log(np.clip(pc, 1e-12, None)) / Tt, 1)
    rows.append({**base, "model": "MLP-ens+T", **metrics(ptT, yte, pcT, yca, NE), "T": Tt})
    # Proper class-conditional conformal calibration: exactly one independently
    # selected representative stellar observation per calibration EoS.
    # This is distinct from the pooled-score, empirically calibrated legacy estimate.
    group_res = grouped_mondrian(pcT, yca, gca, ptT, yte, alpha=0.10,
                                  seed=seed0 + 1000 * rep + 20 * fold + zlib.crc32(scen.encode()) % 100000)
    rows.append({**base, "model": "MLP-ens+T-Mondrian", **metrics(ptT, yte),
                 "cov": group_res["coverage"], "setsize": group_res["mean_set_size"],
                 **{f"cov_class_{k}": v for k, v in group_res["coverage_by_class"].items()},
                 "ncal_min": min(group_res["calibration_eos_per_class"]),
                 "ncal_max": max(group_res["calibration_eos_per_class"])})
    cm = np.zeros((4, 4), int)
    for a, b in zip(yte, ptT.argmax(1)): cm[a, b] += 1
    rel = {"MLP-ens": reliab(pt, yte), "MLP-ens+T": reliab(ptT, yte)}
    # other learners
    for nm, m in (("LogReg", LogisticRegression(max_iter=2000)), ("kNN", KNeighborsClassifier(25)),
                  ("RandForest", RandomForestClassifier(200, min_samples_leaf=5, n_jobs=1, random_state=seed0)),
                  ("HistGB", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed0))):
        m.fit(T(Xtr), ytr); pp = m.predict_proba(T(Xte))
        rows.append({**base, "model": nm, **metrics(pp, yte)})
    sub = rng.choice(len(Xtr), min(8000, len(Xtr)), replace=False)
    svm = SVC(C=3.0, gamma="scale").fit(T(Xtr)[sub], ytr[sub])
    rows.append({**base, "model": "SVM-RBF", "acc": float((svm.predict(T(Xte)) == yte).mean())})
    # kernel-Bayes using the same training EoS (exact noise model); needs sigma > 0
    if not noise_free:
        ref_mass = np.arange(0, NM, 3)
        lpt = bayes_kernel(Xte, tr, ref_mass, cols, sig, classes); lpc = bayes_kernel(Xca, tr, ref_mass, cols, sig, classes)
        rows.append({**base, "model": "Bayes-kernel", **metrics(np.exp(lpt), yte, np.exp(lpc), yca, NE)})
        rel["Bayes-kernel"] = reliab(np.exp(lpt), yte)
        # larger reference: every EoS except the test fold
        ref2 = np.where(part != fold)[0]
        lpt2 = bayes_kernel(Xte, ref2, ref_mass, cols, sig, classes)
        rows.append({**base, "model": "Bayes-kernel(all other EoS)", **metrics(np.exp(lpt2), yte)})
    # per-EoS accuracy (to show EoS-level variability)
    return rows, dict(scen=scen, rep=rep, fold=fold, cm=cm.tolist(), rel=rel)


if __name__ == "__main__":
    print(len(RECS), "EoS;", {k: int((CLS == k).sum()) for k in range(4)}, flush=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    scen = [s for s in SCEN if which == "all" or s in which.split(",")]
    jobs = [(s, rep, f) for s in scen for rep in range(N_REP) for f in range(K_FOLDS)]
    out = Parallel(n_jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 20, verbose=5)(delayed(task)(*j) for j in jobs)
    SUF = os.environ.get("OUT_SUFFIX", "")
    pd.DataFrame([r for o in out for r in o[0]]).to_csv(f"results_pop_main{SUF}.csv", index=False)
    json.dump([o[1] for o in out], open(f"results_pop_conf{SUF}.json", "w"))
    print("done")

"""Nucleonic-versus-dark-matter classification (noise-free), by stellar mass and by dark self-interaction strength G.
Two-class whole-EoS 5-fold cross-validation (10 repetitions); the test stars are restricted to a mass bin, or the dark-matter test EoS to a G tercile
(balanced accuracy = mean of the two recalls). Tests whether a subset of the population is separable even though the whole population is not."""
import json, warnings, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from pop_data_utils import load_all, tables, NM, M_LO, M_HI
warnings.filterwarnings("ignore")
RECS = load_all(); CLS, MG, OBS, LAM = tables(RECS)
sel = np.where(np.isin(CLS, (0, 2)))[0]
G = np.array([RECS[i]["par"].get("G", np.nan) for i in range(len(RECS))])
terc = np.nanpercentile(G[CLS == 2], [33.3, 66.7])
MI = np.arange(0, NM, 2)
MODELS = {"LogReg": lambda s: LogisticRegression(max_iter=3000), "RandForest": lambda s: RandomForestClassifier(200, min_samples_leaf=3, n_jobs=2, random_state=s),
          "HistGB": lambda s: HistGradientBoostingClassifier(random_state=s)}
BINS = {"all": (M_LO, M_HI + 1e-9), "1.20-1.40": (1.2, 1.4), "1.40-1.60": (1.4, 1.6), "1.60-1.85": (1.6, M_HI + 1e-9)}
res = {}
def bal(pred, y):
    r = [np.mean(pred[y == c] == c) for c in (0, 2) if (y == c).any()]; return float(np.mean(r))
acc = {(m, b): [] for m in MODELS for b in list(BINS) + ["G_low", "G_mid", "G_high"]}
for rep in range(10):
    rng = np.random.default_rng(100 + rep); fold = np.zeros(len(CLS), int)
    for c in (0, 2):
        ii = np.where(CLS == c)[0]; fold[rng.permutation(ii)] = np.arange(len(ii)) % 5
    for k in range(5):
        te = sel[fold[sel] == k]; tr = sel[fold[sel] != k]
        Xtr = np.log(OBS[np.ix_(tr, MI)].reshape(-1, 4)); ytr = np.repeat(CLS[tr], len(MI))
        sc = StandardScaler().fit(Xtr); A = sc.transform(Xtr)
        Xte = np.log(OBS[np.ix_(te, MI)].reshape(-1, 4)); yte = np.repeat(CLS[te], len(MI)); mte = np.tile(MG[MI], len(te)); gte = np.repeat(G[te], len(MI))
        B = sc.transform(Xte)
        for name, mk in MODELS.items():
            pred = mk(rep).fit(A, ytr).predict(B)
            for b, (lo, hi) in BINS.items():
                m = (mte >= lo) & (mte < hi); acc[(name, b)].append(bal(pred[m], yte[m]))
            for b, cond in (("G_low", gte <= terc[0]), ("G_mid", (gte > terc[0]) & (gte <= terc[1])), ("G_high", gte > terc[1])):
                m = (yte == 0) | cond; acc[(name, b)].append(bal(pred[m], yte[m]))
out = {f"{m}|{b}": [float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v)))] for (m, b), v in acc.items()}
json.dump(out, open("results_pair_subsets.json", "w"), indent=1)
for b in list(BINS) + ["G_low", "G_mid", "G_high"]:
    print(b, {m: round(100 * out[f"{m}|{b}"][0], 1) for m in MODELS})
print("G terciles", terc)

"""E1: re-run the ORIGINAL protocol (random stratified split, 2x64 MLP, 7 raw columns incl. redshift)
over many seeds; plus M-only / M+R baselines and a blocked-split variant."""
import numpy as np, pandas as pd, json, warnings
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from common import *
warnings.filterwarnings("ignore")
d = load()
RAW = ["M", "R", "f", "Q", "z", "tau", "hc"]

def mlp(seed): return MLPClassifier((64, 64), alpha=1e-4, batch_size=64, learning_rate_init=1e-3, max_iter=500,
                                    early_stopping=True, n_iter_no_change=20, validation_fraction=.15, random_state=seed)
def one(seed, cols, model):
    X = d[cols].values.astype(float); y = d.y.values
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=.15, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtv)
    m = {"mlp": mlp(seed), "logreg": LogisticRegression(max_iter=3000),
         "rf": RandomForestClassifier(300, random_state=seed), "knn": KNeighborsClassifier(5)}[model]
    m.fit(sc.transform(Xtv), ytv)
    return float((m.predict(sc.transform(Xte)) == yte).mean())

res = {}
sets = {"paper_all7": RAW, "no_redshift": [c for c in RAW if c != "z"],
        "M_only": ["M"], "M_R": ["M", "R"], "M_R_f_tau": ["M", "R", "f", "tau"]}
N = 100
for nm, cols in sets.items():
    for model in (["mlp", "logreg", "rf", "knn"] if nm in ("paper_all7", "no_redshift", "M_only", "M_R") else ["mlp"]):
        a = np.array(Parallel(n_jobs=6)(delayed(one)(s, cols, model) for s in range(N)))
        res[f"{nm}|{model}"] = dict(mean=a.mean(), sd=a.std(ddof=1), min=a.min(), max=a.max(), q05=np.quantile(a, .05), q95=np.quantile(a, .95),
                                    frac_ge_974=float((a >= .974).mean()), seed42=float(a[42]), all=a.tolist())
        print(nm, model, round(a.mean(), 4), round(a.std(ddof=1), 4), flush=True)
json.dump(res, open("results/repro.json", "w"), indent=1)

"""E2: noise-free accuracy vs size of the contiguous mass block held out (autocorrelation leakage)."""
import numpy as np, pandas as pd, warnings
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from common import *
warnings.filterwarnings("ignore")
D = load()
SETS = {"M,R,f,tau": ["M", "R", "f", "tau"], "all six (no z)": ["M", "R", "f", "Q", "tau", "hc"]}

def task(block, rep, fs, model):
    rng = np.random.default_rng(100 * rep + block); d = D; role = block_split(d, rng, block=block, frac=(.7, 0., .3))
    cols = SETS[fs]; f = lambda x: np.column_stack([np.log(np.abs(x[:, i]) + 1e-30) if c not in ("Q",) else np.arcsinh(x[:, i] / 1e30) for i, c in enumerate(cols)])
    Xtr, Xte = f(d[cols].values[role == 0]), f(d[cols].values[role == 2]); ytr, yte = d.y.values[role == 0], d.y.values[role == 2]
    sc = StandardScaler().fit(Xtr)
    m = {"kNN": KNeighborsClassifier(5), "RF": RandomForestClassifier(200, random_state=rep, n_jobs=1),
         "MLP": MLPClassifier((64, 64), max_iter=500, early_stopping=True, n_iter_no_change=20, random_state=rep)}[model]
    m.fit(sc.transform(Xtr), ytr)
    return dict(block=block, rep=rep, feats=fs, model=model, acc=float((m.predict(sc.transform(Xte)) == yte).mean()))

jobs = [(b, r, fs, m) for b in (1, 2, 3, 6, 12, 24) for r in range(15) for fs in SETS for m in ("kNN", "RF", "MLP")]
out = Parallel(n_jobs=6)(delayed(task)(*j) for j in jobs)
pd.DataFrame(out).to_csv("results/blocksize.csv", index=False)
print(pd.DataFrame(out).groupby(["feats", "model", "block"]).acc.mean().unstack("block").round(3))

import numpy as np, pandas as pd, json
from common import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors

d = load(); out = {}
out["n"] = int(len(d))
out["classes"] = {NAMES[k]: dict(n=int((d.y == k).sum()), Mmin=float(d[d.y == k].M.min()), Mmax=float(d[d.y == k].M.max()),
                  Rmin=float(d[d.y == k].R.min()), Rmax=float(d[d.y == k].R.max()),
                  frac_below_05=float((d[d.y == k].M < 0.5).mean()), frac_below_1=float((d[d.y == k].M < 1).mean()),
                  n_ge_05=int(((d.y == k) & (d.M >= .5)).sum()), n_ge_1=int(((d.y == k) & (d.M >= 1)).sum())) for k in range(4)}
d["zres"] = np.log(d.z / z_from_MR(d.M, d.R).clip(lower=1e-12))
out["zres"] = {NAMES[k]: dict(median=float(d[d.y == k].zres.median()), p05=float(d[d.y == k].zres.quantile(.05)), p95=float(d[d.y == k].zres.quantile(.95)))
               for k in range(4)}
# leakage: can the redshift-consistency residual alone separate {nuc,strange} from {DM,hyperon}?
g = d.y.isin([2, 3]).astype(int)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
for nm, cols in [("zres_only", ["zres"]), ("M_only", ["M"]), ("M_R", ["M", "R"])]:
    X = d[cols].values
    if nm != "zres_only": X = np.log(X)
    out["leak_" + nm] = float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, g, cv=cv).mean())
# resolution of the radius grid and nearest-neighbour interpolation distance
out["R_grid_step"] = float(np.unique(np.round(np.diff(d[d.y == 0].R.values), 3))[0])
Z = StandardScaler().fit_transform(np.column_stack([np.log(d[c]) for c in ["M", "R", "f", "tau"]]))
dist, idx = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
out["nn_same_class"] = float((d.y.values[idx[:, 1]] == d.y.values).mean())
out["nn_median_dist"] = float(np.median(dist[:, 1]))
json.dump(out, open("results/audit.json", "w"), indent=1)
print(json.dumps(out, indent=1))
d.to_pickle("results/data.pkl")

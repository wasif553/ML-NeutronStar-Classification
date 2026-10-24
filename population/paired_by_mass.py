"""Fractional shifts of f and R caused by the dark component at fixed nuclear EoS, as a function of stellar mass (paired stars)."""
import glob, json, numpy as np
from scipy.interpolate import PchipInterpolator
from pop_data_utils import load_all, M_LO, M_HI, NM
MG = np.linspace(M_LO, M_HI, NM); dm = {r["idx"]: r for r in load_all() if r["cls"] == 2}
df, dR = [], []
for fn in sorted(glob.glob("pop_base/b*.json")):
    b = json.load(open(fn)); r = dm.get(b["idx"])
    if r is None or len(b["masses"]) < 5 or max(b["masses"]) < M_HI - 1e-9: continue
    cur = lambda rec, key: np.exp(PchipInterpolator(rec["masses"], np.log(rec[key]))(MG))
    df.append((cur(r, "f") - cur(b, "f")) / cur(b, "f")); dR.append((cur(r, "R") - cur(b, "R")) / cur(b, "R"))
df, dR = np.array(df), np.array(dR); M = {}
for m in (1.2, 1.4, 1.6, 1.8):
    j = int(np.argmin(abs(MG - m))); M[f"bm_f_{int(m*10)}"] = f"{100*np.median(df[:, j]):.2f}"; M[f"bm_R_{int(m*10)}"] = f"{100*np.median(dR[:, j]):.2f}"
json.dump(M, open("macros_bymass.json", "w"), indent=1); print(len(df), M)

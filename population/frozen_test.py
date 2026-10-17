"""Sensitivity of the dark-matter f-mode shift to the adiabatic response: equilibrium barotrope vs frozen composition."""
import json, numpy as np, sys
from scipy.optimize import brentq
from joblib import Parallel, delayed
from families import *
import fmode2 as F
from fmode import Background
from pop_data_utils import load_all


def f14(eos, guess, frozen):
    F.FROZEN = frozen
    try:
        lm = max_mass(eos)[1]; lpc = pc_for_mass(eos, 1.4, lm); bg = Background(eos, np.exp(lpc))
        return F.find_mode(bg, guess, 0.2)[0]
    finally:
        F.FROZEN = False


def one(rec, base_rec):
    p = rec["par"]
    base = read_pp(p["logP1"], p["G1"], p["G2"], p["G3"]); eos = dm_admixed(base, p["G"], p["mchi"])
    f_base = float(np.interp(1.4, base_rec["masses"], base_rec["f"]))
    f_eq = f14(eos, f_base * 1.005, False); f_fr = f14(eos, f_base * 1.005, True)
    return dict(idx=rec["idx"], G=p["G"], f_base=f_base, f_eq=f_eq, f_frozen=f_fr)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    dm = [r for r in load_all() if r["cls"] == 2][:n]
    bases = {int(fn.split("b")[-1][:4]): json.load(open(fn)) for fn in __import__("glob").glob("pop_base/b*.json")}
    res = Parallel(n_jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 8)(delayed(one)(r, bases[r["idx"]]) for r in dm)
    out = dict(rows=res)
    sh_eq = np.array([(r["f_eq"] / r["f_base"] - 1) for r in res]); sh_fr = np.array([(r["f_frozen"] / r["f_base"] - 1) for r in res])
    d = sh_fr - sh_eq
    out["summary"] = dict(n=len(res), shift_eq_med=float(np.median(sh_eq)), shift_frozen_med=float(np.median(sh_fr)),
                          dshift_med=float(np.median(d)), dshift_max=float(np.max(abs(d))), ratio_med=float(np.median(abs(d) / abs(sh_eq))))
    json.dump(out, open("results_frozen.json", "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))

"""For every dark-matter-admixed EoS of the population, compute the same observables for its *own nuclear base EoS*
(no dark component), on the same mass grid: this gives exactly paired (no DM, DM) stars."""
import numpy as np, json, os, sys, time
from joblib import Parallel, delayed
from families import *
from fmode import Background
import fmode2 as F
from pop_data_utils import load_all

OUT = "pop_base"


def one(rec):
    fn = f"{OUT}/b{rec['idx']:04d}.json"
    if os.path.exists(fn): return fn
    p = rec["par"]
    base = read_pp(p["logP1"], p["G1"], p["G2"], p["G3"])
    Mmax, lpcm = max_mass(base)
    out = dict(idx=rec["idx"], par=p, Mmax_base=Mmax, masses=[], R=[], f=[], tau=[], Lam=[], fail=[])
    for M, fdm in zip(rec["masses"], rec["f"]):
        if M > Mmax - 0.03: continue
        try:
            lpc = pc_for_mass(base, M, lpcm)
            bg = Background(base, np.exp(lpc)); s = solve_star(base, np.exp(lpc))
            ok = False
            for mult in (0.99, 1.0, 0.97, 1.02):
                try:
                    f, tau, w = F.find_mode(bg, fdm * mult, 0.2)
                except Exception:
                    continue
                if 0.8 * fdm < f < 1.25 * fdm and 0.03 < tau < 3: ok = True; break
            if not ok: out["fail"].append(M); continue
            out["masses"].append(M); out["R"].append(s["R"]); out["f"].append(f); out["tau"].append(tau); out["Lam"].append(s["Lambda"])
        except Exception:
            out["fail"].append(M)
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(fn, "w"))
    return fn


if __name__ == "__main__":
    recs = [r for r in load_all() if r["cls"] == 2]
    print(len(recs), "dark-matter EoS", flush=True)
    Parallel(n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 12, verbose=5)(delayed(one)(r) for r in recs)
    print("done")

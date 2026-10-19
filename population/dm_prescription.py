"""Sensitivity of the dark-matter frequency shift to the baryon-number prescription of the effective one-fluid barotrope.
The baryon number density of the fitted polytropic EoS is n_B = rho / m0; the baryon chemical potential mu_N = (eps+p)/n_B then tends
to m0 at zero density, which sets where mu_N exceeds m_chi and the dark component appears."""
import json, sys, numpy as np
from joblib import Parallel, delayed
from families import *
import fmode2 as F
from fmode import Background
from pop_data_utils import load_all

M0S = (931.494, 931.192, 935.0, 938.9)


def f14(eos, guess):
    lm = max_mass(eos)[1]; lpc = pc_for_mass(eos, 1.4, lm); bg = Background(eos, np.exp(lpc))
    return F.find_mode(bg, guess, 0.2)[0]


def one(rec, base_rec):
    p = rec["par"]; base = read_pp(p["logP1"], p["G1"], p["G2"], p["G3"])
    f_base = float(np.interp(1.4, base_rec["masses"], base_rec["f"])); out = dict(idx=rec["idx"], G=p["G"], mchi=p["mchi"], f_base=f_base)
    for m0 in M0S:
        try:
            e = dm_admixed(base, p["G"], p["mchi"], m0_mev=m0); out[f"f_{m0}"] = f14(e, f_base * 1.005)
        except Exception as ex:
            out[f"f_{m0}"] = float("nan")
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    dm = [r for r in load_all() if r["cls"] == 2][:n]
    bases = {int(fn.split("b")[-1][:4]): json.load(open(fn)) for fn in __import__("glob").glob("pop_base/b*.json")}
    res = Parallel(n_jobs=int(sys.argv[2]) if len(sys.argv) > 2 else 8)(delayed(one)(r, bases[r["idx"]]) for r in dm)
    summ = {}
    for m0 in M0S:
        sh = np.array([(r[f"f_{m0}"] / r["f_base"] - 1) for r in res]); sh = sh[np.isfinite(sh)]
        summ[str(m0)] = dict(median=float(np.median(sh)), p5=float(np.percentile(sh, 5)), p95=float(np.percentile(sh, 95)), n=int(len(sh)))
        print(m0, summ[str(m0)])
    # EoS-by-EoS change relative to the default prescription
    d = np.array([(r["f_938.9"] - r["f_931.494"]) / (r["f_931.494"] - r["f_base"]) for r in res if np.isfinite(r["f_938.9"])])
    summ["rel_change_938.9"] = dict(median=float(np.median(d)), max_abs=float(np.max(abs(d))))
    d2 = np.array([(r["f_931.192"] - r["f_931.494"]) / (r["f_931.494"] - r["f_base"]) for r in res if np.isfinite(r["f_931.192"])])
    summ["rel_change_931.192"] = dict(median=float(np.median(d2)), max_abs=float(np.max(abs(d2))))
    json.dump(dict(rows=res, summary=summ), open("results_dm_prescription.json", "w"), indent=1)
    print(summ["rel_change_938.9"])

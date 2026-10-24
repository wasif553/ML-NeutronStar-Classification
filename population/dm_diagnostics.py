"""Where does the dark component appear? For every dark-matter EoS of the primary population: the baryon density at which the dark component first
appears, the central density of the 1.4 and 1.8 Msun stars, and the dark share of the central energy density (diagnostic of the one-fluid model)."""
import json, numpy as np
from joblib import Parallel, delayed
from families import *
from pop_data_utils import load_all

NSAT = 2.7e14      # g cm^-3, nuclear saturation (reference scale for reporting only)


def one(rec):
    p = rec["par"]; base = read_pp(p["logP1"], p["G1"], p["G2"], p["G3"]); eos = dm_admixed(base, p["G"], p["mchi"]); d = eos._diag
    lm = max_mass(eos)[1]; out = dict(idx=rec["idx"], G=p["G"], mchi=p["mchi"])
    on = np.where(d["nchi"] > 0)[0]; out["rho_onset"] = float(d["rho"][on[0]]) if len(on) else float("nan")
    for M in (1.4, 1.8):
        pc = np.exp(pc_for_mass(eos, M, lm)); i = int(np.searchsorted(d["p_g"], pc)); i = min(i, len(d["rho"]) - 1)
        out[f"rho_c_{M}"] = float(d["rho"][i]); out[f"dark_core_{M}"] = bool(d["nchi"][i] > 0)
        out[f"eps_dark_frac_{M}"] = float((d["e_tot"][i] - d["eN"][i]) / d["e_tot"][i])
    return out


if __name__ == "__main__":
    res = Parallel(n_jobs=4)(delayed(one)(r) for r in load_all() if r["cls"] == 2)
    A = lambda k: np.array([r[k] for r in res], float)
    M = {"dgN": len(res)}
    for m in (1.4, 1.8):
        M[f"dgCore_{str(m).replace('.','')}"] = f"{100*np.mean([r[f'dark_core_{m}'] for r in res]):.0f}"
        M[f"dgFracMed_{str(m).replace('.','')}"] = f"{100*np.median(A(f'eps_dark_frac_{m}')):.2f}"; M[f"dgFracMax_{str(m).replace('.','')}"] = f"{100*np.max(A(f'eps_dark_frac_{m}')):.1f}"
    r_on = A("rho_onset") / 1e11; M["dgOnsetMed"] = f"{np.nanmedian(r_on):.1f}"; M["dgOnsetLo"] = f"{np.nanmin(r_on):.1f}"; M["dgOnsetHi"] = f"{np.nanmax(r_on):.1f}"
    M["dgRhoC14Med"] = f"{np.median(A('rho_c_1.4'))/NSAT:.2f}"; M["dgRhoC18Med"] = f"{np.median(A('rho_c_1.8'))/NSAT:.2f}"
    json.dump(dict(rows=res, macros=M), open("results_dm_diagnostics.json", "w"), indent=1); json.dump(M, open("macros_dmdiag.json", "w"), indent=1); print(M)

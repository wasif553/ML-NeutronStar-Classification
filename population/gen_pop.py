"""Generate the population data set: for each sampled EoS (accepted by the observational screen) compute R, Lambda, f and tau
on a mass grid with the validated TOV + full-GR f-mode solver."""
import numpy as np, json, os, sys, time, traceback
from joblib import Parallel, delayed
from families import *
from fmode import Background
import fmode2 as F

FAM = os.environ.get("POP_FAMILY", "A")
if FAM == "B":
    import families_b as FB
    sample_params, build = FB.sample_params, FB.build
OUT = "pop_data" if FAM == "A" else "pop_data_" + FAM
K2HZ = 2 * np.pi / F.C_KMS
MASSES = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])


def ak_guess(M, R):
    return 1e3 * (0.79 + 1.50 * np.sqrt((M / 1.4) / (R / 10.0) ** 3))


def one_eos(cls, idx, seed0=20261017):
    rng = np.random.default_rng([seed0, cls, idx])
    t0 = time.time()
    fn = f"{OUT}/c{cls}_{idx:04d}.json"
    if os.path.exists(fn): return fn
    tries, why = 0, {}
    while True:
        tries += 1
        par = sample_params(cls, rng)
        try:
            eos = build(par)
            sc = screen(eos)
        except Exception as e:
            continue
        if sc is None: continue
        if sc["ok"]: break
        why[sc["why"]] = why.get(sc["why"], 0) + 1
        if tries > 400: raise RuntimeError("no acceptable EoS for class %d" % cls)
    Mmax, lpcm = sc["Mmax"], sc["lpcmax"]
    masses = [m for m in MASSES if m <= Mmax - 0.03]
    rec = dict(cls=cls, idx=idx, par=par, Mmax=Mmax, R14=sc["R14"], Lam14=sc["Lam14"], tries=tries, reject=why, masses=[], R=[], f=[], tau=[], Lam=[], fail=[])
    f_prev = Mprev = Rprev = None
    for M in masses:
        try:
            lpc = pc_for_mass(eos, M, lpcm)
            bg = Background(eos, np.exp(lpc))
            s = solve_star(eos, np.exp(lpc))
            R = s["R"]
            if f_prev is None: g0 = ak_guess(M, R)
            else: g0 = f_prev * ak_guess(M, R) / ak_guess(Mprev, Rprev)
            ok = False
            for mult in (1.0, 0.97, 1.03, 0.94, 1.06):
                try:
                    f, tau, w = F.find_mode(bg, g0 * mult, 0.2)
                except Exception:
                    continue
                if np.isfinite(f) and 0.7 * g0 < f < 1.35 * g0 and 0.03 < tau < 3.0:
                    ok = True; break
            if not ok:
                rec["fail"].append(float(M)); continue
            rec["masses"].append(float(M)); rec["R"].append(float(R)); rec["f"].append(float(f)); rec["tau"].append(float(tau)); rec["Lam"].append(float(s["Lambda"]))
            f_prev, Mprev, Rprev = f, M, R
        except Exception:
            rec["fail"].append(float(M))
    rec["seconds"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(fn, "w"))
    return fn


if __name__ == "__main__":
    n = int(sys.argv[1]); jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    os.makedirs(OUT, exist_ok=True)
    work = [(c, i) for i in range(n) for c in range(4)]
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(one_eos)(c, i) for c, i in work)
    print("done", len(res))

"""Nucleonic and dark-matter-admixed EoS drawn from *narrowed* nuclear priors: the four polytropic parameters are restricted to the fraction w of their main-prior half-widths about a fiducial (SLy) nuclear EoS (same restriction for both classes). Used to measure how the nucleonic-versus-dark-matter accuracy depends on how well the nuclear EoS is
known (Sec. 'Narrowed nuclear priors'). Same solver and screens as the main population; five masses per EoS."""
import numpy as np, os, sys
from joblib import Parallel, delayed

RANGES = dict(logP1=(34.15, 34.60), G1=(2.4, 3.8), G2=(2.4, 3.9), G3=(2.2, 3.9))     # main prior (sets the half-widths)
CENTRE = dict(logP1=34.384, G1=3.005, G2=2.988, G3=2.851)      # SLy piecewise-polytrope fit (Read et al. 2009), a viable fiducial nuclear EoS
MASSES = [1.2, 1.4, 1.6, 1.8, 1.9]


def sample_w(cls, rng, w):
    d = {}
    for k, (lo, hi) in RANGES.items():
        c, h = CENTRE[k], w * (hi - lo) / 2
        d[k] = rng.uniform(c - h, c + h)
    d["kind"] = "pp"
    if cls == 2:
        d.update(kind="dm", G=rng.uniform(50, 500), mchi=rng.uniform(937.9, 938.783))
    return d


def task(w, cls, idx):
    import gen_pop as G
    G.sample_params = lambda c, r: sample_w(c, r, w)
    G.MASSES = np.array(MASSES)
    G.OUT = f"pop_narrow/w{int(round(100 * w)):03d}"
    os.makedirs(G.OUT, exist_ok=True)
    try:
        return G.one_eos(cls, idx, seed0=20261024 + int(round(100 * w)))
    except RuntimeError:
        return None


if __name__ == "__main__":
    n = int(sys.argv[1]); jobs = int(sys.argv[2]); ws = [float(x) for x in sys.argv[3].split(",")]
    work = [(w, c, i) for i in range(n) for w in ws for c in (0, 2)]
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(task)(*j) for j in work)
    print("done", len(res))

"""Load the population data set and build dense (EoS, mass) observable tables."""
import glob, json, os, numpy as np
from scipy.interpolate import PchipInterpolator

NAMES = {0: "Nucleonic", 1: "Strange", 2: "Dark-matter", 3: "Hyperonic"}
M_LO, M_HI, NM = 1.2, 1.85, 66            # observable mass window (Msun) and number of grid masses


POP_DIR = os.environ.get("POP_DIR", "pop_data")
N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 90))                           # frozen, class-balanced population (lowest sampling indices)


def load_all(path=None):
    path = path or POP_DIR
    recs = [json.load(open(f)) for f in sorted(glob.glob(path + "/c*.json"))]
    recs = [r for r in recs if len(r["masses"]) >= 5 and max(r["masses"]) >= M_HI - 1e-9]
    mchi_max = float(os.environ.get("MCHI_MAX", "inf"))        # optional: restrict the dark-fermion mass (stability window m_chi < m_p + m_e = 938.783 MeV)
    recs = [r for r in recs if not (r["cls"] == 2 and r["par"].get("mchi", 0) > mchi_max)]
    out = []
    for k in range(4):
        out += sorted([r for r in recs if r["cls"] == k], key=lambda r: r["idx"])[:N_PER_CLASS]
    return out


def tables(recs):
    """returns cls (N,), mgrid (NM,), obs (N, NM, 4) with columns (M, R, f[Hz], tau[s]) and Lambda (N, NM)."""
    mg = np.linspace(M_LO, M_HI, NM)
    obs = np.zeros((len(recs), NM, 4)); lam = np.zeros((len(recs), NM)); cls = np.zeros(len(recs), int)
    for i, r in enumerate(recs):
        m = np.array(r["masses"])
        obs[i, :, 0] = mg
        for j, key in ((1, "R"), (2, "f"), (3, "tau")):
            obs[i, :, j] = np.exp(PchipInterpolator(m, np.log(r[key]))(mg))
        lam[i] = PchipInterpolator(m, np.log(r["Lam"]))(mg)
        lam[i] = np.exp(lam[i]); cls[i] = r["cls"]
    return cls, mg, obs, lam

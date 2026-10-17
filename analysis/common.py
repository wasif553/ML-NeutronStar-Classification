"""Shared utilities for the revised neutron-star EoS classification study."""
import numpy as np, pandas as pd
from scipy.special import logsumexp

NAMES = {0: "Nucleonic", 1: "Strange", 2: "Dark-matter", 3: "Hyperonic"}
G, C, MSUN = 6.67430e-11, 2.99792458e8, 1.98847e30

# Observable sets used in the noise study (Q and h_c are model-derived amplitudes, not
# independently measurable here, so they are only used in the noise-free analyses).
FEATS = {"S": ["M", "R"], "SF": ["M", "R", "f"], "SO": ["M", "R", "f", "tau"]}
REL_SIG = {"M": 0.05, "R": 0.05, "f": 0.02, "tau": 0.20}   # base (scale s = 1) 1-sigma fractional errors
REGIMES = {"A": 0.0, "B": 0.5, "C": 1.0}                    # lower mass cut in Msun


def load(path="DataML.xlsx"):
    d = pd.read_excel(path)
    d.columns = ["M", "R", "f", "Q", "z", "tau", "hc", "y"]
    d["y"] = d["y"].astype(int)
    return d.sort_values(["y", "M"]).reset_index(drop=True)


def z_from_MR(M, R):
    """Exact Schwarzschild surface redshift implied by (M, R)."""
    comp = G * M * MSUN / (C ** 2 * R * 1e3)
    return (1 - 2 * comp) ** -0.5 - 1


def regime(d, name):
    return d[d.M >= REGIMES[name]].reset_index(drop=True)


def logfeat(X):
    """Log-transform positive observables (relative errors are ~Gaussian in log space)."""
    return np.log(np.clip(X, 1e-30, None))


def block_split(d, rng, block=6, frac=(0.6, 0.2, 0.2)):
    """Assign contiguous mass blocks (within each class) at random to train / cal / test.
    Returns an integer role array: 0 train, 1 calibration, 2 test."""
    role = np.zeros(len(d), int)
    for k in sorted(d.y.unique()):
        idx = np.where(d.y.values == k)[0]            # already sorted by mass within class
        nb = int(np.ceil(len(idx) / block))
        b = rng.permutation(nb)
        cut = (int(round(frac[0] * nb)), int(round((frac[0] + frac[1]) * nb)))
        r = np.empty(nb, int)
        r[b[:cut[0]]] = 0; r[b[cut[0]:cut[1]]] = 1; r[b[cut[1]:]] = 2
        role[idx] = np.repeat(r, block)[: len(idx)]
    return role


def noisy_copies(d, cols, scale, reps, rng):
    """Replicate rows `reps` times with multiplicative Gaussian noise; returns (X, y, idx)."""
    base = d[cols].values.astype(float)
    sig = scale * np.array([REL_SIG[c] for c in cols])
    X = np.repeat(base, reps, 0)
    X = X * (1 + sig * rng.standard_normal(X.shape))
    return X, np.repeat(d.y.values, reps), np.repeat(np.arange(len(d)), reps)


def bayes_logpost(Xobs, grid, gy, cols, scale, classes, chunk=2000):
    """Oracle Bayes classifier: class-conditional likelihood is the mean over that class's
    grid points of the heteroscedastic Gaussian noise model; equal class priors."""
    sig = scale * np.array([REL_SIG[c] for c in cols])
    out = np.zeros((len(Xobs), len(classes)))
    for s in range(0, len(Xobs), chunk):
        xo = Xobs[s:s + chunk][:, None, :]
        sd = grid[None, :, :] * sig
        ll = -0.5 * ((xo - grid[None]) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)
        ll = ll.sum(-1)                                    # (n, m)
        for j, k in enumerate(classes):
            m = gy == k
            out[s:s + chunk, j] = logsumexp(ll[:, m], axis=1) - np.log(m.sum())
    out -= logsumexp(out, axis=1, keepdims=True)
    return out

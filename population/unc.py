"""Calibration and conformal-prediction utilities (shared with the single-sequence study)."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

ALPHA = 0.10


def ece(p, y, nb=15):
    conf, pred = p.max(1), p.argmax(1); acc = (pred == y).astype(float); e = 0.
    edges = np.linspace(0, 1, nb + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def reliab(p, y, nb=10):
    conf, ok = p.max(1), (p.argmax(1) == y).astype(float); edges = np.linspace(0, 1, nb + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append([int(m.sum()), float(conf[m].sum()), float(ok[m].sum())])
    return out


def nll(p, y): return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
def brier(p, y): oh = np.eye(p.shape[1])[y]; return float(((p - oh) ** 2).sum(1).mean())
def ent(p): return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def fit_T(logp, y):
    f = lambda t: nll(softmax(logp / np.exp(t), 1), y)
    return float(np.exp(minimize_scalar(f, bounds=(-2, 3), method="bounded").x))


def conformal(pc, yc, pt, yt, alpha=ALPHA, n_eff=None):
    """Split conformal prediction (LAC score).  n_eff is the number of exchangeable calibration units (EoS)."""
    s = 1 - pc[np.arange(len(yc)), yc]; n = n_eff or len(s)
    q = np.quantile(s, min(1, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = pt >= 1 - q
    return float(sets[np.arange(len(yt)), yt].mean()), float(sets.sum(1).mean()), sets


def metrics(p, y, pc=None, yc=None, n_eff=None):
    out = dict(acc=float((p.argmax(1) == y).mean()), nll=nll(p, y), brier=brier(p, y), ece=float(ece(p, y)), H=float(ent(p).mean()))
    if pc is not None:
        out["cov"], out["setsize"], _ = conformal(pc, yc, p, y, n_eff=n_eff)
    return out

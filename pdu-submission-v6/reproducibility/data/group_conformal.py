"""Class-conditional split conformal prediction with EoS as the independent sampling unit.

For each calibration EoS, draw exactly one stellar observation (mass/noise
draw) using a prespecified RNG. Condition on the EoS class and calibrate
separate nonconformity quantiles for each class. In a population where EoS
draws are IID within classes, this gives marginal coverage for a *random*
future stellar observation from a new EoS of the same class (conditional on
the fitted model and the stated sampling design).

It does NOT promise simultaneous coverage of every mass on a new EoS,
nor coverage under another population prior or an actual NICER posterior.
"""
import numpy as np


def grouped_mondrian(pc, yc, gc, pt, yt, alpha=0.10, seed=20261019):
    pc = np.asarray(pc, dtype=float)
    yc = np.asarray(yc, dtype=int)
    gc = np.asarray(gc)
    pt = np.asarray(pt, dtype=float)
    yt = np.asarray(yt, dtype=int)
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1")
    if len(pc) != len(yc) or len(yc) != len(gc) or len(pt) != len(yt):
        raise ValueError("mismatched observation and label lengths")
    if pc.ndim != 2 or pt.ndim != 2 or pc.shape[1] != pt.shape[1]:
        raise ValueError("calibration/test prediction shapes must agree")
    rng = np.random.default_rng(seed)
    sampled = []
    for group in np.unique(gc):
        indices = np.flatnonzero(gc == group)
        if len(np.unique(yc[indices])) != 1:
            raise ValueError("A calibration EoS has inconsistent class labels")
        sampled.append(int(rng.choice(indices)))
    sampled = np.asarray(sampled, dtype=int)
    labels = yc[sampled]
    scores = 1.0 - pc[sampled, labels]
    thresholds = np.empty(pc.shape[1], dtype=float)
    n_by_class = []
    for k in range(pc.shape[1]):
        x = np.sort(scores[labels == k])
        n = len(x)
        if n == 0:
            raise ValueError(f"No calibration EoS in class {k}")
        # Mondrian split conformal: quantile of n scores at rank
        # ceil((n+1)*(1-alpha)); infinity when rank exceeds n.
        rank = int(np.ceil((n + 1) * (1 - alpha)))
        thresholds[k] = np.inf if rank > n else x[rank - 1]
        n_by_class.append(n)
    pred_sets = (1.0 - pt) <= thresholds[None, :]
    valid = pred_sets[np.arange(len(yt)), yt]
    by_class = {str(k): float(valid[yt == k].mean()) if np.any(yt == k) else None
                for k in range(pc.shape[1])}
    return {
        "coverage": float(valid.mean()),
        "mean_set_size": float(pred_sets.sum(axis=1).mean()),
        "coverage_by_class": by_class,
        "calibration_eos_per_class": n_by_class,
        "thresholds": thresholds.tolist(),
    }


if __name__ == "__main__":
    # Small deterministic smoke test and failure-mode checks.
    rng = np.random.default_rng(7)
    k, n_eos, stars = 4, 80, 5
    y = np.repeat(np.arange(k), n_eos // k)
    g = np.repeat(np.arange(n_eos), stars)
    yc = np.repeat(y, stars)
    pc = rng.dirichlet(np.ones(k), size=len(yc))
    yt = np.repeat(y, stars)
    pt = rng.dirichlet(np.ones(k), size=len(yt))
    r = grouped_mondrian(pc, yc, g, pt, yt)
    assert 0 <= r["coverage"] <= 1 and 0 <= r["mean_set_size"] <= k
    assert r["calibration_eos_per_class"] == [n_eos // k] * k
    assert r == grouped_mondrian(pc, yc, g, pt, yt)
    try:
        gc_bad = g.copy(); yc_bad = yc.copy(); yc_bad[1] = 2
        grouped_mondrian(pc, yc_bad, gc_bad, pt, yt)
        raise AssertionError("Mixed-class EoS must raise")
    except ValueError:
        pass
    print("group-conformal smoke tests passed", r)

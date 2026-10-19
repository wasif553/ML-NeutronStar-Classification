"""Figures, tables and number macros for the population study."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pop_data_utils import *

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {0: "#c0392b", 1: "#2c6fbb", 2: "#2e8b57", 3: "#e08e0b"}
FC = {"S": "#7f8c8d", "SF": "#2c6fbb", "SO": "#c0392b"}
FL = {"S": r"$M,R$", "SF": r"$M,R,f$", "SO": r"$M,R,f,\tau$"}
RECS = load_all(); CLS, MG, OBS, LAM = tables(RECS)
D = pd.read_csv("results_pop_main.csv"); CJ = json.load(open("results_pop_conf.json"))
M = {}
mac = lambda k, v: M.__setitem__(k, v)


def ci(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), (stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)


def rep_means(df, scen, model, col):
    """mean over folds within each repetition, then (mean, 95% t-interval) over repetitions"""
    x = df[(df.scen == scen) & (df.model == model)].groupby("rep")[col].mean()
    return ci(x.values)


# ---------------------------------------------------------------- data summary
rows = []
for k in range(4):
    r = [x for x in RECS if x["cls"] == k]
    f14 = [np.interp(1.4, x["masses"], x["f"]) for x in r]
    rows.append(f"{NAMES[k]} & {len(r)} & {min(x['Mmax'] for x in r):.2f}--{max(x['Mmax'] for x in r):.2f} & {min(x['R14'] for x in r):.1f}--{max(x['R14'] for x in r):.1f} & "
                f"{min(x['Lam14'] for x in r):.0f}--{max(x['Lam14'] for x in r):.0f} & {min(f14):.0f}--{max(f14):.0f} \\\\")
    mac(f"nEoS{k}", len(r))
mac("nEoS", len(RECS))
open("tables/tab_pop_data.tex", "w").write("\n".join(rows).rstrip()[:-2])

fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
for k in range(4):
    sel = CLS == k
    for a, j, yl in ((ax[0], 1, "R (km)"), (ax[1], 2, "f (Hz)")):
        lo, mid, hi = np.percentile(OBS[sel][:, :, j], [5, 50, 95], axis=0)
        a.fill_between(MG, lo, hi, color=COL[k], alpha=.25); a.plot(MG, mid, color=COL[k], label=NAMES[k]); a.set(xlabel=r"M ($M_\odot$)", ylabel=yl)
    ax[2].plot(OBS[sel][:, 18, 1], OBS[sel][:, 18, 2], ".", ms=4, color=COL[k], alpha=.6)
ax[0].set_title("(a) Radius, 5-95% band"); ax[1].set_title("(b) f-mode frequency"); ax[0].legend(frameon=False, fontsize=7)
ax[2].set(xlabel="R (km)", ylabel="f (Hz)", title=r"(c) EoS at $M\approx1.5\,M_\odot$")
fig.tight_layout(); fig.savefig("figs/figP1_population.pdf"); plt.close()

# ---------------------------------------------------------------- noise-free separability
mods = ["LogReg", "kNN", "RandForest", "HistGB", "SVM-RBF", "MLP-ens"]
fig, a0 = plt.subplots(figsize=(5.6, 3.2)); ax = [a0]
w = .13
for i, m in enumerate(mods):
    vals = [rep_means(D, s, m, "acc") for s in ("S@0", "SF@0", "SO@0")]
    ax[0].bar(np.arange(3) + (i - 2.5) * w, [v[0] for v in vals], w, yerr=[v[1] for v in vals], capsize=1.5, label=m)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([FL["S"], FL["SF"], FL["SO"]]); ax[0].axhline(.25, c="k", ls=":")
ax[0].set(ylabel="Held-out-EoS accuracy", title="Noise-free"); ax[0].legend(frameon=False, fontsize=6, ncol=2)
for m in mods:
    for s in ("S@0", "SF@0", "SO@0", "ALL@0"):
        a, h = rep_means(D, s, m, "acc"); mac(f"nf_{m}_{s.split('@')[0]}", f"{100*a:.0f}")
fig.tight_layout(); fig.savefig("figs/figP2_noisefree.pdf"); plt.close()

# ---------------------------------------------------------------- dial
SC = [0.25, 0.5, 1.0, 2.0, 4.0]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for a, mo in zip(ax, ["MLP-ens", "Bayes-kernel"]):
    for fs in ("S", "SF", "SO"):
        v = [rep_means(D, f"{fs}@{s}", mo, "acc") for s in SC]
        a.errorbar(range(5), [x[0] for x in v], [x[1] for x in v], marker="o", ms=4, capsize=2, color=FC[fs], label=FL[fs])
    a.axhline(.25, c="k", ls=":"); a.set_xticks(range(5)); a.set_xticklabels([str(s) for s in SC]); a.set(xlabel="Noise scale $s$ (1 = 5% $M$, 5% $R$, 2% $f$, 20% $\\tau$)", title=mo)
ax[0].set_ylabel("Held-out-EoS accuracy"); ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("figs/figP3_dial.pdf"); plt.close()
for fs in ("S", "SF", "SO"):
    for s in SC:
        for mo, tag in (("MLP-ens", "mlp"), ("Bayes-kernel", "bk"), ("LogReg", "lr"), ("RandForest", "rf"), ("HistGB", "hgb"), ("kNN", "knn"), ("SVM-RBF", "svm"), ("Bayes-kernel(all other EoS)", "bk2")):
            try: mac(f"pd_{fs}_{str(s).replace('.', 'p')}_{tag}", f"{100*rep_means(D, f'{fs}@{s}', mo, 'acc')[0]:.0f}")
            except Exception: pass

# ---------------------------------------------------------------- scenario table (literature-anchored + dial)
scen_rows = []
scen_list = [("NICER", r"NICER-like: $M,R$ at 10\%, 9\%"), ("NICER+f20", r"+ $f$-mode at 20\%"), ("SF@1.0", r"$s=1$: 5\%, 5\%, 2\%"), ("SO@1.0", r"$s=1$ with $\tau$ 20\%"), ("SO@0.25", r"$s=0.25$ (1.3\%, 1.3\%, 0.5\%, 5\%)")]
mcols = ["LogReg", "kNN", "RandForest", "HistGB", "SVM-RBF", "MLP-ens", "Bayes-kernel"]
for sc, lab in scen_list:
    cells = []
    for m in mcols:
        try:
            a, h = rep_means(D, sc, m, "acc"); cells.append(f"{100*a:.0f}$\\pm${100*h:.0f}")
        except Exception: cells.append("--")
    scen_rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
open("tables/tab_pop_scen.tex", "w").write("\n".join(scen_rows).rstrip()[:-2])
for sc in ("NICER", "NICER+f20"):
    for m, tag in (("MLP-ens", "mlp"), ("Bayes-kernel", "bk"), ("Bayes-kernel(all other EoS)", "bk2")):
        try: mac(f"sc_{sc.replace('+', 'p')}_{tag}", f"{100*rep_means(D, sc, m, 'acc')[0]:.0f}")
        except Exception: pass

# ---------------------------------------------------------------- confusion
def cmsum(scen):
    c = [o for o in CJ if o["scen"] == scen]
    return np.sum([np.array(o["cm"]) for o in c], 0)
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
for a, sc in zip(ax, ["ALL@0", "SO@0.25", "NICER+f20"]):
    cm = cmsum(sc); n = cm / cm.sum(1, keepdims=True)
    a.imshow(n, vmin=0, vmax=1, cmap="viridis"); a.grid(False)
    for i in range(4):
        for j in range(4): a.text(j, i, f"{100*n[i, j]:.0f}", ha="center", va="center", color="w" if n[i, j] < .6 else "k", fontsize=8)
    a.set_xticks(range(4)); a.set_yticks(range(4)); a.set_xticklabels([NAMES[c][:4] + "." for c in range(4)]); a.set_yticklabels([NAMES[c] for c in range(4)])
    a.set(xlabel="Predicted", title=f"{sc}: acc. {100*np.trace(cm)/cm.sum():.0f}%")
    for i in range(4):
        for j in range(4): mac(f"cm_{sc.replace('+', 'p').replace('@', '_').replace('.', 'p')}_{i}{j}", f"{100*n[i, j]:.0f}")
ax[0].set_ylabel("True class")
fig.tight_layout(); fig.savefig("figs/figP4_confusion.pdf"); plt.close()

# ---------------------------------------------------------------- uncertainty
U = {}
cols = ["ece", "nll", "cov", "setsize", "acc"]
CMAP = {"cov": "cov1", "setsize": "size1"}      # conformal sets: one star per calibration EoS (exchangeable units)
for fs in ("SO",):
    for s in SC:
        for mo in ("MLP-ens", "MLP-ens+T", "Bayes-kernel"):
            U[(s, mo)] = {c: rep_means(D, f"{fs}@{s}", mo, CMAP.get(c, c) if mo != "MLP-ens" else c) for c in cols}
            if mo != "MLP-ens": U[(s, mo)]["covP"] = rep_means(D, f"{fs}@{s}", mo, "cov"); U[(s, mo)]["sizeP"] = rep_means(D, f"{fs}@{s}", mo, "setsize")
for sc in ("NICER", "NICER+f20"):
    for mo in ("MLP-ens", "MLP-ens+T", "Bayes-kernel"):
        U[(sc, mo)] = {c: rep_means(D, sc, mo, CMAP.get(c, c) if mo != "MLP-ens" else c) for c in cols}
        if mo != "MLP-ens": U[(sc, mo)]["covP"] = rep_means(D, sc, mo, "cov"); U[(sc, mo)]["sizeP"] = rep_means(D, sc, mo, "setsize")
rows = []
for key in ["NICER", "NICER+f20"] + SC:
    e = U
    lab = {"NICER": "NICER-like", "NICER+f20": "+ $f$ at 20\\%"}.get(key, f"$s={key:g}$" if not isinstance(key, str) else key)
    rows.append(f"{lab} & {e[(key,'MLP-ens')]['ece'][0]:.3f} & {e[(key,'MLP-ens+T')]['ece'][0]:.3f} & {e[(key,'Bayes-kernel')]['ece'][0]:.3f} & "
                f"{e[(key,'MLP-ens')]['nll'][0]:.2f} & {e[(key,'MLP-ens+T')]['nll'][0]:.2f} & {e[(key,'Bayes-kernel')]['nll'][0]:.2f} & "
                f"{e[(key,'MLP-ens+T')]['cov'][0]:.2f} & {e[(key,'MLP-ens+T')]['setsize'][0]:.2f} & {e[(key,'Bayes-kernel')]['cov'][0]:.2f} & {e[(key,'Bayes-kernel')]['setsize'][0]:.2f} \\\\")
open("tables/tab_pop_unc.tex", "w").write("\n".join(rows).rstrip()[:-2])
for key in ["NICER", "NICER+f20", 1.0, 0.25]:
    kk = str(key).replace('+', 'p').replace('.', 'p')
    for mo, tg in (("MLP-ens", "r"), ("MLP-ens+T", "t"), ("Bayes-kernel", "b")):
        for c in cols: mac(f"pu_{kk}_{tg}_{c}", f"{U[(key, mo)][c][0]:.3f}" if c == "ece" else f"{U[(key, mo)][c][0]:.2f}")
        if mo != "MLP-ens":
            mac(f"puP_{kk}_{tg}_cov", f"{U[(key, mo)]['covP'][0]:.2f}"); mac(f"puP_{kk}_{tg}_setsize", f"{U[(key, mo)]['sizeP'][0]:.2f}")
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
def pool(model, sc):
    acc = np.zeros((10, 3))
    for o in CJ:
        if o["scen"] == sc and model in o["rel"]: acc += np.array(o["rel"][model])
    return acc
for mo, c, lb in [("MLP-ens", "#c0392b", "MLP ensemble"), ("MLP-ens+T", "#2c6fbb", "+ temperature scaling"), ("Bayes-kernel", "#2e8b57", "Kernel Bayes")]:
    p = pool(mo, "SO@1.0"); ok = p[:, 0] > 0
    ax[0].plot(p[ok, 1] / p[ok, 0], p[ok, 2] / p[ok, 0], "o-", ms=4, color=c, label=lb)
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set(xlabel="Mean confidence", ylabel="Empirical accuracy", title="(a) Reliability, $s=1$"); ax[0].legend(frameon=False, fontsize=7)
for mo, c, lb in [("MLP-ens+T", "#2c6fbb", "MLP + T"), ("Bayes-kernel", "#2e8b57", "Kernel Bayes")]:
    cov = [U[(s, mo)]["cov"] for s in SC]; sz = [U[(s, mo)]["setsize"] for s in SC]
    ax[1].errorbar(range(5), [x[0] for x in cov], [x[1] for x in cov], marker="o", ms=4, capsize=2, color=c, label=lb)
    ax[2].errorbar(range(5), [x[0] for x in sz], [x[1] for x in sz], marker="o", ms=4, capsize=2, color=c, label=lb)
ax[1].axhline(.9, c="k", ls=":"); ax[2].axhline(4, c="gray", ls=":")
for a, t, yl in ((ax[1], "(b) Conformal coverage (target 0.90)", "Coverage"), (ax[2], "(c) Mean set size (dotted: 4 classes)", "Set size")):
    a.set_xticks(range(5)); a.set_xticklabels([str(s) for s in SC]); a.set(xlabel="Noise scale $s$", title=t, ylabel=yl)
ax[1].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig("figs/figP5_uncertainty.pdf"); plt.close()
# merged-class and pairwise degeneracy summaries from the stored confusion matrices
for sc in ("ALL@0", "SO@0.25", "SO@1.0", "NICER+f20", "NICER"):
    cm = cmsum(sc).astype(float); key = sc.replace('+', 'p').replace('@', '_').replace('.', 'p')
    merged = [[0, 2], [1], [3]]
    ok = sum(cm[np.ix_(g, g)].sum() if len(g) > 1 else cm[g[0], g[0]] for g in merged) / cm.sum()
    # note: for the merged group the correct predictions are those landing anywhere inside the group
    ok = (cm[np.ix_([0, 2], [0, 2])].sum() + cm[1, 1] + cm[3, 3]) / cm.sum()
    pair = (cm[0, 0] + cm[2, 2]) / (cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2])
    mac(f"merged_{key}", f"{100*ok:.0f}"); mac(f"pairNucDM_{key}", f"{100*pair:.0f}")
json.dump(M, open("macros_pop.json", "w"), indent=1)
print(len(M), "macros")

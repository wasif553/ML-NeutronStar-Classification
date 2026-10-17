"""Turn saved results into figures, LaTeX tables and number macros (so no number in the paper is typed by hand)."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from common import *

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
FC = {"S": "#7f8c8d", "SF": "#2c6fbb", "SO": "#c0392b"}
FL = {"S": r"$M,R$", "SF": r"$M,R,f$", "SO": r"$M,R,f,\tau$"}
CL = [NAMES[k] for k in range(4)]
main = pd.read_csv("results/main.csv"); ext = pd.read_csv("results/extrap.csv"); blk = pd.read_csv("results/blocksize.csv")
rep = json.load(open("results/repro.json")); aud = json.load(open("results/audit.json")); conf = json.load(open("results/confusions.json"))
M = {}  # macros


def ci(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), (stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)

def agg(df, col):
    g = df.groupby(["regime", "feats", "scale", "model"])[col]
    out = g.apply(lambda x: pd.Series(dict(m=ci(x)[0], h=ci(x)[1]))).unstack()
    return out

def pm(m, h, d=3, pct=False):
    if pct: m, h = 100 * m, 100 * h; d = 1
    return f"{m:.{d}f}$\\pm${h:.{d}f}"

def mac(k, v): M[k] = v

def sci(x):
    """LaTeX scientific notation, e.g. 7.2e-08 -> $7.2\\times10^{-8}$."""
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"
SC = sorted(main.scale.unique())

# ------------------------------------------------------------------ audit table
rows = []
for k in range(4):
    c = aud["classes"][NAMES[k]]
    rows.append(f"{NAMES[k]} & {c['n']} & {sci(c['Mmin'])}--{c['Mmax']:.2f} & {c['Rmin']:.2f}--{c['Rmax']:.2f} & {100*c['frac_below_05']:.0f}\\% & {c['n_ge_05']} & {c['n_ge_1']} \\\\")
open("tables/tab_audit.tex", "w").write("\n".join(rows))
mac("nTot", aud["n"]); mac("leakZ", f"{100*aud['leak_zres_only']:.0f}"); mac("leakM", f"{100*aud['leak_M_only']:.0f}"); mac("leakMR", f"{100*aud['leak_M_R']:.0f}")
mac("nnSame", f"{100*aud['nn_same_class']:.1f}")
zr = aud["zres"]
mac("zresNuc", f"{100*np.expm1(zr['Nucleonic']['median']):.0f}"); mac("zresStr", f"{100*np.expm1(zr['Strange']['median']):.0f}")
mac("zresDM", f"{100*np.expm1(zr['Dark-matter']['median']):.3f}"); mac("zresHyp", f"{100*np.expm1(zr['Hyperonic']['median']):.3f}")

# ------------------------------------------------------------------ replication table
lab = [("paper_all7", "mlp", "Published setup: 7 columns incl.\\ $z$, MLP"), ("paper_all7", "rf", "Published setup, random forest"),
       ("no_redshift", "mlp", "Redshift removed, MLP"), ("no_redshift", "rf", "Redshift removed, random forest"),
       ("no_redshift", "knn", "Redshift removed, $k$-NN"), ("M_R_f_tau", "mlp", "$M,R,f,\\tau$ only, MLP"),
       ("M_R", "knn", "$M,R$ only, $k$-NN"), ("M_R", "logreg", "$M,R$ only, logistic regression"), ("M_only", "logreg", "$M$ only, logistic regression")]
rows = []
for a, b, t in lab:
    r = rep[f"{a}|{b}"]
    rows.append(f"{t} & {100*r['mean']:.1f} & {100*r['sd']:.1f} & {100*r['min']:.1f}--{100*r['max']:.1f} \\\\")
open("tables/tab_repro.tex", "w").write("\n".join(rows))
r = rep["paper_all7|mlp"]; mac("reproMean", f"{100*r['mean']:.1f}"); mac("reproSd", f"{100*r['sd']:.1f}"); mac("reproMin", f"{100*r['min']:.1f}"); mac("reproMax", f"{100*r['max']:.1f}")
mac("reproRF", f"{100*rep['paper_all7|rf']['mean']:.1f}"); mac("reproNoZ", f"{100*rep['no_redshift|mlp']['mean']:.1f}")
mac("reproMR", f"{100*rep['M_R|knn']['mean']:.1f}"); mac("reproMonly", f"{100*rep['M_only|logreg']['mean']:.1f}"); mac("reproMRft", f"{100*rep['M_R_f_tau|mlp']['mean']:.1f}")

# ------------------------------------------------------------------ block-size leakage
bs = blk.groupby(["feats", "model", "block"]).acc.apply(lambda x: pd.Series(dict(m=ci(x)[0], h=ci(x)[1]))).unstack()
fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
for a, fs in zip(ax, ["all six (no z)", "M,R,f,tau"]):
    for mo, c in zip(["kNN", "RF", "MLP"], ["#2c6fbb", "#2e8b57", "#c0392b"]):
        s = bs.loc[(fs, mo)]; a.errorbar(s.index, s.m, s.h, marker="o", ms=4, capsize=2, color=c, label=mo)
    a.set(xscale="log", xlabel="Held-out block length (grid points, $\\Delta R=0.05$ km each)", title={"all six (no z)": "(a) $M,R,f,Q,\\tau,h_c$", "M,R,f,tau": "(b) $M,R,f,\\tau$"}[fs])
    a.set_xticks([1, 2, 3, 6, 12, 24]); a.set_xticklabels([1, 2, 3, 6, 12, 24]); a.axhline(.25, c="gray", ls=":", lw=1)
ax[0].set_ylabel("Noise-free test accuracy"); ax[0].legend(frameon=False, fontsize=8); ax[1].set_ylim(0, 1.02)
fig.tight_layout(); fig.savefig("figs/fig3_blocksize.pdf"); plt.close()
for fs, key in [("all six (no z)", "six"), ("M,R,f,tau", "four")]:
    for mo in ("MLP", "RF", "kNN"):
        for b in (1, 6, 24):
            mac(f"blk{key}{mo}{b}", f"{100*bs.loc[(fs, mo, b)].m:.0f}")

# ------------------------------------------------------------------ noise study figure
A = agg(main[main.scale > 0], "acc"); A0 = agg(main[main.scale == 0], "acc")
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
xs = np.arange(len(SC))
for a, rg in zip(ax, "ABC"):
    K = 4 if rg != "C" else 3
    for fs in FEATS:
        m = [A0.loc[(rg, fs, 0.0, "MLP-ens")].m if s == 0 else A.loc[(rg, fs, s, "MLP-ens")].m for s in SC]
        h = [A0.loc[(rg, fs, 0.0, "MLP-ens")].h if s == 0 else A.loc[(rg, fs, s, "MLP-ens")].h for s in SC]
        a.errorbar(xs, m, h, color=FC[fs], marker="o", ms=4, capsize=2, label=f"MLP ensemble, {FL[fs]}")
        mo = [np.nan if s == 0 else A.loc[(rg, fs, s, "Bayes-oracle")].m for s in SC]
        a.plot(xs, mo, color=FC[fs], ls="--", lw=1.2, label=f"Oracle bound, {FL[fs]}" if rg == "A" else None)
    a.axhline(1 / K, c="k", ls=":", lw=1); a.set_xticks(xs); a.set_xticklabels([str(s) for s in SC])
    a.set(xlabel="Noise scale $s$ (1 = 5% $M$, 5% $R$, 2% $f$, 20% $\\tau$)", title={"A": "(a) All masses (as released)", "B": r"(b) $M\geq0.5\,M_\odot$", "C": r"(c) $M\geq1\,M_\odot$ (3 classes)"}[rg])
ax[0].set_ylabel("Test accuracy"); ax[0].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig("figs/fig4_noise.pdf"); plt.close()

# main results table
rows = []
for rg in "ABC":
    for fs in FEATS:
        for s in (0.25, 1.0, 4.0):
            cells = []
            for mo in ["MLP-ens", "LogReg", "RandForest", "Bayes-oracle"]:
                x = A.loc[(rg, fs, s, mo)]; cells.append(pm(x.m, x.h, pct=True))
            rows.append(f"{rg} & {FL[fs]} & {s:g} & " + " & ".join(cells) + r" \\")
    rows.append("\\midrule")
open("tables/tab_main.tex", "w").write("\n".join(rows[:-1]))
for rg in "ABC":
    for fs in FEATS:
        for s in (0.25, 1.0, 4.0):
            for mo, tag in [("MLP-ens", "mlp"), ("Bayes-oracle", "orc"), ("LogReg", "lr"), ("RandForest", "rf")]:
                mac(f"acc{rg}{fs}{str(s).replace('.', 'p')}{tag}", f"{100*A.loc[(rg, fs, s, mo)].m:.0f}")
for rg in "ABC":
    for fs in FEATS:
        x = A0.loc[(rg, fs, 0.0, "MLP-ens")]; mac(f"acc{rg}{fs}0mlp", f"{100*x.m:.0f}")

# ------------------------------------------------------------------ confusion matrices
def cmsum(rg, fs, s):
    c = [o for o in conf if o["regime"] == rg and o["feats"] == fs and abs(o["scale"] - s) < 1e-9]
    return np.sum([np.array(o["cm"]) for o in c], 0), c[0]["classes"]
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
for a, (rg, s) in zip(ax, [("B", 0.25), ("B", 1.0), ("C", 1.0)]):
    cm, cls = cmsum(rg, "SO", s); n = cm / cm.sum(1, keepdims=True)
    im = a.imshow(n, vmin=0, vmax=1, cmap="viridis"); a.grid(False)
    for i in range(len(cls)):
        for j in range(len(cls)): a.text(j, i, f"{100*n[i, j]:.0f}", ha="center", va="center", color="w" if n[i, j] < .6 else "k", fontsize=8)
    a.set_xticks(range(len(cls))); a.set_yticks(range(len(cls))); a.set_xticklabels([NAMES[c][:4] + "." for c in cls]); a.set_yticklabels([NAMES[c] for c in cls])
    a.set(xlabel="Predicted", title=f"Regime {rg}, $s={s:g}$: acc. {100*np.trace(cm)/cm.sum():.0f}%")
ax[0].set_ylabel("True class")
fig.tight_layout(); fig.savefig("figs/fig5_confusion.pdf"); plt.close()
cm, cls = cmsum("B", "SO", 1.0); n = cm / cm.sum(1, keepdims=True)
for i, ci_ in enumerate(cls):
    for j, cj in enumerate(cls): mac(f"cmB1_{ci_}{cj}", f"{100*n[i, j]:.0f}")

# ------------------------------------------------------------------ calibration + conformal
U = main[(main.feats == "SO") & (main.scale > 0)]
cols = ["ece", "nll", "cov", "setsize", "acc", "brier", "H"]
rows = []
Ucache = {}
for rg in "BC":
    for s in SC[1:]:
        cells = []
        for mo in ["MLP-ens", "MLP-ens+T", "Bayes-oracle"]:
            x = U[(U.regime == rg) & (U.scale == s) & (U.model == mo)]
            Ucache[(rg, s, mo)] = {c: ci(x[c]) for c in cols}
        e = Ucache
        rows.append(f"{rg} & {s:g} & {e[(rg,s,'MLP-ens')]['ece'][0]:.3f} & {e[(rg,s,'MLP-ens+T')]['ece'][0]:.3f} & {e[(rg,s,'Bayes-oracle')]['ece'][0]:.3f} & "
                    f"{e[(rg,s,'MLP-ens')]['nll'][0]:.2f} & {e[(rg,s,'MLP-ens+T')]['nll'][0]:.2f} & {e[(rg,s,'Bayes-oracle')]['nll'][0]:.2f} \\\\")
open("tables/tab_calib.tex", "w").write("\n".join(rows))
rows = []
for rg in "BC":
    for s in SC[1:]:
        e = Ucache
        rows.append(f"{rg} & {s:g} & {pm(*e[(rg,s,'MLP-ens+T')]['cov'], d=3)} & {e[(rg,s,'MLP-ens+T')]['setsize'][0]:.2f} & {pm(*e[(rg,s,'Bayes-oracle')]['cov'], d=3)} & {e[(rg,s,'Bayes-oracle')]['setsize'][0]:.2f} \\\\")
open("tables/tab_conf.tex", "w").write("\n".join(rows))
mac("covMin", f"{min(Ucache[(rg,s,'MLP-ens')]['cov'][0] for rg in 'B' for s in SC[1:]):.2f}")
for rg in "BC":
    for s in (0.25, 1.0):
        for mo, tag in [("MLP-ens", "r"), ("MLP-ens+T", "t"), ("Bayes-oracle", "o")]:
            for c in ("ece", "nll", "cov", "setsize"):
                mac(f"u{rg}{str(s).replace('.', 'p')}{tag}{c}", f"{Ucache[(rg, s, mo)][c][0]:.2f}" if c != "ece" else f"{Ucache[(rg, s, mo)][c][0]:.3f}")

# reliability diagram (regime B, SO, s=1), pooled across reps
def pool(model, rg="B", s=1.0):
    acc = np.zeros((10, 3))
    for o in conf:
        if o["regime"] == rg and o["feats"] == "SO" and abs(o["scale"] - s) < 1e-9 and model in o["rel"]: acc += np.array(o["rel"][model])
    return acc
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
for mo, c, lb in [("MLP-ens", "#c0392b", "MLP ensemble"), ("MLP-ens+T", "#2c6fbb", "+ temperature scaling"), ("Bayes-oracle", "#2e8b57", "Oracle Bayes")]:
    p = pool(mo); ok = p[:, 0] > 0
    ax[0].plot(p[ok, 1] / p[ok, 0], p[ok, 2] / p[ok, 0], "o-", ms=4, color=c, label=lb)
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set(xlabel="Mean confidence", ylabel="Empirical accuracy", title="(a) Reliability, regime B, $s=1$"); ax[0].legend(frameon=False, fontsize=7)
for mo, c, lb in [("MLP-ens", "#c0392b", "MLP ensemble"), ("MLP-ens+T", "#2c6fbb", "+ T-scaling"), ("Bayes-oracle", "#2e8b57", "Oracle Bayes")]:
    m = [Ucache[("B", s, mo)]["nll"][0] for s in SC[1:]]; h = [Ucache[("B", s, mo)]["nll"][1] for s in SC[1:]]
    ax[1].errorbar(range(len(SC) - 1), m, h, marker="o", ms=4, capsize=2, color=c, label=lb)
ax[1].axhline(np.log(4), c="gray", ls=":"); ax[1].text(0, np.log(4) + .02, "uniform", fontsize=7, color="gray")
ax[1].set_xticks(range(len(SC) - 1)); ax[1].set_xticklabels([str(s) for s in SC[1:]]); ax[1].set(xlabel="Noise scale $s$", ylabel="Test NLL (nats)", title="(b) Log-loss, regime B")
for mo, c, lb in [("MLP-ens+T", "#2c6fbb", "MLP + T-scaling"), ("Bayes-oracle", "#2e8b57", "Oracle Bayes")]:
    for rg, ls in [("B", "-"), ("C", "--")]:
        m = [Ucache[(rg, s, mo)]["cov"][0] for s in SC[1:]]; h = [Ucache[(rg, s, mo)]["cov"][1] for s in SC[1:]]
        ax[2].errorbar(range(len(SC) - 1), m, h, ls=ls, marker="o", ms=4, capsize=2, color=c, label=f"{lb}, regime {rg}")
ax[2].axhline(.9, c="k", ls=":"); ax[2].set_xticks(range(len(SC) - 1)); ax[2].set_xticklabels([str(s) for s in SC[1:]])
ax[2].set(xlabel="Noise scale $s$", ylabel="Conformal coverage (target 0.90)", title="(c) Marginal coverage"); ax[2].legend(frameon=False, fontsize=6.5)
fig.tight_layout(); fig.savefig("figs/fig6_uncertainty.pdf"); plt.close()

# set size figure
fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.1))
for a, rg in zip(ax, "BC"):
    for mo, c, lb in [("MLP-ens+T", "#2c6fbb", "MLP + T-scaling"), ("Bayes-oracle", "#2e8b57", "Oracle Bayes")]:
        m = [Ucache[(rg, s, mo)]["setsize"][0] for s in SC[1:]]; h = [Ucache[(rg, s, mo)]["setsize"][1] for s in SC[1:]]
        a.errorbar(range(len(SC) - 1), m, h, marker="o", ms=4, capsize=2, color=c, label=lb)
    a.axhline(4 if rg == "B" else 3, c="gray", ls=":"); a.set_xticks(range(len(SC) - 1)); a.set_xticklabels([str(s) for s in SC[1:]])
    a.set(xlabel="Noise scale $s$", ylabel="Mean prediction-set size", title=f"Regime {rg} ($M,R,f,\\tau$; dotted: all classes)")
ax[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig("figs/fig7_setsize.pdf"); plt.close()

# ------------------------------------------------------------------ extrapolation
E = ext.groupby(["feats", "test", "model"])[["acc", "H", "MI", "cov", "setsize"]].agg(lambda x: ci(x)[0])
Eh = ext.groupby(["feats", "test", "model"])[["acc", "H", "MI", "cov", "setsize"]].agg(lambda x: ci(x)[1])
rows = []
for fs in FEATS:
    for te, mo in [("in-distribution", "MLP-ens"), ("unseen mass window", "MLP-ens"), ("unseen mass window", "Bayes-oracle")]:
        e, h = E.loc[(fs, te, mo)], Eh.loc[(fs, te, mo)]
        mi = f"{e.MI:.3f}" if mo == "MLP-ens" else "--"
        rows.append(f"{FL[fs]} & {te} & {'MLP ens.' if mo=='MLP-ens' else 'Oracle'} & {pm(e.acc, h.acc, pct=True)} & {pm(e['cov'], h['cov'], d=2)} & {e.setsize:.2f} & {mi} \\\\")
open("tables/tab_extrap.tex", "w").write("\n".join(rows))
fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
w = .27
for i, (te, mo, c, lb) in enumerate([("in-distribution", "MLP-ens", "#2c6fbb", "MLP, held-out blocks"), ("unseen mass window", "MLP-ens", "#c0392b", "MLP, unseen window"), ("unseen mass window", "Bayes-oracle", "#2e8b57", "Oracle, unseen window")]):
    for a, col in zip(ax, ["acc", "cov", "MI"]):
        if col == "MI" and mo == "Bayes-oracle": continue
        a.bar(np.arange(3) + (i - 1) * w, [E.loc[(fs, te, mo)][col] for fs in FEATS], w, yerr=[Eh.loc[(fs, te, mo)][col] for fs in FEATS], color=c, label=lb, capsize=2)
for a, t in zip(ax, ["(a) Accuracy", "(b) Conformal coverage (target 0.90)", "(c) Ensemble mutual information (nats)"]):
    a.set_xticks(range(3)); a.set_xticklabels([FL[f] for f in FEATS]); a.set_title(t)
ax[1].axhline(.9, c="k", ls=":")
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(); fig.savefig("figs/fig8_extrap.pdf"); plt.close()
for fs in FEATS:
    for te, mo, tag in [("in-distribution", "MLP-ens", "in"), ("unseen mass window", "MLP-ens", "ex"), ("unseen mass window", "Bayes-oracle", "orc")]:
        e = E.loc[(fs, te, mo)]
        mac(f"x{fs}{tag}acc", f"{100*e.acc:.0f}"); mac(f"x{fs}{tag}cov", f"{e['cov']:.2f}"); mac(f"x{fs}{tag}mi", f"{e.MI:.3f}" if mo == "MLP-ens" else "")


# extra audit macros
ncl = aud["classes"]
mac("nGe05", sum(c["n_ge_05"] for c in ncl.values())); mac("nGe1", sum(c["n_ge_1"] for c in ncl.values()))
mac("nBelow1pct", f"{100*(1-sum(c['n_ge_1'] for c in ncl.values())/aud['n']):.0f}"); mac("nBelow05pct", f"{100*(1-sum(c['n_ge_05'] for c in ncl.values())/aud['n']):.0f}")
mac("nDM05", ncl["Dark-matter"]["n_ge_05"])
d_ = pd.read_pickle("results/data.pkl"); mac("minMass", sci(d_.M.min())[1:-1])
mac("nMainRows", len(main))
# ------------------------------------------------------------------ number macros
json.dump(M, open("results/macros.json", "w"), indent=1)
print(len(M), "macros")

import numpy as np, pandas as pd, json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
COL = {0: "#c0392b", 1: "#2c6fbb", 2: "#2e8b57", 3: "#e08e0b"}
d = pd.read_pickle("results/data.pkl")

# Fig 1: data as released -- M-R, mass coverage, redshift consistency
fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
for k in range(4):
    s = d[d.y == k]; ax[0].plot(s.R, s.M, color=COL[k], lw=1.8, label=NAMES[k])
    ax[1].hist(s.M, bins=np.linspace(0, 2.2, 23), histtype="step", color=COL[k], lw=1.6)
    ax[2].plot(s.M, np.abs(np.expm1(s.zres)), ".", ms=3, color=COL[k], label=NAMES[k])
ax[0].set(xlabel="R (km)", ylabel=r"M ($M_\odot$)", title="(a) Released M-R sequences"); ax[0].legend(frameon=False, fontsize=7)
ax[1].axvline(1, c="k", ls=":"); ax[1].set(xlabel=r"M ($M_\odot$)", ylabel="Number of models", title=r"(b) Mass coverage (dotted: 1 $M_\odot$)")
ax[2].set(xscale="log", yscale="log", xlabel=r"M ($M_\odot$)", ylabel=r"$|z_{\rm tab}/z(M,R)-1|$", title="(c) Tabulated vs. exact redshift")
fig.tight_layout(); fig.savefig("figs/fig1_data_audit.pdf"); plt.close()

# Fig 2: original protocol over seeds
r = json.load(open("results/repro.json"))
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
keys = [("paper_all7|mlp", "Paper setup\n(7 cols, MLP)"), ("no_redshift|mlp", "No redshift\n(MLP)"), ("no_redshift|rf", "No redshift\n(random forest)"),
        ("M_R|knn", "M, R only\n(k-NN)"), ("M_R_f_tau|mlp", r"$M,R,f,\tau$" + "\n(MLP)")]
ax[0].boxplot([r[k]["all"] for k, _ in keys], tick_labels=[l for _, l in keys], showfliers=False)
for i, (k, _) in enumerate(keys): ax[0].plot(i + 1, r[k]["seed42"], "r*", ms=7)
ax[0].set(ylabel="Test accuracy (100 random 70/15/15 splits)", title="(a) Random-split accuracy; star = seed 42"); ax[0].tick_params(axis="x", labelsize=7)
ax[1].hist(r["paper_all7|mlp"]["all"], bins=15, color="#2c6fbb"); ax[1].axvline(.974, c="r", ls="--", label="published 97.4%")
ax[1].set(xlabel="Test accuracy", ylabel="Splits", title="(b) Spread of the headline number (N=116 test)"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig("figs/fig2_repro.pdf"); plt.close()

"""Dark-matter detectability when the nuclear EoS is known exactly (paired stars) versus unknown (population)."""
import glob, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm
from pop_data_utils import load_all, M_LO, M_HI, NM

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
MG = np.linspace(M_LO, M_HI, NM)
dm = {r["idx"]: r for r in load_all() if r["cls"] == 2}
pairs = []
for fn in sorted(glob.glob("pop_base/b*.json")):
    b = json.load(open(fn)); r = dm.get(b["idx"])
    if r is None or len(b["masses"]) < 5 or max(b["masses"]) < M_HI - 1e-9: continue
    cur = lambda rec, key: np.exp(PchipInterpolator(rec["masses"], np.log(rec[key]))(MG))
    pairs.append(dict(idx=b["idx"], G=r["par"]["G"], mchi=r["par"]["mchi"], f_dm=cur(r, "f"), f_b=cur(b, "f"), R_dm=cur(r, "R"), R_b=cur(b, "R")))
n = len(pairs); M = {}
df = np.array([(p["f_dm"] - p["f_b"]) / p["f_b"] for p in pairs]); dR = np.array([(p["R_dm"] - p["R_b"]) / p["R_b"] for p in pairs])
M["pairN"] = n
M["pairDfMed"] = f"{100*np.median(df):.2f}"; M["pairDfP5"] = f"{100*np.percentile(df, 5):.2f}"; M["pairDfP95"] = f"{100*np.percentile(df, 95):.2f}"
M["pairDRMed"] = f"{100*np.median(dR):.2f}"; M["pairDRP5"] = f"{100*np.percentile(dR, 5):.2f}"; M["pairDRP95"] = f"{100*np.percentile(dR, 95):.2f}"
M["pairDfPos"] = f"{100*(df > 0).mean():.0f}"

def acc(sig_f, sig_R):
    d2 = (df / sig_f) ** 2 + (dR / sig_R) ** 2          # Mahalanobis distance^2 between the two hypotheses (M is identical)
    return norm.cdf(np.sqrt(d2) / 2)                    # Bayes accuracy for two equal-prior Gaussian hypotheses

# accuracy vs f-mode precision for several radius precisions
sf = np.logspace(-4, np.log10(0.3), 60)
curves = {sR: [acc(s, sR).mean() for s in sf] for sR in (0.05, 0.01, 0.002)}
for sR, c in curves.items():
    c = np.array(c)
    for lvl in (0.75, 0.90):
        j = np.where(c >= lvl)[0]
        M[f"pairSf_{int(sR*1000)}_{int(lvl*100)}"] = f"{100*sf[j[-1]]:.2f}" if len(j) else "<0.01"
for s_ in (0.25, 1.0, 4.0):    # points of the precision scale (sigma_f = 2% s, sigma_R = 5% s)
    M[f"pairAcc_s{str(s_).replace('.', 'p')}"] = f"{100*acc(0.02*s_, 0.05*s_).mean():.0f}"
for sfv in (0.20, 0.05, 0.01, 0.003):
    M[f"pairAccf_{int(sfv*1000)}"] = f"{100*acc(sfv, 0.09).mean():.0f}"
# dependence on G at 1.4 Msun
j14 = int(np.argmin(abs(MG - 1.4)))
G = np.array([p["G"] for p in pairs])
M["pairCorrG"] = f"{np.corrcoef(np.log(G), df[:, j14])[0, 1]:.2f}"
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
ax[0].scatter(G, 100 * df[:, j14], s=10, c="#2e8b57")
ax[0].set(xlabel=r"Self-interaction $G$ (fm$^2$)", ylabel=r"$\Delta f/f$ at $1.4\,M_\odot$ (%)", title="(a) Dark-matter shift at fixed nuclear EoS")
for sR, c, ls in ((0.05, "#2c6fbb", "-"), (0.01, "#e08e0b", "--"), (0.002, "#c0392b", ":")):
    ax[1].plot(100 * sf, 100 * np.array(curves[sR]), color=c, ls=ls, label=rf"$\sigma_R={100*sR:g}\%$")
ax[1].axhline(50, c="k", ls=":", lw=1)
ax[1].set(xscale="log", xlabel=r"$f$-mode error $\sigma_f$ (%)", ylabel="Accuracy, DM vs no DM (%)", title="(b) Known nuclear EoS (paired)")
ax[1].legend(frameon=False, fontsize=7); ax[1].set_ylim(45, 101)
fig.tight_layout(); fig.savefig("figs/figP6_paired.pdf"); plt.close()
json.dump(M, open("macros_paired.json", "w"), indent=1)
print(json.dumps(M, indent=1))

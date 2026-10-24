"""Macros and tables for the added analyses: nucleonic-vs-DM subsets (mass bins, G terciles), tuned learners / engineered features, narrowed nuclear priors."""
import json, os, numpy as np
M = {}
if os.path.exists("results_pair_subsets.json"):
    S = json.load(open("results_pair_subsets.json"))
    for b, tag in (("all", "all"), ("1.20-1.40", "m12"), ("1.40-1.60", "m14"), ("1.60-1.85", "m16"), ("G_low", "Glo"), ("G_mid", "Gmid"), ("G_high", "Ghi")):
        for m, mt in (("RandForest", "rf"), ("HistGB", "hgb"), ("LogReg", "lr")):
            M[f"sub_{tag}_{mt}"] = f"{100*S[f'{m}|{b}'][0]:.0f}"
    allv = [100 * S[f"{m}|{b}"][0] for m in ("RandForest", "HistGB") for b in ("1.20-1.40", "1.40-1.60", "1.60-1.85", "G_low", "G_mid", "G_high")]
    M["sub_min"] = f"{min(allv):.0f}"; M["sub_max"] = f"{max(allv):.0f}"
    M["sub_se"] = f"{100*max(v[1] for v in S.values()):.1f}"
if os.path.exists("results_tuned.json"):
    T = json.load(open("results_tuned.json")); names = ["LogReg", "kNN", "RandForest", "HistGB", "SVM", "MLP"]; rows = []
    for s_ in ("noise-free", "s=0.25"):
        for k, kn in (("raw", "raw"), ("eng", "engineered")):
            R = [o for o in T if o["scen"] == s_ and o["kind"] == k]
            if not R: continue
            cells = []
            for n in names:
                sel = np.mean([o["res"][n]["sel"] for o in R]); best = np.mean([o["res"][n]["best"] for o in R]); dflt = np.mean([o["res"][n]["default"] for o in R])
                cells.append(f"{100*sel:.0f} ({100*best:.0f})")
                M[f"tu_{s_.replace('=', '').replace('.', 'p').replace('-', '')}_{k}_{n}"] = f"{100*sel:.0f}"; M[f"tub_{s_.replace('=', '').replace('.', 'p').replace('-', '')}_{k}_{n}"] = f"{100*best:.0f}"
            allbest = max(np.mean([o["res"][n]["best"] for o in R]) for n in names)
            M[f"tuMax_{s_.replace('=', '').replace('.', 'p').replace('-', '')}_{k}"] = f"{100*allbest:.0f}"
            pr = np.mean([np.mean([o["res"][n]["pair"] for o in R]) for n in names]); M[f"tuPair_{s_.replace('=', '').replace('.', 'p').replace('-', '')}_{k}"] = f"{100*pr:.0f}"
            rows.append(f"{s_.replace('s=', '$s=$')} & {kn} & " + " & ".join(cells) + " \\\\")
    open("tables/tab_tuned.tex", "w").write("\n".join(rows).rstrip()[:-2])
    M["tuN"] = len([o for o in T if o["scen"] == "noise-free" and o["kind"] == "raw"])
if os.path.exists("results_narrow.json"):
    N = json.load(open("results_narrow.json")); rows = []; sc = ("noise-free", "s=0.25", "s=1")
    for k, v in N.items():
        if k == "main": continue
        a = v["acc"]; rows.append(f"{v['w']:g} & {v['n_nuc']} & {100*v['sigma_f14']:.1f} & " + " & ".join(f"{100*a[s_][m]:.0f}" for s_ in sc for m in ("LogReg", "RandForest", "HistGB")) + " \\\\")
        tag = str(v["w"]).replace(".", "p"); M[f"nr_sig_{tag}"] = f"{100*v['sigma_f14']:.1f}"; M[f"nr_n_{tag}"] = f"{v['n_nuc']}+{v['n_dm']}"
        allacc = [100 * a[s_][m] for s_ in sc for m in ("RandForest", "HistGB")]; M[f"nr_max_{tag}"] = f"{max(allacc):.0f}"
        M[f"nr_nf_best_{tag}"] = f"{100*max(a['noise-free'].values()):.0f}"
    M["nr_sig_main"] = f"{100*N['main']['sigma_f14']:.1f}"
    M["nrN"] = str(max(v["n_nuc"] for k, v in N.items() if k != "main"))
    open("tables/tab_narrow.tex", "w").write(chr(10).join(rows).rstrip()[:-2])

if os.path.exists("results_narrow.json"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
    N = json.load(open("results_narrow.json")); ws = [k for k in N if k != "main"]; sg = np.array([100 * N[k]["sigma_f14"] for k in ws])
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for sc, lab, c in (("noise-free", "noise-free", "#2c6fbb"), ("s=0.25", r"$s=0.25$ ($\sigma_f=0.5\%$)", "#e08e0b"), ("s=1", r"$s=1$ ($\sigma_f=2\%$)", "#c0392b")):
        y = [100 * np.mean([N[k]["acc"][sc][m] for m in ("RandForest", "HistGB")]) for k in ws]
        ax.plot(sg, y, "o-", color=c, label=lab)
    ax.axhline(50, c="k", ls=":", lw=1); ax.axvline(float(json.load(open("macros_paired.json"))["pairDfMed"]), c="#2e8b57", ls="--", lw=1)
    ax.text(float(json.load(open("macros_paired.json"))["pairDfMed"]) * 1.05, 92, "median dark-matter shift", color="#2e8b57", fontsize=7, rotation=90, va="top")
    ax.set(xscale="log", xlabel=r"spread of $f$ at $1.4\,M_\odot$ among nucleonic EoS (%)", ylabel="Balanced accuracy (%)", ylim=(40, 95))
    ax.set_xticks([0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5"]); ax.minorticks_off(); ax.legend(frameon=False, fontsize=7, loc="upper right"); fig.tight_layout(); fig.savefig("figs/figP8_narrow.pdf"); plt.close()
json.dump(M, open("macros_new.json", "w"), indent=1); print(M)

"""Tables, figures and macros for the robustness analyses (error budget, frozen composition, EoS-level bootstrap,
learning curves, dark-fermion-mass window, alternative EoS population, out-of-family tests)."""
import json, os, shutil, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
M = {}
NAMES = {0: "Nucleonic", 1: "Strange", 2: "Dark-matter", 3: "Hyperonic"}


def ci(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), (stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan)


def key(s): return s.replace("+", "p").replace("@", "_").replace(".", "p")


# ------------------------------------------------------------------ merge the corrected (truncated-noise) high-noise scenarios
def merge_fix():
    if not os.path.exists("results_pop_main_fix.csv"): return
    for f in ("results_pop_main.csv", "results_pop_conf.json"):
        if not os.path.exists("orig_" + f): shutil.copy(f, "orig_" + f)
    D0 = pd.read_csv("orig_results_pop_main.csv"); F = pd.read_csv("results_pop_main_fix.csv")
    pd.concat([D0[~D0.scen.isin(F.scen.unique())], F]).to_csv("results_pop_main.csv", index=False)
    C0 = json.load(open("orig_results_pop_conf.json")); CF = json.load(open("results_pop_conf_fix.json")); fs = set(F.scen.unique())
    json.dump([o for o in C0 if o["scen"] not in fs] + CF, open("results_pop_conf.json", "w"))
    print("merged corrected scenarios:", sorted(fs))
merge_fix()

# ------------------------------------------------------------------ black-hole QNM benchmark and frozen composition
if os.path.exists("results_bh_qnm.json"):
    B = json.load(open("results_bh_qnm.json"))["runs"]["rmax=60"]
    M["bhRe"] = f"{B['re']:.5f}"; M["bhIm"] = f"{B['im']:.5f}"; M["bhErrRe"] = f"{abs(B['rel_err_re']):.0e}"; M["bhErrIm"] = f"{max(abs(v['rel_err_im']) for v in json.load(open('results_bh_qnm.json'))['runs'].values()):.0e}"
if os.path.exists("results_frozen.json"):
    S = json.load(open("results_frozen.json"))["summary"]
    M["frN"] = S["n"]; M["frShiftEq"] = f"{100*S['shift_eq_med']:.2f}"; M["frShiftFr"] = f"{100*S['shift_frozen_med']:.2f}"
    M["frDMed"] = f"{100*S['dshift_med']:.3f}"; M["frDMax"] = f"{100*S['dshift_max']:.3f}"; M["frRatio"] = f"{100*S['ratio_med']:.1f}"

# ------------------------------------------------------------------ dark-fermion mass window
if os.path.exists("results_pop_main_wide.csv"):
    Dm = pd.read_csv("results_pop_main_wide.csv"); Cm = json.load(open("results_pop_conf_wide.json"))
    for sc in ("ALL@0", "NICER+f20", "SO@1.0", "SO@0.25"):
        for mo, tg in (("MLP-ens", "mlp"), ("Bayes-kernel", "bk")):
            try:
                M[f"wide_{key(sc)}_{tg}"] = f"{100*ci(Dm[(Dm.scen == sc) & (Dm.model == mo)].groupby('rep').acc.mean())[0]:.0f}"
            except Exception: pass
    _dm = sorted([r for r in (json.load(open(f)) for f in glob.glob("pop_data/c2_*.json")) if len(r["masses"]) >= 5 and max(r["masses"]) >= 1.85 - 1e-9], key=lambda r: r["idx"])[:90]
    M["wide_nOut"] = sum(r["par"]["mchi"] > 938.783 for r in _dm); M["wide_nIn"] = 90 - M["wide_nOut"]
    cm = np.sum([np.array(o["cm"]) for o in Cm if o["scen"] == "ALL@0"], 0).astype(float)
    M["wide_pair"] = f"{100*(cm[0, 0] + cm[2, 2]) / (cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]):.0f}"; M["wide_strange"] = f"{100*cm[1, 1] / cm[1].sum():.0f}"

# ------------------------------------------------------------------ EoS-level bootstrap and learning curves
if os.path.exists("results_stat_robust.json"):
    R = pd.DataFrame(json.load(open("results_stat_robust.json")))
    rng = np.random.default_rng(1)
    rows = []
    def metrics(df, idx_by_cls):
        out = {}
        a = {k: df.loc[v] for k, v in idx_by_cls.items()}
        out["acc"] = np.mean([ (a[k].correct.sum() / a[k].n.sum()) for k in range(4)])
        pr = lambda k, j: np.sum([p[j] for p in a[k].pred])
        den = pr(0, 0) + pr(0, 2) + pr(2, 0) + pr(2, 2)
        out["pair"] = (pr(0, 0) + pr(2, 2)) / den if den else np.nan
        out["strange"] = a[1].correct.sum() / a[1].n.sum(); out["hyper"] = a[3].correct.sum() / a[3].n.sum()
        return out
    for sc in ("ALL@0", "NICER+f20", "SO@1.0", "SO@0.25"):
        for mo in ("MLP-ens", "LogReg", "RandForest", "Bayes-kernel"):
            d = R[(R.scen == sc) & (R.model == mo) & (R.frac == 1.0)]
            if d.empty: continue
            g = d.groupby("eos").agg(cls=("cls", "first"), n=("n", "sum"), correct=("correct", "sum"), pred=("pred", lambda s: np.sum(np.array(list(s)), 0).tolist())).reset_index()
            by = {k: g.index[g.cls == k].to_numpy() for k in range(4)}
            est = metrics(g, by)
            boots = []
            for _ in range(2000):
                bb = {k: rng.choice(v, len(v), replace=True) for k, v in by.items()}
                boots.append(metrics(g, bb))
            bm = pd.DataFrame(boots)
            for m_ in ("acc", "pair", "strange", "hyper"):
                lo, hi = np.nanpercentile(bm[m_], [2.5, 97.5])
                M[f"bs_{key(sc)}_{mo}_{m_}"] = f"{100*est[m_]:.0f}"; M[f"bs_{key(sc)}_{mo}_{m_}_lo"] = f"{100*lo:.0f}"; M[f"bs_{key(sc)}_{mo}_{m_}_hi"] = f"{100*hi:.0f}"
            rows.append(f"{sc.replace('@', ' ')} & {mo} & {100*est['acc']:.0f} [{100*np.nanpercentile(bm['acc'],2.5):.0f}, {100*np.nanpercentile(bm['acc'],97.5):.0f}] & "
                        f"{100*est['pair']:.0f} [{100*np.nanpercentile(bm['pair'],2.5):.0f}, {100*np.nanpercentile(bm['pair'],97.5):.0f}] & "
                        f"{100*est['strange']:.0f} [{100*np.nanpercentile(bm['strange'],2.5):.0f}, {100*np.nanpercentile(bm['strange'],97.5):.0f}] \\\\")
    open("tables/tab_boot.tex", "w").write("\n".join(rows).rstrip()[:-2])
    # learning curves
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=False)
    for a, sc in zip(ax, ("ALL@0", "NICER+f20", "SO@1.0")):
        for mo, c in (("LogReg", "#7f8c8d"), ("RandForest", "#2e8b57"), ("Bayes-kernel", "#e08e0b"), ("MLP-ens", "#c0392b")):
            d = R[(R.scen == sc) & (R.model == mo)]
            if d.empty: continue
            x, y = [], []
            for fr, g in d.groupby("frac"):
                x.append(fr * 216); y.append(100 * g.correct.sum() / g.n.sum()); M[f"lc_{key(sc)}_{mo}_{int(fr*100)}"] = f"{y[-1]:.0f}"
            a.plot(x, y, "o-", ms=4, color=c, label=mo)
        a.axhline(25, c="k", ls=":"); a.set(xlabel="Number of training EoS", title=sc.replace("@", " "), xticks=[54, 108, 162, 216]); a.set_xticklabels([54, 108, 162, 216])
    ax[0].set_ylabel("Held-out-EoS accuracy (%)"); ax[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig("figs/figP7_learning.pdf"); plt.close()

# ------------------------------------------------------------------ alternative EoS population (B)
import pop_data_utils as U
if os.path.isdir("pop_data_B"):
    U.N_PER_CLASS = int(os.environ.get("NB", 48)); RB = U.load_all("pop_data_B")
    rows = []
    for k in range(4):
        r = [x for x in RB if x["cls"] == k]
        if not r: continue
        f14 = [np.interp(1.4, x["masses"], x["f"]) for x in r]
        rows.append(f"{NAMES[k]} & {len(r)} & {min(x['Mmax'] for x in r):.2f}--{max(x['Mmax'] for x in r):.2f} & {min(x['R14'] for x in r):.1f}--{max(x['R14'] for x in r):.1f} & "
                    f"{min(x['Lam14'] for x in r):.0f}--{max(x['Lam14'] for x in r):.0f} & {min(f14):.0f}--{max(f14):.0f} \\\\")
        M[f"nB{k}"] = len(r)
    open("tables/tab_popB_data.tex", "w").write("\n".join(rows).rstrip()[:-2]); M["nBtot"] = len(RB)
    M["accB"] = {}
    DB = None
    if os.path.exists("results_pop_main_B.csv"):
        DB = pd.read_csv("results_pop_main_B.csv"); CB = json.load(open("results_pop_conf_B.json"))
        for sc in ("ALL@0", "NICER+f20", "SO@1.0", "SO@0.25", "S@0"):
            for mo, tg in (("MLP-ens", "mlp"), ("Bayes-kernel", "bk"), ("LogReg", "lr"), ("RandForest", "rf")):
                x = DB[(DB.scen == sc) & (DB.model == mo)]
                if len(x): M[f"B_{key(sc)}_{tg}"] = f"{100*ci(x.groupby('rep').acc.mean())[0]:.0f}"; M[f"B_{key(sc)}_{tg}_h"] = f"{100*ci(x.groupby('rep').acc.mean())[1]:.0f}"
        for sc in ("ALL@0", "SO@0.25", "NICER+f20"):
            cm = np.sum([np.array(o["cm"]) for o in CB if o["scen"] == sc], 0).astype(float); n = cm / cm.sum(1, keepdims=True)
            for i in range(4):
                M[f"Bcm_{key(sc)}_{i}"] = f"{100*n[i, i]:.0f}"
            M[f"Bpair_{key(sc)}"] = f"{100*(cm[0, 0] + cm[2, 2]) / (cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]):.0f}"
            M[f"Bmerged_{key(sc)}"] = f"{100*(cm[np.ix_([0, 2], [0, 2])].sum() + cm[1, 1] + cm[3, 3]) / cm.sum():.0f}"
M.pop("accB", None)

# ------------------------------------------------------------------ cross-family
if os.path.exists("results_cross_family.json"):
    X = pd.DataFrame(json.load(open("results_cross_family.json"))); rows = []
    for d in ("A->B", "B->A"):
        for sc in ("ALL@0", "NICER+f20", "SO@1.0", "SO@0.25"):
            cells = []
            for mo in ("LogReg", "RandForest", "HistGB", "MLP-ens", "Bayes-kernel"):
                x = X[(X.direction == d) & (X.scen == sc) & (X.model == mo)]
                cells.append("--" if x.empty else f"{100*ci(x.acc)[0]:.0f}$\\pm${100*ci(x.acc)[1]:.0f}")
                if not x.empty: M[f"X_{d[0]}{d[-1]}_{key(sc)}_{mo}"] = f"{100*ci(x.acc)[0]:.0f}"
            rows.append(f"{d.replace('->', r'$\to$')} & {sc.replace('@', ' ')} & " + " & ".join(cells) + " \\\\")
            cm = np.sum([np.array(r["cm"]) for r in X[(X.direction == d) & (X.scen == sc) & (X.model == "MLP-ens")].to_dict("records")], 0).astype(float)
            if cm.sum():
                n = cm / cm.sum(1, keepdims=True)
                for i in range(4): M[f"Xcm_{d[0]}{d[-1]}_{key(sc)}_{i}"] = f"{100*n[i, i]:.0f}"
                M[f"Xpair_{d[0]}{d[-1]}_{key(sc)}"] = f"{100*(cm[0, 0] + cm[2, 2]) / (cm[0, 0] + cm[0, 2] + cm[2, 0] + cm[2, 2]):.0f}"
    open("tables/tab_cross.tex", "w").write("\n".join(rows).rstrip()[:-2])

json.dump(M, open("macros_robust.json", "w"), indent=1)
print(len(M), "macros")

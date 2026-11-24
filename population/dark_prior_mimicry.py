"""Expanded interaction-prior screening and PP mimicry diagnostic.

Run with --proposals 8 for pilot; then use >=100 per G stratum and
--fit-count 59. This is not a replacement for independently validated
complex-frequency benchmarks. All numerical failures are retained.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from joblib import Parallel, delayed
from families import sample_params, build, read_pp, screen, pc_for_mass, solve_star
from fmode import Background
import fmode2 as F

G_BINS=[(5,25),(25,50),(50,100),(100,200),(200,500),(500,1000),(1000,2000)]
LO=np.array([34.15,2.4,2.4,2.2])
HI=np.array([34.60,3.8,3.9,3.9])

def full_causal(eos,sc):
    ps=np.geomspace(max(eos.p_surface()*(1+1e-8),1e-16),np.exp(sc["lpcmax"]),256)
    c=np.array([eos.cs2(float(p)) for p in ps])
    return bool(np.isfinite(c).all() and (c>0).all() and (c<=1.001).all())

def mode_at(eos,sc,mass):
    pc=np.exp(pc_for_mass(eos,mass,sc["lpcmax"]))
    s=solve_star(eos,pc)
    bg=Background(eos,pc)
    guess=1000*(.79+1.50*np.sqrt((mass/1.4)/(s["R"]/10)**3))
    for mult in (1.,.97,1.03,.94,1.06):
        try:
            f,tau,_=F.find_mode(bg,guess*mult,.2)
            if np.isfinite(f) and .7*guess<f<1.35*guess and .03<tau<3:
                return dict(R=float(s["R"]),Lambda=float(s["Lambda"]),f=float(f),tau=float(tau))
        except Exception:
            pass
    raise RuntimeError("mode search failed")

def proposal(i,b):
    lo,hi=G_BINS[b]
    # Same nuclear prior draw and stable particle mass for index i across G strata.
    rng=np.random.default_rng([20261124,i])
    par=sample_params(0,rng)
    mchi=float(rng.uniform(937.900001,938.782999))
    g=float(lo+(hi-lo)*np.random.default_rng([20261124,i,b,99]).random())
    dm={**par,"kind":"dm","G":g,"mchi":mchi}
    row=dict(index=i,bin=b,G=g,mchi=mchi,nuclear_par=par,accepted=False)
    try:
        eos=build(dm)
        sc=screen(eos)
        if sc is None:return {**row,"reason":"screen_solver"}
        row["screen"]=sc
        if not sc["ok"]:return {**row,"reason":sc["why"]}
        if not full_causal(eos,sc):return {**row,"reason":"full_causality"}
        row["accepted"]=True
        row["Mmax_le_2p3"]=bool(sc["Mmax"]<=2.3)
        base=build(par);bs=screen(base)
        row["base_screen"]=bs
        if bs is None or not bs["ok"] or not full_causal(base,bs):
            return {**row,"paired_mode_status":"nuclear_base_failed_screen"}
        row["base"]=mode_at(base,bs,1.4)
        row["dm"]=mode_at(eos,sc,1.4)
        row["shift_percent"]={k:100*(row["dm"][k]/row["base"][k]-1) for k in ("R","Lambda","f","tau")}
        row["paired_mode_status"]="ok"
    except Exception as exc:
        row["error"]=repr(exc)
        row["reason"]="paired_mode_failure" if row["accepted"] else "numerical_exception"
    return row

def fit_one(rec):
    row=dict(idx=rec["idx"],G=rec["par"]["G"],mchi=rec["par"]["mchi"])
    try:
        target=build(rec["par"]);sc=screen(target)
        if not sc or not sc["ok"] or not full_causal(target,sc):raise ValueError("target screen failed")
        base=read_pp(*(rec["par"][k] for k in ("logP1","G1","G2","G3")))
        p=np.geomspace(base.p_of_rho(1e14),min(float(target.p[-1]),np.exp(sc["lpcmax"])),161)
        et=np.array([target.eps(float(x)) for x in p])
        if not np.all(np.diff(et)>0):raise ValueError("target nonmonotone")
        # Fit inverse curve eps(P) on alternating pressure points. Evaluate
        # P(eps) residual on the other points using monotone interpolation.
        def residual(x):
            try:
                e=read_pp(*x)
                return np.log(np.array([e.eps(float(z)) for z in p[::2]])/et[::2])
            except (ValueError,ArithmeticError):
                return np.full(len(p[::2]),1e3)
        initial=np.clip(np.array([rec["par"][k] for k in ("logP1","G1","G2","G3")]),LO+1e-9,HI-1e-9)
        fit=least_squares(residual,initial,bounds=(LO,HI),xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=300)
        if not fit.success:raise RuntimeError(fit.message)
        e=read_pp(*fit.x)
        ef=np.array([e.eps(float(z)) for z in p])
        if not np.all(np.diff(ef)>0):raise ValueError("fitted EoS nonmonotone")
        check=np.arange(1,len(p),2)
        check=check[(et[check]>=ef[0])&(et[check]<=ef[-1])]
        if len(check)<.9*(len(p)//2):raise ValueError("insufficient common energy-density support")
        pf=np.exp(PchipInterpolator(np.log(ef),np.log(p))(np.log(et[check])))
        d=100*(pf/p[check]-1)
        row.update(fit_parameters=fit.x.tolist(),n_holdout=int(len(check)),
                   pressure_relative_rms_percent=float(np.sqrt(np.mean(d*d))),
                   pressure_max_abs_percent=float(np.max(np.abs(d))))
        fs=screen(e)
        row["fitted_screen"]=fs
        row["fitted_causal"]=bool(fs and fs["ok"] and full_causal(e,fs))
        if row["fitted_causal"]:
            row["target"]=mode_at(target,sc,1.4)
            row["fitted"]=mode_at(e,fs,1.4)
            row["stellar_residual_percent"]={k:100*(row["fitted"][k]/row["target"][k]-1) for k in ("R","Lambda","f","tau")}
    except Exception as exc:
        row["error"]=repr(exc)
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--proposals",type=int,default=8)
    ap.add_argument("--fit-count",type=int,default=8)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--output",type=Path,default=Path("results_dark_prior_mimicry_pilot.json"))
    a=ap.parse_args()
    from joint_stability_classification import derive_balanced_records
    records,_=derive_balanced_records()
    dm=[r for r in records if r["cls"]==2][:a.fit_count]
    proposals=Parallel(n_jobs=a.workers,verbose=5)(delayed(proposal)(i,b) for b in range(len(G_BINS)) for i in range(a.proposals))
    fits=Parallel(n_jobs=a.workers,verbose=5)(delayed(fit_one)(r) for r in dm)
    summary=[]
    for b,bounds in enumerate(G_BINS):
        s=[r for r in proposals if r["bin"]==b];accepted=[r for r in s if r["accepted"]]
        good=[r for r in accepted if r.get("paired_mode_status")=="ok"]
        shifts=[r["shift_percent"]["f"] for r in good]
        summary.append(dict(G_bounds=bounds,proposed=len(s),accepted=len(accepted),
                            acceptance_fraction=len(accepted)/len(s),paired_modes=len(good),
                            f_shift_median_percent=float(np.median(shifts)) if shifts else None,
                            Mmax_le_2p3=sum(r["Mmax_le_2p3"] for r in accepted)))
    a.output.write_text(json.dumps(dict(proposal_rows=proposals,screening_summary=summary,fit_rows=fits,
        caveat="Stratified proposals, not an inferred G prior. Mmax<=2.3 is illustrative sensitivity, not a universal observational exclusion. Fit tests representational flexibility only; no new classification result."),indent=2))
    print("DARK_PRIOR_MIMICRY_COMPLETE",a.output,flush=True)

if __name__=="__main__":main()

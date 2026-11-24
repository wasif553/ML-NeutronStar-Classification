"""Focused PDU sensitivity of the archived stable causal catalogue.
No fresh stellar models are generated. All reference and test roles are
disjoint by EoS. Selection-conditioned empirical prior, equal family weights.
"""
import json, platform, time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp
from shared_eos_evidence import fold_ids, noisy, eos_loglike, family_logz, summarize, SCENARIOS
from joint_stability_classification import derive_balanced_records
from pop_data_utils import tables
from joblib import Parallel, delayed

LO=np.array([34.15,2.4,2.4,2.2])
HI=np.array([34.60,3.8,3.9,3.9])
KEYS=("logP1","G1","G2","G3")
SCENS=("one_MRf","three_MR_MLambda_Mf","three_MR_MLambda_Mf_precise")

def width(r):
    return float(np.max(np.abs(2*(np.array([r["par"][k] for k in KEYS])-LO)/(HI-LO)-1)))

def run_case(name, recs, labels, ndraw=8, reps=3, nquad=132):
    cls,m,obs,lam=tables(recs)
    counts={str(k):int(np.sum(cls==k)) for k in labels}
    out={"selection":name,"counts":counts}
    if min(counts.values())<10:
        return {**out,"status":"insufficient EoSs (minimum 10 per family)"}
    obs=np.concatenate([obs,lam[...,None]],axis=2)
    assert np.all(np.isfinite(obs)) and np.all(obs>0)
    interp=PchipInterpolator(m,np.log(obs),axis=1)
    q=m[0]+(np.arange(nquad)+.5)/nquad*(m[-1]-m[0])
    ref_all=np.exp(interp(q))
    result={}
    for scenario in SCENS:
        specs=SCENARIOS[scenario]
        rows=[]
        for rep in range(reps):
            part=fold_ids(cls,777+rep)
            for fold in range(5):
                te=np.flatnonzero(part==fold);tr=np.flatnonzero(part!=fold)
                assert not np.intersect1d(te,tr).size
                refs=[ref_all[tr][:,:,cols] for cols,_ in specs]
                for g in te:
                    for d in range(ndraw):
                        r=recs[g]
                        rng=np.random.default_rng([20261124,rep,int(r["cls"]),int(r["idx"]),d])
                        ll=np.zeros(len(tr))
                        for (cols,sig),ref in zip(specs,refs):
                            mass=rng.uniform(m[0],m[-1])
                            x=noisy(np.exp(interp(mass)[g,cols]),sig,rng)
                            ll+=eos_loglike(x,ref,sig)
                        z=family_logz(ll,cls[tr],labels)
                        lp=z-logsumexp(z);p=np.exp(lp)
                        k=labels.index(int(cls[g]))
                        rows.append(dict(eos=int(g),cls=int(cls[g]),pred=labels[int(p.argmax())],
                          correct=int(p.argmax()==k),log_true=float(lp[k]),entropy=float(-np.sum(p*lp)),
                          log_bf_2_vs_0=float(z[labels.index(2)]-z[labels.index(0)])))
        result[scenario]=summarize(rows,labels)
    return {**out,"status":"complete","results":result}

def main():
    start=time.time()
    records,selection=derive_balanced_records()
    assert len(records)==236
    binary=[r for r in records if r["cls"] in (0,2)]
    cases=[("all_four",records,[0,1,2,3]),("all_binary",binary,[0,2]),
           ("Mmax_2p3_four",[r for r in records if r["Mmax"]<=2.3],[0,1,2,3]),
           ("Mmax_2p3_binary",[r for r in binary if r["Mmax"]<=2.3],[0,2])]
    for lo,hi in [(50,150),(150,300),(300,500.00001)]:
        cases.append((f"G_{lo}_{hi:g}",[r for r in binary if r["cls"]==0 or lo<=r["par"]["G"]<hi],[0,2]))
    for w in (.95,.85,.75,.65):
        cases.append((f"prior_width_{w}",[r for r in binary if width(r)<=w],[0,2]))
    results=Parallel(n_jobs=4,verbose=5)(delayed(run_case)(*case) for case in cases)
    output=dict(selection=selection,quadrature=132,draws=8,repetitions=3,
        caveat="Finite empirical catalogues; central rectangular prior restriction about original prior midpoints, not a posterior. G bins condition only DM class. Mmax upper cut is illustrative. Changed sample size and reference density confound prior effects; do not infer continuous-prior bounds.",
        results=results,runtime_seconds=time.time()-start,python=platform.python_version())
    Path("results_pdu_targeted_sensitivity.json").write_text(json.dumps(output,indent=2))
    for r in results:
        print("PDU_SENSITIVITY",r["selection"],r["counts"],r["status"],{k:round(v["accuracy"]*100,2) for k,v in r.get("results",{}).items()},flush=True)
    print("PDU_TARGETED_SENSITIVITY_COMPLETE",flush=True)

if __name__=="__main__":main()

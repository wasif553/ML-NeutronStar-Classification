"""Joint full-interior causality and stable-dark-fermion-mass sensitivity analysis.

This is an explicitly restricted supplementary analysis, not a claim that
all models are microscopically realistic or a replacement for the 340-EoS
primary numerical benchmark. Uses independent EoS-level holdout.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
from pop_data_utils import load_all, tables, NM
from pop_experiment import SCEN, mlp, folds

SOURCE=Path("results_causality_audit.json")
OUT=Path("results_joint_stability_classification.json")
SCENARIOS=("ALL@0","NICER+f20")


def derive_balanced_records():
    """Intersect full-interior causality with the stable dark fermion window.
    Rebalance all four families; no EoS is used in more than one fold role.
    """
    records = load_all()
    if len(records) != 360:
        raise RuntimeError(f"Expected 360 original EoSs; got {len(records)}")
    audit = json.loads(SOURCE.read_text())
    passed = {(int(r["cls"]), int(r["idx"]))
              for r in audit["groups"]["pop_data"]["records"] if r["causal_full"]}
    causal = {c: [r for r in records if r["cls"] == c and (c, r["idx"]) in passed]
              for c in range(4)}
    stable_dark = [r for r in causal[2]
                   if 937.9 < float(r["par"]["mchi"]) <= 938.783]
    available = {**causal, 2: stable_dark}
    n = min(len(x) for x in available.values())
    if n < 45:
        raise RuntimeError(f"Insufficient jointly screened models: {[len(x) for x in available.values()]}")
    sampled = [r for c in range(4) for r in available[c][:n]]
    assert len(sampled) == 4*n
    counts = {"full_causal_by_class": {str(c): len(causal[c]) for c in range(4)},
              "joint_eligible_by_class": {str(c): len(available[c]) for c in range(4)},
              "used_per_class": n, "total": len(sampled),
              "dark_fermion_mev_window": [937.9, 938.783],
              "screen": "256-point pressure-grid interior causality plus stable chi mass for DM; rebalanced all classes"}
    return sampled, counts

def draw(obs,cls,idx,mass_idx,n_draw,cols,sig,rng):
    x=obs[np.ix_(idx,mass_idx)][:,:,cols].reshape(-1,len(cols))
    x=np.repeat(x,n_draw,axis=0)
    y=np.repeat(np.repeat(cls[idx],len(mass_idx)),n_draw)
    sg=np.asarray(sig)
    errors=rng.standard_normal(x.shape)
    invalid=(1+sg*errors)<=0
    while invalid.any():
        errors[invalid]=rng.standard_normal(invalid.sum())
        invalid=(1+sg*errors)<=0
    return x*(1+sg*errors),y


def task(scen,rep,fold,cls,obs):
    cols,sig=SCEN[scen]
    rng=np.random.default_rng([777,rep,fold,sum(map(ord,scen))])
    part=folds(cls,np.random.default_rng([777,rep]))
    te=np.flatnonzero(part==fold)
    ca=np.flatnonzero(part==(fold+1)%5)
    tr=np.flatnonzero((part!=fold)&(part!=(fold+1)%5))
    masses=np.arange(0,NM,2)
    nd=1 if max(sig)==0 else 3
    xtr,ytr=draw(obs,cls,tr,masses,nd,cols,sig,rng)
    xte,yte=draw(obs,cls,te,masses,nd,cols,sig,rng)
    scaler=StandardScaler().fit(np.log(xtr))
    xt=scaler.transform(np.log(xtr))
    xq=scaler.transform(np.log(xte))
    lr=LogisticRegression(max_iter=2000).fit(xt,ytr)
    lpred=lr.predict(xq)
    ens=[mlp(777+10*j+fold).fit(xt,ytr) for j in range(3)]
    probabilities=np.mean([m.predict_proba(xq) for m in ens],axis=0)
    predicted=probabilities.argmax(1)
    def metric(pred):
        mat=confusion_matrix(yte,pred,labels=np.arange(4))
        den=mat[0,0]+mat[0,2]+mat[2,0]+mat[2,2]
        return dict(acc=float(np.mean(pred==yte)),cm=mat.tolist(),
                    pair_acc_conditional=float((mat[0,0]+mat[2,2])/den) if den else None)
    return dict(scen=scen,rep=rep,fold=fold,n_train_eos=len(tr),
                n_test_eos=len(te),MLP=metric(predicted),LogReg=metric(lpred))


def main():
    recs,counts=derive_balanced_records()
    cls,mg,obs,_=tables(recs)
    jobs=[(sc,r,f,cls,obs) for sc in SCENARIOS for r in range(3) for f in range(5)]
    rows=Parallel(n_jobs=4,verbose=5)(delayed(task)(*a) for a in jobs)
    summaries={}
    for sc in SCENARIOS:
        sub=[r for r in rows if r["scen"]==sc]
        stats={}
        for learner in ("MLP","LogReg"):
            rep_means=[np.mean([r[learner]["acc"] for r in sub if r["rep"]==rep]) for rep in range(3)]
            cms=np.sum([np.asarray(r[learner]["cm"]) for r in sub],axis=0)
            den=cms[0,0]+cms[0,2]+cms[2,0]+cms[2,2]
            stats[learner]=dict(accuracy_mean=float(np.mean(rep_means)),
                                repetition_means=[float(v) for v in rep_means],
                                pair_accuracy_conditional=float((cms[0,0]+cms[2,2])/den),
                                pooled_confusion_matrix=cms.tolist())
        summaries[sc]=stats
    out=dict(description=__doc__,selection=counts,scenarios=summaries,
             runs=rows,notes="Whole-EoS validation. Both interior-causality and stable dark-fermion mass cuts; conditioned subset of originally generated EoSs.")
    OUT.write_text(json.dumps(out,indent=2))
    print("JOINT_STABILITY_RESULT",json.dumps(dict(selection=counts,scenarios=summaries)))
    assert counts["used_per_class"]>=45
    print("JOINT_STABILITY_VALIDATION_PASS")

if __name__=="__main__":
    main()

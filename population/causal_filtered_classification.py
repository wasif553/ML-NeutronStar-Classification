"""Sensitivity of the original MLP classification to removal of EoSs that
fail a full-pressure-range causality check. This does not replace the original
population-based results without a new full analysis of all figures/tables.
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
OUT=Path("results_causal_filtered_classification.json")
SCENARIOS=("ALL@0","NICER+f20")


def derive_balanced_records():
    records=load_all()
    if len(records)!=360:raise RuntimeError(f"expected 360 original accepted EoS, got {len(records)}")
    audit=json.loads(SOURCE.read_text())
    passed={(int(r["cls"]),int(r["idx"])) for r in audit["groups"]["pop_data"]["records"]
            if r["causal_full"]}
    grouped={c:[r for r in records if r["cls"]==c and (c,r["idx"]) in passed]
             for c in range(4)}
    n=min(len(x) for x in grouped.values())
    if n<60:raise RuntimeError(f"too few causal EoS per class: {[len(x) for x in grouped.values()]}")
    sampled=[r for c in range(4) for r in grouped[c][:n]]
    return sampled,{"available_by_class":{str(c):len(grouped[c]) for c in grouped},
                    "used_per_class":n,"total":len(sampled)}


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
             runs=rows,notes="Whole-EoS validation. Unlike an independent population draw, this is a conditioned subset of the original sampled EoS.")
    OUT.write_text(json.dumps(out,indent=2))
    print("CAUSAL_FILTERED_RESULT",json.dumps(dict(selection=counts,scenarios=summaries)))
    assert counts["used_per_class"]==85
    print("CAUSAL_FILTERED_VALIDATION_PASS")

if __name__=="__main__":
    main()

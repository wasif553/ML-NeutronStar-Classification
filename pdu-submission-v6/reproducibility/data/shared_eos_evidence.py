"""Held-out-EoS finite-catalogue evidence for shared-EoS observations.

Not continuous-prior Bayes-optimal inference. Independent Gaussian fractional
error and uniform latent mass are illustrative. All stars in a synthetic
system share a single generating EoS; reference EoS are held out as groups.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, log_ndtr
from scipy.interpolate import PchipInterpolator
from scipy.special import softmax

SCENARIOS = {
    "one_MR": [([0, 1], [.10, .09])],
    "one_MRf": [([0, 1, 2], [.10, .09, .20])],
    "two_MR_MLambda": [([0, 1], [.10, .09]), ([0, 4], [.05, .30])],
    "three_MR_MLambda_Mf": [([0, 1], [.10, .09]), ([0, 4], [.05, .30]), ([0, 2], [.05, .20])],
    "three_MR_MLambda_Mf_precise": [([0, 1], [.03, .03]), ([0, 4], [.03, .10]), ([0, 2], [.03, .005])],
}

def fold_ids(cls, seed):
    rng = np.random.default_rng(seed)
    part = np.empty(len(cls), int)
    for c in np.unique(cls):
        ix = rng.permutation(np.flatnonzero(cls == c))
        part[ix] = np.arange(len(ix)) % 5
    return part

def noisy(x, sig, rng):
    sig = np.asarray(sig)
    f = 1 + rng.standard_normal(x.shape) * sig
    while np.any(f <= 0):
        bad = f <= 0
        f[bad] = (1 + rng.standard_normal(x.shape) * sig)[bad]
    return x * f

def eos_loglike(x, ref, sig):
    # ref shape (EoS, mass quadrature, observable); midpoint uniform quadrature.
    sd = ref * np.asarray(sig)
    ll = (-.5 * ((x[None, None, :] - ref) / sd)**2
          - np.log(sd) - .5*np.log(2*np.pi)
          - log_ndtr(1 / np.asarray(sig))).sum(axis=-1)
    return logsumexp(ll, axis=1) - np.log(ref.shape[1])

def family_logz(ll, cls, labels):
    return np.array([logsumexp(ll[cls == k])-np.log(np.sum(cls == k)) for k in labels])

def summarize(rows, labels, nboot=2000):
    # Every EoS receives equal weight within each class; draws and repeat
    # partitions for an EoS remain in the same resampling block.
    labels = np.asarray(labels)
    groups = sorted({r["eos"] for r in rows})
    y = np.array([next(r["cls"] for r in rows if r["eos"] == g) for g in groups])
    arr = np.array([[np.mean([r[m] for r in rows if r["eos"] == g])
                     for m in ("correct", "log_true", "entropy", "log_bf_2_vs_0")]
                    for g in groups])
    def mean(a, cl):
        return np.mean([a[cl == k].mean(axis=0) for k in labels], axis=0)
    point = mean(arr, y)
    rng = np.random.default_rng(8801)
    ix = [np.flatnonzero(y == k) for k in labels]
    boot = np.array([np.mean([arr[rng.choice(i,len(i),replace=True)].mean(axis=0)
                             for i in ix],axis=0) for _ in range(nboot)])
    cm = np.zeros((len(labels), len(labels)), float)
    for g in groups:
        rr = [r for r in rows if r["eos"] == g]
        j = list(labels).index(rr[0]["cls"])
        for r in rr:
            cm[j,list(labels).index(r["pred"])] += 1/len(rr)
    cm /= cm.sum(axis=1, keepdims=True)
    precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(len(labels)), where=cm.sum(axis=0)>0)
    return dict(n_eos=len(groups), n_evaluation_draws=len(rows), accuracy=float(point[0]),
                accuracy_cluster_percentile=np.quantile(boot[:,0],[.025,.975]).tolist(),
                nll_nats=float(-point[1]),
                predictive_information_score_nats=float(np.log(len(labels))+point[1]),
                predictive_information_cluster_percentile=(np.log(len(labels))+np.quantile(boot[:,1],[.025,.975])).tolist(),
                mean_posterior_entropy_nats=float(point[2]), confusion_row_normalized=cm.tolist(),
                labels=labels.tolist(), precision_equal_prior=precision.tolist(), recall=np.diag(cm).tolist(),
                log_bf_2_vs_0_by_true_class={str(k):float(arr[y==k,3].mean()) for k in labels},
                bootstrap_note="Conditional on frozen catalogue and fitted reference folds; not full model/prior uncertainty.")

def evaluate(cls, masses, obs, nquad, ndraw, reps):
    # obs includes M,R,f,tau,Lambda and has common interpolation support.
    assert np.all(np.isfinite(obs)) and np.all(obs>0)
    interp = PchipInterpolator(masses, np.log(obs), axis=1)
    quad = masses[0] + (np.arange(nquad)+.5)/nquad*(masses[-1]-masses[0])
    ref_all = np.exp(interp(quad))
    out = {}
    for scenario, specs in SCENARIOS.items():
        rows = {"four":[], "binary":[]}
        for rep in range(reps):
            part = fold_ids(cls, 777+rep)
            for fold in range(5):
                test = np.flatnonzero(part == fold)
                train = np.flatnonzero(part != fold)
                assert not np.intersect1d(test, train).size
                refs = [ref_all[train][:,:,cols] for cols,_ in specs]
                for g in test:
                    for draw in range(ndraw):
                        rng = np.random.default_rng([20261124,rep,int(g),draw])
                        # Each scenario uses the same latent star masses and noise seeds
                        # for a given quadrature resolution. All are on EoS g.
                        ll = np.zeros(len(train))
                        for (cols,sig), ref in zip(specs,refs):
                            m = rng.uniform(masses[0], masses[-1])
                            true = np.exp(interp(m)[g,cols])
                            x = noisy(true, sig, rng)
                            ll += eos_loglike(x, ref, sig)
                        for key, labels in (("four",[0,1,2,3]),("binary",[0,2])):
                            if int(cls[g]) not in labels:
                                continue
                            z = family_logz(ll, cls[train], labels)
                            lp = z-logsumexp(z)
                            p = np.exp(lp)
                            k = labels.index(int(cls[g]))
                            rows[key].append(dict(eos=int(g), cls=int(cls[g]), pred=labels[int(p.argmax())],
                                                  correct=int(p.argmax()==k), log_true=float(lp[k]),
                                                  entropy=float(-np.sum(p*lp)),
                                                  log_bf_2_vs_0=float(z[labels.index(2)]-z[labels.index(0)])))
        out[scenario] = {key:summarize(rr, [0,1,2,3] if key=="four" else [0,2]) for key,rr in rows.items()}
    return out

def self_test():
    # Two observations individually prefer distinct EoSs. Joint evidence must
    # sum log likelihoods at fixed EoS before marginalizing over EoS.
    a=np.array([0.,-20.]); b=np.array([-20.,0.])
    assert np.isclose(family_logz(a+b,np.array([0,0]),[0])[0],-20)
    assert not np.isclose(logsumexp(a)-np.log(2)+logsumexp(b)-np.log(2),-20)
    r=np.array([[[1.],[2.]],[[2.],[3.]]])
    assert eos_loglike(np.array([1.]),r,[.1])[0]>eos_loglike(np.array([1.]),r,[.1])[1]
    cls=np.repeat(np.arange(4),5)
    for f in range(5): assert np.all(np.bincount(cls[fold_ids(cls,42)==f],minlength=4)==1)
    print("SHARED_EOS_EVIDENCE_UNIT_CHECK_PASS")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--sample",choices=["joint","causal"],default="joint")
    ap.add_argument("--draws",type=int,default=8)
    ap.add_argument("--reps",type=int,default=3)
    ap.add_argument("--quadrature",type=int,nargs="+",default=[66,132,264])
    ap.add_argument("--output",type=Path,default=Path("results_shared_eos_evidence.json"))
    args=ap.parse_args()
    self_test()
    if args.self_test:return
    from pop_data_utils import tables
    if args.sample=="joint":
        from joint_stability_classification import derive_balanced_records
    else:
        from causal_filtered_classification import derive_balanced_records
    recs,selection=derive_balanced_records()
    cls,mg,obs,lam=tables(recs)
    assert len(recs)==(236 if args.sample=="joint" else 340)
    all_obs=np.concatenate((obs,lam[...,None]),axis=2)
    results={}
    for nq in args.quadrature:
        results[str(nq)]=evaluate(cls,mg,all_obs,nq,args.draws,args.reps)
        print("QUADRATURE_COMPLETE",nq,flush=True)
    out=dict(selection=selection, scenarios=SCENARIOS, draws=args.draws,
             repetitions=args.reps, results=results,
             caveats="Finite held-out catalogue, illustrative independent errors, uniform masses; predictive information score is not the exact continuous-prior mutual information. Mixed scenarios are not a real observational posterior.")
    args.output.write_text(json.dumps(out,indent=2))
    print("SHARED_EOS_EVIDENCE_RESULT",args.output,flush=True)

if __name__=="__main__":main()

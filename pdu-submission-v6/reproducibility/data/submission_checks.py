"""Lower-noise paired injection, tidal shifts and dependence-robust omnibus test."""
import json, os
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,log_ndtr
from scipy.interpolate import PchipInterpolator
from joblib import Parallel,delayed
from final_checks import pair_arrays,mg,Q,NREP,ND,fold_ids,noisy,SCENARIOS,selections
GRID=[0,.25,.5,1,2,3,4,5,10,20,40]
SPECS={'precise':SCENARIOS['three_MR_MLambda_Mf_precise'],'low_noise':[([0,1],[.01,.01]),([0,4],[.01,.05]),([0,2],[.01,.001])]}
NQUAD=int(os.environ.get('PDU_INJECTION_NQUAD', '132'))
Q=mg[0]+(np.arange(NQUAD)+.5)/NQUAD*(mg[-1]-mg[0])
if NQUAD!=132: SPECS={'low_noise':SPECS['low_noise']}
def run(scale):
    ids,b,d=pair_arrays();n=len(ids);lb=np.log(b)
    interp=PchipInterpolator(mg,np.concatenate((lb,lb+scale*(np.log(d)-lb))),axis=1)
    ref=np.exp(interp(Q));y=np.repeat([0,2],n);out={}
    for scenario,spec in SPECS.items():
        acc=np.zeros((2,n))
        for rep in range(NREP):
            part=np.tile(fold_ids(np.zeros(n,dtype=int),777+rep),2)
            for fold in range(5):
                te=np.flatnonzero(part==fold);tr=np.flatnonzero(part!=fold);assert not set(te%n)&set(tr%n)
                xs=[[] for _ in spec]
                for g in te:
                    for dd in range(ND):
                        rng=np.random.default_rng([20261124,rep,int(ids[g%n]),dd])
                        for j,(cols,sig) in enumerate(spec):
                            mass=rng.uniform(mg[0],mg[-1]);xs[j].append(noisy(np.exp(interp(mass)[g,cols]),sig,rng))
                ll=np.zeros((len(te)*ND,len(tr)))
                for (cols,sig),x in zip(spec,xs):
                    rr=ref[tr][:,:,cols];sig=np.asarray(sig);sd=rr*sig;norm=np.log(sd)+.5*np.log(2*np.pi)+log_ndtr(1/sig)
                    for start in range(0,len(x),16):
                        v=np.asarray(x[start:start+16])[:,None,None,:];a=(-.5*((v-rr)/sd)**2-norm).sum(-1)
                        ll[start:start+16]+=logsumexp(a,axis=2)-np.log(len(Q))
                z=np.stack([logsumexp(ll[:,y[tr]==k],axis=1)-np.log(np.sum(y[tr]==k)) for k in [0,2]],axis=1)
                correct=np.where(np.abs(z[:,1]-z[:,0])<1e-12,.5,2*np.argmax(z,axis=1)==np.repeat(y[te],ND))
                acc[te//n,te%n]+=correct.reshape(-1,ND).mean(1)/NREP
        out[scenario]=acc.mean(0).tolist()
    print('INJECTION',scale,{k:float(np.mean(v)) for k,v in out.items()},flush=True);return str(scale),out
if __name__=='__main__':
    res=dict(Parallel(n_jobs=4)(delayed(run)(s) for s in GRID));ids,b,d=pair_arrays();n=len(ids);j=np.argmin(abs(mg-1.4));tidal={}
    for name,mask in selections()[0].items():
        ix=mask[ids];v=100*(d[ix,j,4]/b[ix,j,4]-1);tidal[name]={'n':int(ix.sum()),'median_percent':float(np.median(v))}
    rng=np.random.default_rng(8801);ii=rng.integers(n,size=(50000,n));out={}
    for s in SPECS:
        mat=np.array([res[str(a)][s] for a in GRID]);boots=mat[:,ii].mean(2)
        out[s]={str(a):{'mean':float(mat[k].mean()),'pointwise_95':np.quantile(boots[k],[.025,.975]).tolist(),'simultaneous_lower_95':float(np.quantile(boots[k],.05/(2*(len(GRID)-1))))} for k,a in enumerate(GRID)}
    p={k:v['p_lower'] for k,v in json.load(open('results_final_perm.json')).items()};m=len(p);ps=np.sort(list(p.values()));H=sum(1/i for i in range(1,m+1))
    joint={'m':m,'marginal_p':p,'ordered_p':ps.tolist(),'simes_statistic':float(np.min(m*ps/np.arange(1,m+1))),'harmonic_factor':H,'joint_p_bound':min(1.,float(H*np.min(m*ps/np.arange(1,m+1)))),'bonferroni_min_p':min(1.,m*float(ps[0]))}
    out=dict(quadrature=NQUAD,grid=GRID,scenarios=SPECS,results=out,per_pair=res,tidal=tidal,joint_permutation=joint,bootstrap_replicates=len(ii),note='Dependence-robust combination of marginal permutation probabilities; overlapping label-dependent selections are not a single exchangeable catalogue. Simultaneous one-sided percentile bounds use Bonferroni over both curves and nonzero points, conditional on frozen reference folds.')
    Path('results_submission_checks.json' if NQUAD==132 else f'results_submission_checks_q{NQUAD}.json').write_text(json.dumps(out,indent=2));print('JOINT',joint,'TIDAL',tidal,flush=True)

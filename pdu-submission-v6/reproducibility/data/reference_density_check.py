"""Reference-density control: 20 nested training-pair subsamples per fold; unchanged test pairs, draws and mass quadrature. Bounds conditional on held-out pairs and averaged reference subsets, adjusted over scenario/grid within each fraction."""
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
        acc=np.zeros((3,2,n))
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
                n0=np.flatnonzero(y[tr]==0); n2=np.flatnonzero(y[tr]==2)
                assert np.array_equal(tr[n0]%n,tr[n2]%n)
                rng=np.random.default_rng([79124,rep,fold]); perm=np.array([rng.permutation(len(n0)) for _ in range(20)])
                for fi,frac in enumerate([.5,.75,1.]):
                    k=round(frac*len(n0)); subsets=perm[:,:k] if frac<1 else np.arange(k)[None,:]
                    correct=np.zeros(len(te)*ND)
                    for sub in subsets:
                        z0=logsumexp(ll[:,n0[sub]],axis=1)-np.log(k);z2=logsumexp(ll[:,n2[sub]],axis=1)-np.log(k)
                        correct+=np.where(np.abs(z2-z0)<1e-12,.5,np.where(z2>z0,2,0)==np.repeat(y[te],ND))/len(subsets)
                    acc[fi,te//n,te%n]+=correct.reshape(-1,ND).mean(1)/NREP
        out[scenario]=acc.mean(1).tolist()
    print('INJECTION',scale,{k:float(np.mean(v)) for k,v in out.items()},flush=True);return str(scale),out
if __name__=='__main__':
    res=dict(Parallel(n_jobs=4)(delayed(run)(s) for s in GRID)); n=len(pair_arrays()[0]);rng=np.random.default_rng(8801);ii=rng.integers(n,size=(50000,n));out={}
    old=json.loads(Path('results_submission_checks.json').read_text())
    for s in SPECS:
        out[s]={}
        for fi,frac in enumerate([.5,.75,1.]):
            mat=np.array([res[str(a)][s][fi] for a in GRID]);boots=mat[:,ii].mean(2)
            out[s][str(frac)]={str(a):{'mean':float(mat[k].mean()),'pointwise_95':np.quantile(boots[k],[.025,.975]).tolist(),'simultaneous_lower_95':float(np.quantile(boots[k],.05/(2*(len(GRID)-1))))} for k,a in enumerate(GRID)}
            if frac==1: assert np.max(np.abs(mat-np.array([old['per_pair'][str(a)][s] for a in GRID])))<1e-10
    Path('results_reference_density_check.json').write_text(json.dumps(dict(fractions=[.5,.75,1.],subset_repetitions=20,quadrature=NQUAD,grid=GRID,scenarios=SPECS,results=out,per_pair=res,bootstrap_replicates=len(ii),note=__doc__),indent=2))

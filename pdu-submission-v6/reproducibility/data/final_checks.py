"""Conditional EoS-label permutation and matched-pair observable injection.
Permute selected EoS labels once per null replicate, then re-stratify folds.
Injection scales log-observable differences; not self-consistent stellar models.
"""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp
from joblib import Parallel, delayed
from referee_checks import records,cls,mg,obs,lam,selections
from shared_eos_evidence import fold_ids,noisy,eos_loglike,SCENARIOS
NREP=3;ND=8;NQ=132;NPERM=199;NSUB=20
Q=mg[0]+(np.arange(NQ)+.5)/NQ*(mg[-1]-mg[0])
def ci(a,y):
    rng=np.random.default_rng(8801); ix=[np.flatnonzero(y==k) for k in np.unique(y)]
    vals=np.mean([a[rng.choice(i,(2000,len(i)),replace=True)].mean(1) for i in ix],axis=0)
    return [float(np.mean([a[i].mean() for i in ix])),*np.quantile(vals,[.025,.975]).tolist()]
def likelihood(interp,ids,specs,rep,seedids):
    ref=np.exp(interp(Q)); ll=np.zeros((len(ids)*ND,len(ids)))
    for j,g in enumerate(ids):
      for d in range(ND):
        rng=np.random.default_rng([20261124,rep,int(seedids[j]),d]);row=ll[j*ND+d]
        for cols,sig in specs:
          m=rng.uniform(mg[0],mg[-1]);x=noisy(np.exp(interp(m)[g,cols]),sig,rng);row+=eos_loglike(x,ref[ids][:,:,cols],sig)
    return ll
def evaluate(lls,ids,yfull):
    y=yfull[ids];out=np.zeros(len(ids))
    for rep,ll in enumerate(lls):
      part=fold_ids(yfull,777+rep)[ids]
      for f in range(5):
        te=np.flatnonzero(part==f);tr=np.flatnonzero(part!=f);gr=[tr[y[tr]==k] for k in [0,2]];n=min(map(len,gr));assert n>0
        rng=np.random.default_rng([9871,rep,f]);W=np.zeros((len(ids),2*NSUB))
        for s in range(NSUB):
          for k,a in enumerate(gr):
            sub=rng.choice(a,n,replace=False);W[sub,2*s+k]=1/n
        ii=(te[:,None]*ND+np.arange(ND)).ravel();L=ll[ii];L=np.exp(L-L.max(1,keepdims=True));Z=L@W
        pred=2*(Z[:,1::2]>Z[:,0::2]);correct=(pred==np.repeat(y[te],ND)[:,None]).mean(1)
        out[te]+=correct.reshape(-1,ND).mean(1)/NREP
    return out
def perm_case(name,mask):
    ids=np.flatnonzero(mask);interp=PchipInterpolator(mg,np.log(np.concatenate((obs,lam[...,None]),axis=2)),axis=1)
    lls=[likelihood(interp,ids,SCENARIOS['three_MR_MLambda_Mf'],rep,ids) for rep in range(NREP)]
    observed=ci(evaluate(lls,ids,cls),cls[ids]);rng=np.random.default_rng(135708);null=[]
    for j in range(NPERM):
      yy=cls.copy();yy[ids]=rng.permutation(yy[ids]);a=evaluate(lls,ids,yy);null.append(np.mean([a[yy[ids]==k].mean() for k in [0,2]]))
    null=np.array(null);r=dict(observed=observed,null_mean=float(null.mean()),null_95=np.quantile(null,[.025,.975]).tolist(),p_lower=float((1+np.sum(null<=observed[0]))/(1+NPERM)),p_upper=float((1+np.sum(null>=observed[0]))/(1+NPERM)),null=null.tolist());print('PERM',name,{k:v for k,v in r.items() if k!='null'},flush=True);return name,r
def pair_arrays():
    ids=np.flatnonzero(cls==2);arr=[]
    for i in ids:
      r=records[i];b=json.loads(Path(f"pop_base/b{r['idx']:04d}.json").read_text());assert r['par']==b['par'];assert min(b['masses'])<=mg[0] and max(b['masses'])>=mg[-1]
      arr.append(np.stack([mg]+[np.exp(PchipInterpolator(b['masses'],np.log(b[k]))(mg)) for k in ['R','f','tau','Lam']],axis=-1))
    return ids,np.array(arr),np.concatenate((obs[ids],lam[ids,:,None]),axis=2)
def inject(scale):
    ids,b,d=pair_arrays();n=len(ids);logb=np.log(b);logs=np.concatenate((logb,logb+scale*(np.log(d)-logb)));interp=PchipInterpolator(mg,logs,axis=1);y=np.repeat([0,2],n);seedids=np.tile(ids,2);out={}
    for scenario,spec in [('MR',SCENARIOS['one_MR']),('MRf',SCENARIOS['one_MRf']),('three',SCENARIOS['three_MR_MLambda_Mf']),('precise',SCENARIOS['three_MR_MLambda_Mf_precise'])]:
      acc=np.zeros((2,n));ref=np.exp(interp(Q))
      for rep in range(NREP):
        part=np.tile(fold_ids(np.zeros(n,dtype=int),777+rep),2)
        for f in range(5):
          te=np.flatnonzero(part==f);tr=np.flatnonzero(part!=f);assert not set(te%n)&set(tr%n);rr=[ref[tr][:,:,cols] for cols,_ in spec]
          for g in te:
            aa=[]
            for dd in range(ND):
              rng=np.random.default_rng([20261124,rep,int(seedids[g]),dd]);ll=np.zeros(len(tr))
              for (cols,sig),rref in zip(spec,rr):
                m=rng.uniform(mg[0],mg[-1]);x=noisy(np.exp(interp(m)[g,cols]),sig,rng);ll+=eos_loglike(x,rref,sig)
              z=np.array([logsumexp(ll[y[tr]==k])-np.log(np.sum(y[tr]==k)) for k in [0,2]]);aa.append(.5 if abs(z[1]-z[0])<1e-12 else float(y[g]==2*np.argmax(z)))
            acc[g//n,g%n]+=np.mean(aa)/NREP
      a=acc.mean(0);rng=np.random.default_rng(8801);boot=a[rng.integers(n,size=(2000,n))].mean(1);out[scenario]=[float(a.mean()),*np.quantile(boot,[.025,.975]).tolist()]
    print('INJECT',scale,out,flush=True);return str(scale),out
def residual():
    ids,b,d=pair_arrays();j=np.argmin(abs(mg-1.4));N=obs[cls==0,j];x=np.log(N[:,1]);y=np.log(N[:,2]);loo=np.array([y[i]-np.polyval(np.polyfit(np.delete(x,i),np.delete(y,i),1),x[i]) for i in range(len(x))]);slope,intercept=np.polyfit(x,y,1)
    dl=np.log(d[:,j,2]/b[:,j,2]);dr=np.log(d[:,j,1]/b[:,j,1]);resid=dl-slope*dr;dens=dl+1.5*dr;spread=lambda a:float(100*(np.quantile(a,.84)-np.quantile(a,.16))/2)
    rng=np.random.default_rng(5621);bs=[]
    for k in range(2000):
      ii=rng.integers(len(ids),size=len(ids));jj=rng.integers(len(x),size=len(x));bb=np.polyfit(x[jj],y[jj],1)[0];bs.append([100*np.median((dl-bb*dr)[ii]),100*np.median(dens[ii]),bb,spread(loo[jj])])
    r=dict(n=len(ids),mass=float(mg[j]),slope=float(slope),r_squared=float(np.corrcoef(x,y)[0,1]**2),nuclear_residual_spread=spread(loo),median_log_f_shift=float(100*np.median(dl)),median_log_R_shift=float(100*np.median(dr)),median_residual_shift=float(100*np.median(resid)),median_density_residual=float(100*np.median(dens)),median_abs_residual=float(100*np.median(abs(resid))),CI=np.quantile(bs,[.025,.975],axis=0).T.tolist(),paired_log_shift_correlation=float(np.corrcoef(dl,dr)[0,1]),cases={})
    for name,mask in selections()[0].items():
      ii=mask[ids];r['cases'][name]=dict(n=int(ii.sum()),residual=float(100*np.median(resid[ii])),density_residual=float(100*np.median(dens[ii])))
    print('RESIDUAL',r,flush=True);return r
if __name__=='__main__':
    import sys
    mode=sys.argv[1]
    if mode=='perm':out=dict(Parallel(n_jobs=4)(delayed(perm_case)(name,mask) for name,mask in selections()[0].items()))
    elif mode=='inject':out=dict(Parallel(n_jobs=4)(delayed(inject)(s) for s in [0,.5,1,2,5,10,20,40]))
    elif mode=='residual':out=residual()
    Path(f'results_final_{mode}.json').write_text(json.dumps(out,indent=2))

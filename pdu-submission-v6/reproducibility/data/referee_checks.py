"""Reproduce focused R1--R5 on frozen catalogue; no new stellar models."""
import json, platform, warnings
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
from joint_stability_classification import derive_balanced_records, draw
from pop_data_utils import tables, load_all, NM
from pop_experiment import folds, mlp
from shared_eos_evidence import fold_ids,noisy,eos_loglike,family_logz,summarize,SCENARIOS
from pdu_targeted_sensitivity import width
warnings.filterwarnings('ignore')
records,selection=derive_balanced_records()
cls,mg,obs,lam=tables(records)
assert len(records)==236

def ml_task(scen,rep,fold):
    cols,sig={'MRf0':([0,1,2],[0]*3),'ALL@0':([0,1,2,3],[0]*4),'NICER+f20':([0,1,2],[.1,.09,.2])}[scen]
    rng=np.random.default_rng([777,rep,fold,sum(map(ord,scen))])
    part=folds(cls,np.random.default_rng([777,rep]));te=np.flatnonzero(part==fold);tr=np.flatnonzero((part!=fold)&(part!=(fold+1)%5))
    masses=np.arange(0,NM,2); nd=1 if not max(sig) else 3
    xtr,ytr=draw(obs,cls,tr,masses,nd,cols,sig,rng);xte,yte=draw(obs,cls,te,masses,nd,cols,sig,rng)
    sc=StandardScaler().fit(np.log(xtr));xt=sc.transform(np.log(xtr));xq=sc.transform(np.log(xte))
    pred={'LogReg':LogisticRegression(max_iter=2000).fit(xt,ytr).predict(xq)}
    pred['MLP']=np.mean([mlp(777+10*j+fold).fit(xt,ytr).predict_proba(xq) for j in range(3)],axis=0).argmax(1)
    g=np.repeat(te,len(masses)*nd)
    return scen,[(model,int(i),confusion_matrix(yte[g==i],p[g==i],labels=np.arange(4)).tolist()) for model,p in pred.items() for i in te]

def ml_summary(tasks):
    out={};rng=np.random.default_rng(8801)
    for scen in ['MRf0','ALL@0','NICER+f20']:
      out[scen]={}
      for model in ['MLP','LogReg']:
        cms=np.zeros((len(records),4,4))
        for s,rows in tasks:
          if s==scen:
            for name,g,cm in rows:
              if name==model:cms[g]+=cm
        def metric(cm):
          den=cm[np.ix_([0,2],[0,2])].sum()
          return [np.trace(cm)/cm.sum(),(cm[0,0]+cm[2,2])/den]
        point=metric(cms.sum(0));ix=[np.flatnonzero(cls==k) for k in range(4)]
        boot=np.array([metric(cms[np.concatenate([rng.choice(i,len(i),replace=True) for i in ix])].sum(0)) for _ in range(2000)])
        out[scen][model]={'accuracy':point[0],'pair_conditional':point[1],'CI':np.quantile(boot,[.025,.975],axis=0).T.tolist(),'pooled_confusion':cms.sum(0).tolist()}
    return out

# Same full-catalogue index and fold assignment for ALL restricted experiments.
# Subsampling references only (not test) isolates reference-count sensitivity.
def ref_case(name,ids,labels):
    ids=np.asarray(ids); labels=list(labels); interp=PchipInterpolator(mg,np.log(np.concatenate((obs,lam[...,None]),axis=2)),axis=1)
    q=mg[0]+(np.arange(132)+.5)/132*(mg[-1]-mg[0]);ref=np.exp(interp(q))
    out={}; specs=SCENARIOS['three_MR_MLambda_Mf'];
    for rep in range(3):
      part=fold_ids(cls,777+rep)
      for fold in range(5):
        te=ids[part[ids]==fold];tr=ids[part[ids]!=fold]; groups=[np.flatnonzero(cls[tr]==k) for k in labels]; n=min(map(len,groups));assert n>0
        rng=np.random.default_rng([9871,rep,fold]); equal=[np.concatenate([rng.choice(a,n,replace=False) for a in groups]) for _ in range(20)]
        for g in te:
          for d in range(8):
            rng=np.random.default_rng([20261124,rep,int(g),d]);ll=np.zeros(len(tr))
            for cols,sig in specs:
              m=rng.uniform(mg[0],mg[-1]);x=noisy(np.exp(interp(m)[g,cols]),sig,rng);ll+=eos_loglike(x,ref[tr][:,:,cols],sig)
            for mode,subsets in [('full',[np.arange(len(tr))]),('equal',equal)]:
              # Average metrics across 20 reference subsamples, not posterior ensemble.
              rows=out.setdefault(mode,[])
              for sub in subsets:
                z=family_logz(ll[sub],cls[tr[sub]],labels);lp=z-logsumexp(z);p=np.exp(lp);k=labels.index(int(cls[g]))
                rows.append(dict(eos=int(g),cls=int(cls[g]),pred=labels[int(p.argmax())],correct=int(p.argmax()==k),log_true=float(lp[k]),entropy=float(-np.sum(p*lp)),log_bf_2_vs_0=float(z[labels.index(2)]-z[labels.index(0)])))
    # efficient aggregation (original summarize repeatedly scans large lists)
    def summary(rows):
      groups={};
      for r in rows:groups.setdefault(r['eos'],[]).append(r)
      # average counts/metrics by group, bootstrap class-balanced
      gkeys=list(groups); yy=np.array([groups[g][0]['cls'] for g in gkeys]); aa=np.array([np.mean([r['correct'] for r in groups[g]]) for g in gkeys]); logs=np.array([np.mean([r['log_true'] for r in groups[g]]) for g in gkeys]); ix=[np.flatnonzero(yy==k) for k in labels];rng=np.random.default_rng(8801)
      bs=np.array([np.mean([aa[rng.choice(i,len(i),replace=True)].mean() for i in ix]) for _ in range(2000)])
      cm=np.zeros((len(labels),len(labels)))
      for g,rr in groups.items():
        j=labels.index(rr[0]['cls']); cm[j]+=np.bincount([labels.index(r['pred']) for r in rr],minlength=len(labels))/len(rr)
      cm/=cm.sum(1,keepdims=True); precision=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(len(labels)),where=cm.sum(0)>0)
      return dict(accuracy=float(np.mean([aa[i].mean() for i in ix])),CI=np.quantile(bs,[.025,.975]).tolist(),score=float(np.log(len(labels))+np.mean([logs[i].mean() for i in ix])),precision=precision.tolist(),recall=np.diag(cm).tolist(),cm=cm.tolist())
    return name,dict(labels=labels,counts=[int(np.sum(cls[ids]==k)) for k in labels],**{k:summary(v) for k,v in out.items()})

def selections():
    binary=np.isin(cls,[0,2]); mx=np.array([r['Mmax']<=2.3 for r in records]);g=np.array([r['par'].get('G',-1) for r in records]); w=np.array([width(r) if r['cls'] in [0,2] else 99 for r in records])
    cases={'full':binary,'Mmax':binary&mx}
    for lo,hi in [(50,150),(150,300),(300,501)]: cases[f'G{lo}']=binary&((cls==0)|((g>=lo)&(g<hi)))
    for v in [.95,.85,.75,.65]:cases[f'w{v}']=binary&(w<=v)
    return cases,mx

def physical(cases):
    def interp(r,k,m=1.4):
      if not min(r['masses'])<=m<=max(r['masses']):return np.nan
      return float(np.exp(PchipInterpolator(r['masses'],np.log(r[k]))(m)))
    out={}
    for name,mask in cases.items():
      nuc=[r for i,r in enumerate(records) if mask[i] and r['cls']==0];dm=[r for i,r in enumerate(records) if mask[i] and r['cls']==2]
      row={'nuclear':len(nuc),'DM':len(dm)}
      for k in ['f','R']:
        a=np.array([interp(r,k) for r in nuc]); b=[]
        for r in dm:
          p=Path('pop_base')/f"b{r['idx']:04d}.json"
          if p.exists():
            base=json.loads(p.read_text()); assert base['par']==r['par'],(p,base['par'],r['par'])
            val=interp(base,k); target=interp(r,k)
            if np.isfinite(val) and np.isfinite(target):b.append(100*(target/val-1))
        spread=100*(np.quantile(a,.84)-np.quantile(a,.16))/(2*np.median(a)); signal=float(np.median(b));row[k]={'spread_percent':spread,'signal_percent':signal,'ratio':abs(signal)/spread,'pairs':len(b),'median_nuclear':float(np.median(a)),'p16_p84':np.quantile(a,[.16,.84]).tolist()}
      out[name]=row
    audit=json.load(open('results_causality_audit.json')); passed={(r['cls'],r['idx']) for r in audit['groups']['pop_data']['records'] if r['causal_full']};css=[r for r in load_all() if r['cls']==1 and (1,r['idx']) in passed];res=[]
    for r in css:
      M=1.5;R,f,tau=[interp(r,k,M) for k in ['R','f','tau']];beta=1.476625*M/R;tm=1.476625*M/299792.458;real=.8811*beta**1.5;imag=.05352*beta**4-.1849*beta**5+.1253*beta**6
      res.append(dict(idx=r['idx'],M=M,R=R,f=f,tau=tau,f_residual=100*(f/(real/(2*np.pi*tm))-1),tau_residual=100*(tau/(tm/imag)-1)))
    assert len(css)==90 and all(np.isfinite([x['f_residual'],x['tau_residual']]).all() for x in res)
    summary={k:dict(zip(['median_abs','p95_abs','max_abs'],map(float,np.quantile(np.abs([r[k] for r in res]),[.5,.95,1])))) for k in ['f_residual','tau_residual']}
    return {'signal_to_spread':out,'css':{'n':len(res),'summary':summary,'rows':res}}

if __name__=='__main__':
    import sys
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    cases,mx=selections()
    if mode in ['physical','all']:
      p=physical(cases);Path('results_physical_referee.json').write_text(json.dumps(p,indent=2));print('PHYSICAL',json.dumps({**p,'css':{k:v for k,v in p['css'].items() if k!='rows'}}),flush=True)
    if mode in ['ml','all']:
      a=Parallel(n_jobs=4,verbose=5)(delayed(ml_task)(s,r,f) for s in ['MRf0','ALL@0','NICER+f20'] for r in range(3) for f in range(5));out=ml_summary(a);Path('results_ml_referee.json').write_text(json.dumps(out,indent=2));print('ML',json.dumps(out),flush=True)
    if mode in ['ref','all']:
      jobs=[(name,np.flatnonzero(mask),[0,2]) for name,mask in cases.items()]+[('four_full',np.arange(len(records)),[0,1,2,3]),('four_cut',np.flatnonzero(mx),[0,1,2,3]),('three_full',np.flatnonzero(cls!=1),[0,2,3]),('three_cut',np.flatnonzero(mx&(cls!=1)),[0,2,3])]
      out=dict(Parallel(n_jobs=4,verbose=5)(delayed(ref_case)(*j) for j in jobs));Path('results_refsize_referee.json').write_text(json.dumps(out,indent=2));print('REF',json.dumps(out),flush=True)

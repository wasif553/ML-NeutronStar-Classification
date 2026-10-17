import numpy as np, warnings; warnings.filterwarnings("ignore")
import main
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
D=main.D
def densify(df, cols, k):
    out=[]
    for c in sorted(df.y.unique()):
        s=df[df.y==c].sort_values('M'); X=s[cols].values.astype(float)
        t=np.arange(len(X)); tt=np.linspace(0,len(X)-1,(len(X)-1)*k+1)
        Xi=np.column_stack([np.interp(tt,t,X[:,j]) for j in range(X.shape[1])])
        out.append(pd.DataFrame(Xi,columns=cols).assign(y=c))
    return pd.concat(out,ignore_index=True)
import pandas as pd
def run(rg,fs,s,k,seed):
    rng=np.random.default_rng(seed); d=regime(D,rg); cols=FEATS[fs]; classes=np.array(sorted(d.y.unique()))
    role=block_split(d,rng); tr,te=d[role==0],d[role==2]
    trd=densify(tr,cols,k) if k>1 else tr
    reps=max(1,20//k)
    Xtr,ytr,_=main.prep(trd,cols,s,reps,rng,classes); Xte,yte,_=main.prep(te,cols,s,30,rng,classes)
    sc=StandardScaler().fit(logfeat(Xtr)); T=lambda X: sc.transform(logfeat(X))
    ens=[MLPClassifier((64,64),alpha=1e-4,max_iter=200,early_stopping=True,n_iter_no_change=15,random_state=seed*10+j).fit(T(Xtr),ytr) for j in range(3)]
    p=np.mean([m.predict_proba(T(Xte)) for m in ens],0)
    bo=np.exp(bayes_logpost(Xte,d[cols].values.astype(float),d.y.values,cols,s,classes)).argmax(1)
    return (p.argmax(1)==yte).mean(),(bo==yte).mean()
for s in (0.25,1.0):
    for k in (1,5,10):
        r=np.array([run('B','SO',s,k,sd) for sd in range(4)]); print(s,k,r.mean(0).round(3))

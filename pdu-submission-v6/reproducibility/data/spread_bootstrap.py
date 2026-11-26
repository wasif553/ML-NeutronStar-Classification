"""Conditional cluster bootstrap of the descriptive signal/spread at fixed mass."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from referee_checks import records,selections
cases,_=selections();rng=np.random.default_rng(7712)
def interp(r,k):return float(np.exp(PchipInterpolator(r['masses'],np.log(r[k]))(1.4)))
out={}
for name,mask in cases.items():
 n=[r for i,r in enumerate(records) if mask[i] and r['cls']==0];d=[r for i,r in enumerate(records) if mask[i] and r['cls']==2]
 out[name]={}
 for k in ['f','R']:
  a=np.array([interp(r,k) for r in n]);b=np.array([100*(interp(r,k)/interp(json.load(open(Path('pop_base')/f"b{r['idx']:04d}.json")),k)-1) for r in d]);aa=rng.choice(a,(2000,len(a)),replace=True);bb=rng.choice(b,(2000,len(b)),replace=True);sp=100*(np.quantile(aa,.84,axis=1)-np.quantile(aa,.16,axis=1))/(2*np.median(aa,axis=1));si=np.median(bb,axis=1);rat=np.abs(si)/sp
  out[name][k]={'spread_ci':np.quantile(sp,[.025,.975]).tolist(),'signal_ci':np.quantile(si,[.025,.975]).tolist(),'ratio_ci':np.quantile(rat,[.025,.975]).tolist()}
Path('results_spread_bootstrap.json').write_text(json.dumps(out,indent=2));print(out['full'])

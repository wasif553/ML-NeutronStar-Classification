import json,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=json.load(open('results_final_inject.json'));u=json.load(open('results_submission_checks.json'));v=json.load(open('results_submission_checks_q528.json'))
fig,axs=plt.subplots(1,2,figsize=(10.3,4.0),sharey=True)
for k,label,marker in [('MR','MR','o'),('MRf','MRf','s'),('three','Three stars','^'),('precise','Precise three stars','D')]:
 x=np.array([float(i) for i in r]);z=np.array([j[k] for j in r.values()])*100
 axs[0].errorbar(x,z[:,0],yerr=np.array([z[:,0]-z[:,1],z[:,2]-z[:,0]]),marker=marker,ms=3.2,lw=1,capsize=2,label=label)
for k,source,label,marker in [('precise',u,'Precise','D'),('low_noise',v,'Low noise','o')]:
 d=source['results'][k];x=np.array([float(i) for i in d]);y=np.array([j['mean'] for j in d.values()])*100;ci=np.array([j['pointwise_95'] for j in d.values()])*100;lo=np.array([j['simultaneous_lower_95'] for j in d.values()])*100
 line=axs[1].errorbar(x,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),marker=marker,ms=3.2,lw=1,capsize=2,label=label);axs[1].plot(x,lo,ls='--',lw=1,color=line[0].get_color(),label=f'{label}: lower bound')
for ax,t in zip(axs,['(a) Original injection curves','(b) Extended lower-noise curves']):
 ax.axhline(50,color='0.3',ls=':',lw=1);ax.axhline(75,color='0.5',ls=':',lw=.8);ax.set_xscale('symlog',linthresh=.5);ax.set_xticks([0,.5,1,2,5,10,20,40]);ax.set_xticklabels(['0','0.5','1','2','5','10','20','40'],fontsize=8);ax.set_xlabel('Observable-shift multiplier $a$');ax.set_ylim(40,101);ax.grid(alpha=.18);ax.legend(fontsize=7,loc='upper left',frameon=False);ax.set_title(t,fontsize=10)
axs[0].set_ylabel('Balanced accuracy (%)');fig.tight_layout();fig.savefig('fig_injection.pdf');plt.close(fig)

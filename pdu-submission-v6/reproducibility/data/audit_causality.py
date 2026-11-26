"""Audit sound-speed causality throughout the stable star, not only at the
maximum-mass central pressure. This is an EoS validation diagnostic; it does
not silently remove EoSs from the published classification population.
"""
import json
from pathlib import Path
import numpy as np
from families import build, max_mass
import families_b
from pop_data_utils import load_all

OUT=Path(__file__).with_name("results_causality_audit.json")
TOL=1e-3

def main():
    groups={}
    for directory, limit in [("pop_data",90),("pop_data_B",48)]:
        if not Path(directory).exists():
            continue
        recs=load_all(directory)
        filtered=[]
        counts={}
        for r in recs:
            c=r["cls"]
            counts[c]=counts.get(c,0)+1
            if counts[c]>limit:continue
            eos=(build if directory=="pop_data" else families_b.build)(r["par"])
            maxm,lpmax=max_mass(eos)
            center=np.exp(lpmax)
            psurf=eos.p_surface()
            p_lo=max(psurf*(1+1e-8),1e-16)
            grid=np.geomspace(p_lo,center,256)
            cs=np.asarray([eos.cs2(float(p)) for p in grid])
            k=int(np.argmax(cs))
            row=dict(idx=r["idx"],cls=c,max_cs2=float(cs[k]),
                     cs2_center=float(cs[-1]),pressure_at_max_cs2=float(grid[k]),
                     causal_full=bool(np.isfinite(cs).all() and np.all(cs>0) and np.all(cs<=1+TOL)),
                     center_only_causal=bool(0<cs[-1]<=1+TOL),
                     Mmax_Msun=float(maxm))
            filtered.append(row)
        per={}
        for c in sorted(set(x["cls"] for x in filtered)):
            arr=[x for x in filtered if x["cls"]==c]
            offenders=sorted([x for x in arr if not x["causal_full"]],
                             key=lambda x:-x["max_cs2"])
            per[str(c)]=dict(total=len(arr),full_causal=sum(x["causal_full"] for x in arr),
                             center_causal=sum(x["center_only_causal"] for x in arr),
                             noncausal_count=len(offenders),worst_max_cs2=max(x["max_cs2"] for x in arr),
                             noncausal_ids=[x["idx"] for x in offenders[:30]])
        groups[directory]=dict(per_class=per,records=filtered)
        print(directory,json.dumps(per),flush=True)
    OUT.write_text(json.dumps(dict(tol=TOL,description=__doc__,groups=groups),indent=1))
    total=sum(x["noncausal_count"] for a in groups.values() for x in a["per_class"].values())
    print("NONCAUSAL_POPULATION_ENTRIES",total)
    # Deliberately do not fail the workflow: this is an audit, not silent data sanitization.

if __name__=="__main__":
    main()

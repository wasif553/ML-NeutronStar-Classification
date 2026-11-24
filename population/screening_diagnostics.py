"""G and Mmax diagnostic on existing accepted Population A, not raw proposals."""
import json
import numpy as np
from pathlib import Path
from pop_data_utils import load_all

def main():
    records=load_all()
    assert len(records)==360
    audit=json.loads(Path("results_causality_audit.json").read_text())
    passed={(int(r["cls"]),int(r["idx"])) for r in audit["groups"]["pop_data"]["records"] if r["causal_full"]}
    bins=[50,100,200,300,400,500]
    dm=[r for r in records if r["cls"]==2]
    g=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        group=[r for r in dm if lo<=r["par"]["G"]<hi]
        causal=[r for r in group if (2,r["idx"]) in passed]
        joint=[r for r in causal if 937.9<r["par"]["mchi"]<=938.783]
        g.append(dict(G_range=[lo,hi],initial_accepted=len(group),causal=len(causal),joint=len(joint)))
    limits={}
    for m in [2.2,2.3,2.4,2.5,3.0]:
        limits[str(m)]={str(k):sum(r["cls"]==k and r["Mmax"]<=m and (k,r["idx"]) in passed for r in records) for k in range(4)}
    out=dict(G_screening=g,Mmax_sensitivity_counts=limits,
             caveat="Counts are conditional on previously accepted catalogue, not original rejection-sampling proposal acceptance rates; no new classifier is fitted.")
    Path("results_screening_diagnostics.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=="__main__":main()

"""Run the existing extended analyses on screened records in isolated folders."""
import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample",choices=["joint","causal"],default="joint")
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--only-prepare",action="store_true")
    args=ap.parse_args()
    os.chdir(ROOT)
    if args.sample=="joint":
        from joint_stability_classification import derive_balanced_records
    else:
        from causal_filtered_classification import derive_balanced_records
    records,meta=derive_balanced_records()
    destination=ROOT/"screened_reanalysis"/args.sample
    data=destination/"catalogue"
    data.mkdir(parents=True,exist_ok=True)
    expected=set()
    for r in records:
        name=f"c{r['cls']}_{r['idx']:04d}.json"
        expected.add(name)
        (data/name).write_text(json.dumps(r))
    assert {p.name for p in data.glob("c*.json")}==expected
    (destination/"selection.json").write_text(json.dumps(dict(selection=meta,ids=sorted(expected)),indent=2))
    print("SCREENED_CATALOGUE",destination,len(records),flush=True)
    if args.only_prepare:return
    env=os.environ.copy()
    env.update(POP_DIR=str(data),N_PER_CLASS=str(meta["used_per_class"]),OUT_SUFFIX="",
               OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
    env.pop("MCHI_MAX",None)
    for name,arguments in [
        ("pop_experiment.py",["all",str(args.workers)]),
        ("stat_robust.py",[str(args.workers)]),
        ("run_group_calibration.py",["NICER+f20,SO@1.0",str(args.workers),"3"])
    ]:
        subprocess.run([sys.executable,str(ROOT/name),*arguments],cwd=destination,env=env,check=True)
    print("SCREENED_EXTENDED_ANALYSIS_COMPLETE",destination)
    print("Review and regenerate all manuscript tables/figures before citing these outputs.")

if __name__=="__main__":main()

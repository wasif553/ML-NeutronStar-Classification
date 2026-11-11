"""Create LaTeX table and numerical macros for jointly screened PDU sensitivity."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "population" / "results_joint_stability_classification.json"
TABLE = ROOT / "tables" / "tab_joint_stability.tex"
MACROS = ROOT / "results" / "macros_joint_stability.json"

def main():
    res=json.loads(SOURCE.read_text(encoding="utf-8"))
    n=int(res["selection"]["used_per_class"])
    assert 45<=n<=63 and int(res["selection"]["total"])==4*n
    assert res["selection"]["dark_fermion_mev_window"]==[937.9,938.783]
    rows=[]
    out={"jointN":str(n),"jointTotal":str(4*n)}
    for scen,label,slug in [("ALL@0","Noise-free","zero"),("NICER+f20","NICER-like + 20\\% $f$","noisy")]:
        for model,name,mslug in [("MLP","MLP ensemble","mlp"),("LogReg","Logistic regression","lr")]:
            m=res["scenarios"][scen][model]
            acc=100.0*float(m["accuracy_mean"])
            pair=100.0*float(m["pair_accuracy_conditional"])
            assert 0<=acc<=100 and 0<=pair<=100
            assert np.isclose(float(np.mean(m["repetition_means"])),acc/100,atol=1e-10)
            out[f"joint_{slug}_{mslug}"]=f"{acc:.1f}"
            out[f"joint_{slug}_{mslug}_pair"]=f"{pair:.1f}"
            rows.append(f"{label} & {name} & {acc:.1f} & {pair:.1f} "+r"\\")
    TABLE.write_text("\n".join(rows).rstrip()[:-2],encoding="utf-8")
    MACROS.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("JOINT_STABILITY_TABLE_PASS",json.dumps(out,sort_keys=True))

if __name__=="__main__":
    main()

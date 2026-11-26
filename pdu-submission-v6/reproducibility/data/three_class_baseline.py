"""One-star MR reference comparison after excluding self-bound EoSs."""
import json,numpy as np
from pathlib import Path
import referee_checks as rc
from shared_eos_evidence import SCENARIOS
SCENARIOS['three_MR_MLambda_Mf']=SCENARIOS['one_MR']
_,mx=rc.selections()
a=dict(rc.ref_case(n,np.flatnonzero(m),[0,2,3]) for n,m in [('three_full',rc.cls!=1),('three_cut',mx&(rc.cls!=1))])
Path('results_three_class_mr.json').write_text(json.dumps(a,indent=2));print(json.dumps(a),flush=True)

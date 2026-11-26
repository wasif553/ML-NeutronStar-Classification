"""Run held-out-EoS class-conditional conformal experiments from the existing population.

Usage from population/:
  python run_group_calibration.py NICER+f20,SO@1.0 2 1
Arguments: comma-separated scenarios, job workers, number of repetitions (1..3).
The 5 folds are always evaluated; NO calibration EoS is in a test fold.
"""
import sys, json
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from pop_experiment import task, SCEN, K_FOLDS


def main():
    scenarios = (sys.argv[1] if len(sys.argv) > 1 else "NICER+f20,SO@1.0").split(",")
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    assert all(s in SCEN for s in scenarios), scenarios
    assert 1 <= reps <= 3
    jobs = [(s,rep,fold) for s in scenarios for rep in range(reps) for fold in range(K_FOLDS)]
    out = Parallel(n_jobs=workers, verbose=5)(delayed(task)(*job) for job in jobs)
    rows = [row for result, _ in out for row in result if row["model"] == "MLP-ens+T-Mondrian"]
    df = pd.DataFrame(rows)
    df.to_csv("results_group_mondrian.csv", index=False)
    summary = []
    for s in scenarios:
        sub = df[df.scen == s]
        x = {"scenario": s, "replications": reps, "folds_per_rep": K_FOLDS,
             "mean_coverage": float(sub["cov"].mean()),
             "mean_set_size": float(sub["setsize"].mean()),
             "minimum_class_coverage_across_folds": float(sub[[f"cov_class_{k}" for k in range(4)]].min().min()),
             "calibration_eos_per_class": [int(sub.ncal_min.min()),int(sub.ncal_max.max())]}
        for k in range(4):
            x[f"coverage_class_{k}"] = float(sub[f"cov_class_{k}"].mean())
        summary.append(x)
        print("GROUP_MONDRIAN_SUMMARY", json.dumps(x), flush=True)
    Path("results_group_mondrian_summary.json").write_text(json.dumps(summary,indent=2))
    assert (df["cov"].between(0,1) & df["setsize"].between(0,4)).all()
    print(f"SUCCESS: {len(df)} rows")


if __name__ == "__main__":
    main()

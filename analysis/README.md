# Revised study: ML classification of neutron-star matter models

`paper_final.pdf` is the revised manuscript (source `paper_final.tex.in`; `build.py` fills in every number from the saved
results and runs pdflatex). It has two parts:

1. the single-sequence benchmark of the original repository (audit, leakage, noise, oracle bound) — scripts in this folder,
   data `../DataML.xlsx`;
2. the population study (360 EoS, whole-EoS hold-out) — code and data in `../population/`.

## Single-sequence analysis (run from this folder)

```
pip install numpy pandas scipy scikit-learn matplotlib openpyxl joblib
python audit.py          # data audit
python repro.py          # replication of the original random-split protocol, 100 seeds
python blocksize.py      # accuracy vs held-out block length
python main.py           # noise / calibration / conformal study (540 tasks, ~15 min on 6 cores)
python extrap.py         # extrapolation to an unseen mass window
python interp_check.py   # optional densification check
python figs_a.py && python report.py   # figures, tables, results/macros.json
python build.py paper_final            # needs ../population results copied to results/macros_pop.json, macros_val.json
```

`results/macros_*.json` are produced by the report scripts in `../population/` (`pop_report.py`, `extra_report.py`,
`paired_analysis.py`, `robust_report.py`, `robust_report2.py`, `published_tau_comparison.py`; copies are included). Seeds are fixed in every script.

Environment used: Python 3.13, scikit-learn 1.7.2, numpy 2.1.2, pandas 2.2.3, scipy 1.15.2.

## Open items for the author
- Complete the `Husain (2026a)` reference (the earlier oscillation code) in `paper_final.tex.in`.
- Create the release tag named in the paper (`v2.0.0`) and deposit an archived copy (DOI) before submission; the Data/code section of the paper must then give the DOI.
- The f, tau, Q and h_c values of the single-sequence data were not recomputed; the redshift inconsistency of the
  nucleonic and strange classes should be traced back to the data-generation code.
- Verify the bibliography entries against the sources.

# PDU submission draft — 24 November 2026 (GitHub and supplementary archive)

This is the current Physics of the Dark Universe submission draft, superseding all earlier PDU and ASCOM packages. The previous scientific results are retained; final targeted checks add the tidal shifts, low-noise injection curve and omnibus permutation statistic. Compile `pdu_submission.tex` with `pdflatex -interaction=nonstopmode -halt-on-error pdu_submission.tex` three times. Figures and tables are included and the bibliography is inline. The separate research archive contains the frozen numerical inputs, source code, environment and results.

## Reproduction
Frozen catalogue: GitHub commit 67d6b378fb7dc10314d1dcd6dabdfc4e91def45b, export run 45324508646 (artifact 14887934767). Original data provenance: commit 8b955ca124dc0a44e54819278c205200d716cae2. Earlier shared-EoS calculation: run 45310397915 (artifact 14883007688). Earlier sensitivity experiment: run 45319374977 (artifact 14885929795); its renumbered seeds are superseded for Table 7. New local computations use the frozen catalogue and the exact package versions in `data/requirements-referee.txt`.

From the extracted research archive, change into `data/`. Install requirements-referee.txt. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. Execute:

```
python referee_checks.py physical
python spread_bootstrap.py
python referee_checks.py ml
python referee_checks.py ref
python three_class_baseline.py
python final_checks.py residual
python final_checks.py perm
python final_checks.py inject
```

The new outputs record all R1–R5 results. The conditional bootstrap intervals resample EoSs, not stellar rows or reference sets. The paired nuclear baseline parameters are asserted identical to the admixed models. Full-references and equal-references controls retain exactly the same test observations, folds and noise seeds; equal references subsample within fold 20 times. The old JSON files are retained for provenance, but the new Table 7 uses only the matched-seed results. All 90 causal CSS models are included in the compactness consistency check.

## Access and provenance
The frozen source and data are available at the pinned GitHub commit above. This archive provides the exact additional local analysis scripts, numerical outputs and environment information. Existing copyright and third-party terms apply. The accompanying manuscript does not claim an independent repository DOI or a new software licence.

## Final diagnostics
The conditional EoS-label permutation uses 199 shuffled assignments per restriction, with re-stratification and fixed noise seeds. The matched-pair injection uses scaled log-observable shifts with pair-disjoint holdout, common random numbers and pair-bootstrap intervals. The multiplier is not a physical dark abundance or EoS. The structural residual uses the independent screened nuclear family and leave-one-out residuals.

Verify integrity using `sha256sum -c SHA256SUMS.txt` from the extracted archive root.

## Final targeted submission checks
From `data/`, with the existing single-thread environment settings, execute:
```
python submission_checks.py
PDU_INJECTION_NQUAD=528 python submission_checks.py
python plot_submission_injection.py
```
On Windows PowerShell set `$env:PDU_INJECTION_NQUAD = "528"` before the second run (unset it for the 132-point run). The new results contain paired accuracy contributions, 50,000-resample conditional intervals, simultaneous one-sided lower bounds, median paired tidal shifts and the dependence-robust omnibus combination of the nine existing marginal permutation probabilities. The 528-point run checks the low-noise curve only; both curves use the same conservative multiplicity denominator of 20. The original precise results at shared grid points reproduce exactly. No physically new EoSs are generated.

## v6 minor revision: reference-density control
From the extracted supplementary archive `data/` directory, run `python reference_density_check.py` with the same single-thread environment and requirements. The new output `results_reference_density_check.json` contains per-pair results, conditional bounds and reference fractions. The 100% case is asserted to reproduce the archived 132-point result. For 50% and 75% the script averages accuracy over 20 nested subsets per fold; conditional bounds are simultaneous within each fraction, not across fractions. No new stellar models were generated.

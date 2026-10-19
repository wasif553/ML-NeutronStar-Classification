# Referee-facing revision record (19 October 2026)

This branch is based on `revised-analysis`, **not the older 16-page PDF**. The parent branch already contains new population, cross-family, frozen-response, numerical, and EoS-level bootstrap experiments; this branch performs a focused scientific and editorial revision of the manuscript. No new numerical experiment was run by this change.

## Changes incorporated into `analysis/paper_final.tex.in` and `analysis/paper_final.tex`

1. **Scope and title page.** Removed the internal revision date and made the abstract explicitly conditional on the chosen phenomenological EoS populations, model priors, and oscillation and noise assumptions.
2. **Physical mass window.** Flagged that the original exploratory `mχ ≤ 939.2 MeV` prior exceeds the approximately `937.9–938.783 MeV` window for the chosen stable-χ decay channel. Out-of-window samples cannot support model-specific physical claims; the restricted-population experiment from the parent branch is highlighted as a sensitivity analysis.
3. **Baryon-number bookkeeping.** Assigned `Bχ=1` in the baryon-conserving benchmark and distinguished the density of ordinary nuclear baryons appearing in the thermodynamic baryonic chemical potential from the total conserved baryon density `nB,total = nn + np + nχ`.
4. **Perturbation physics.** Interpreted the 24-EoS frozen versus equilibrium response experiment as a test under **common fluid displacement**, not as validation of fully independent two-fluid normal modes or unknown conversion timescales.
5. **Numerical precision.** Corrected “seven orders” to “more than five orders” when comparing a relative (3×10^{-8}) solver perturbation with a median (0.56\% = 5.6×10^{-3}) dark-matter shift. Explicitly typeset numerical sensitivities in scientific notation.
6. **Benchmark integrity.** Replaced claims of an exact general-information bound with the valid statement that the finite-reference Gaussian kernel is a model-conditional comparator. Explicitly noted that positivity-truncated synthetic noise at high error scales is not the same as its untruncated Gaussian likelihood.
7. **Prediction sets.** Corrected the inference from pooled, correlated calibration scores. The empirical 90%-target coverage is retained, but **no distribution-free, finite-sample conformal guarantee is claimed**. Formal group-conformal calibration would require a new EoS-level score definition and a recalculation.
8. **Empirical interpretations.** Reworded confidence calibration, damping-time utility, and the tested information plateau as observed, model-conditional findings rather than universal theorems.
9. **Published citations.** Updated the 2022 JCAP neutron-decay paper and 2026 Physics of the Dark Universe f-mode paper to proper journal citations and DOIs.

## Robustness evidence already present on `revised-analysis`

- Population A: 360 EoS; alternative population B: 192 EoS (48/class), using different phenomenological EoS parametrizations.
- Cross-family transfer results in `population/results_cross_family.json`.
- Frozen-vs-equilibrium common-displacement results for 24 EoS in `population/results_frozen.json`: median f-mode shift 0.64% versus 0.63%, difference around 1.1% of the shift, max around 0.021% of f.
- Cluster (EoS-level) bootstrap estimates and learning-curve results in `population/macros_robust.json`: population A noise-free MLP accuracy 64% [61,67], nucleonic-vs-DM pair 51% [45,56]; MLP population B noise-free accuracy 49%, and transfer MLP 52% A-to-B versus 54% B-to-A.
- Restricted stable-χ window experiment and independent black-hole QNM check are part of the parent branch.

## Remaining evidence required for a particularly demanding PRD referee

- An EoS population derived from **independent microphysics**, such as a validated QMC/hyperonic ensemble, rather than only differing phenomenological parametrizations.
- Fully coupled **two-fluid** eigenmode calculations or a justified bound for modes with independent baryonic/dark displacements and finite conversion rates.
- **Independent published neutron-star damping-time benchmarks**; the black-hole exterior benchmark is insufficient for the entire stellar eigenproblem.
- A new, correctly defined **group-conformal** experiment if finite-sample coverage guarantees are desired.
- Correlated real observational posterior treatment for NICER/GW170817 if making claims about actual observed stars.

**Verification status:** Source text was updated through the connected GitHub repository; PDF compilation and all numerical calculations were **not rerun during this edit**. Rebuild with the existing repository build workflow and verify generated tables/figures before submission.

## Rebuild checklist

Run the existing analysis instructions, re-create macro/table outputs if intended, regenerate `analysis/paper_final.tex` from `analysis/paper_final.tex.in`, compile with pdfLaTeX, inspect all figures and tables, and compare the newly generated `.tex` against this editorially updated file before replacing it. The template is the canonical source; do not submit an older pre-edit PDF.

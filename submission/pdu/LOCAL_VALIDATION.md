# Local validation (Windows, PowerShell)

Clone the review branch or switch an existing clean clone to it. Use a separate worktree for reruns. Existing manuscript results are frozen; generated PDFs are stale until rebuilt.

Run from the repository root using an environment with NumPy, SciPy, scikit-learn, pandas, joblib, matplotlib and pdfLaTeX:

    cd population
    python shared_eos_evidence.py --self-test
    python screening_diagnostics.py
    python shared_eos_evidence.py --sample joint --draws 8 --reps 3 --quadrature 66 132 264 --output results_shared_eos_joint.json
    python shared_eos_evidence.py --sample causal --draws 8 --reps 3 --quadrature 132 264 --output results_shared_eos_causal.json
    cd ../analysis
    python causal_primary.py
    python build.py pdu_submission

The shared-EoS evidence test computes log likelihood per EoS, marginalizing each stellar mass with midpoint quadrature, and multiplies likelihoods of independent stars at fixed EoS before averaging over the family. It compares one-star and mixed multi-star observations including tidal deformability. The binary task is separately normalized over nucleonic and DM reference classes. Cluster bootstrap intervals resample entire EoSs, not individual stars. The predictive information score is log(K) minus held-out cross entropy; it is not the exact mutual information of the underlying continuous priors. Compare quadrature resolutions before interpreting results.

The screening diagnostic measures retention conditional on already accepted records, not original proposal acceptance. Expanded-G screening, pressure-curve fit, fully causal extended learner rerun, independent stellar mode benchmarks, old-code reconciliation and archival/licence signoff remain mandatory. Do not upload this candidate to a journal until scientific review is closed.

The GitHub Actions workflow produces an explicitly non-submission candidate archive if it runs successfully. It does not publish a release or change the original review branch.

## Extended screened experiments

These preserve the frozen original results. Use the same Python interpreter as above.

    cd population
    python screened_reanalysis.py --sample joint --workers 4
    python screened_reanalysis.py --sample causal --workers 4

They write new CSV/JSON in screened_reanalysis/<sample>/, not into the archived result files. No automatically regenerated manuscript table is considered final until reviewed. The original statistical and calibration outputs cannot be relabelled as screened.

## Physical interaction and mimicry pilot

    python dark_prior_mimicry.py --proposals 8 --fit-count 8 --workers 4

Review mode failures, pressure overlap, and screening counts before expanding to --proposals 100 --fit-count 59. All proposal failures are recorded. Seven G strata cover 5–2000 fm² with stable masses; the same nuclear parameter and dark mass draws recur across G strata. The optional Mmax<=2.3 cut is a sensitivity diagnostic, not an absolute observational bound. The PP fit minimizes log eps(P) on alternating pressure points and checks P(eps) on the remaining points; it retains physical screen and stellar-observable residuals. It is not a substitute for independent published mode validation.

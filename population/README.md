# Population study (`population/`)

Populations of neutron-star EoS (nucleonic, hyperonic-like, strange, neutron-decay dark-matter-admixed), solved with a
validated full-GR TOV / tidal / f-mode code, and classified with whole-EoS hold-out. See `../analysis/paper_final.pdf`.

## Run order (from this folder)

```
pip install numpy scipy pandas scikit-learn matplotlib joblib sympy
# 1. solver validation (needs only the six published reference values in test_tov.py)
python fit_crust.py        # one crust parameter, fitted to R and Lambda of six EoS at 1.4 Msun
python validate.py         # f held out: results_validation.json
# 2. (optional) regenerate the symbolic perturbation equations
python derive_vac2.py && python derive_matter.py && python prep_matter.py && python elim.py   # writes matter_rhs.py
# 3. generate the population (about 200 s of computation per EoS; resumes if interrupted)
python gen_pop.py 110 12   # writes pop_data/*.json ; the paper uses the first 90 per class (pop_data_utils.N_PER_CLASS)
# dark-matter EoS are drawn with m_chi in the stability window [937.9, 938.783] MeV; pop_data_utils.load_all keeps only
# in-window dark-matter EoS by default (MCHI_MAX=938.783). Records idx < 90 with m_chi > 938.783 belong to the earlier
# wide-prior population (m_chi up to 939.2 MeV); MCHI_MAX=inf reproduces it (sensitivity run).
python gen_base.py 14      # nuclear base EoS (no dark component) for every dark-matter EoS of the population (paired analysis)
# 4. experiments, figures, tables
python pop_experiment.py all 14      # results_pop_main.csv, results_pop_conf.json (primary, in-window population)
MCHI_MAX=inf OUT_SUFFIX=_wide python pop_experiment.py all 14   # wide-prior sensitivity (copies of the earlier run are in wide_prior_results/)
python pop_report.py && python extra_report.py   # figures, tables/, macros_pop.json, macros_val.json
python paired_analysis.py && python dm_prescription.py 90 14 && python frozen_test.py   # dark matter at fixed nuclear EoS, m0 prescription, frozen composition
python published_tau_comparison.py   # damping times against Lioutas & Stergioulas (2018) Eq. (20) (macros_published.json)
python tau_diagnostics.py && python dm_diagnostics.py && python paired_by_mass.py && python pair_subsets.py   # tau/crust check, dark-component onset, shift vs mass, subsets
python tuned_learners.py 4           # tuned learners and engineered features (results_tuned.json)
python gen_narrow.py 40 14 0.4,0.2,0.1,0.05,0.02 && python narrow_analysis.py 40   # EoS in narrowed nuclear windows about SLy (pop_narrow/, results_narrow.json)
python report_new.py                 # tables/tab_tuned.tex, tab_narrow.tex, figs/figP8_narrow.pdf, macros_new.json
```

The included `pop_data/` and `results_*` files are the ones used in the paper; the paper's numbers are generated from them
by `../analysis/build.py`.

## Modules
- `eos.py` – PP EoS and crust, `tov.py` – TOV and tidal deformability, `fmode2.py` + `matter_rhs.py` – full-GR f-mode
  (interior equations derived by `derive_*.py`/`elim.py`; exterior Zerilli matching in `fmode.py`/`fmode2.py`)
- `families.py` – the four EoS families, screening priors and the dark-matter effective barotrope
- `pop_data_utils.py`, `pop_experiment.py`, `unc.py` – data handling, grouped cross-validation, learners, calibration, conformal sets
- `test_tov.py` – the six published reference EoS fits (R, Lambda, f at 1.4 Msun) used for validation

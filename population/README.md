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
# 4. experiments, figures, tables
python pop_experiment.py all 14      # results_pop_main.csv, results_pop_conf.json
python gen_base.py 14                # paired nuclear-base observables of the dark-matter EoS (pop_base/)
python paired_analysis.py            # known-nuclear-EoS dark-matter detectability, macros_paired.json
python pop_report.py && python extra_report.py   # figures, tables/, macros_pop.json, macros_val.json
```

The included `pop_data/` and `results_*` files are the ones used in the paper; the paper's numbers are generated from them
by `../analysis/build.py`.

## Modules
- `eos.py` – PP EoS and crust, `tov.py` – TOV and tidal deformability, `fmode2.py` + `matter_rhs.py` – full-GR f-mode
  (interior equations derived by `derive_*.py`/`elim.py`; exterior Zerilli matching in `fmode.py`/`fmode2.py`)
- `families.py` – the four EoS families, screening priors and the dark-matter effective barotrope
- `pop_data_utils.py`, `pop_experiment.py`, `unc.py` – data handling, grouped cross-validation, learners, calibration, conformal sets
- `test_tov.py` – the six published reference EoS fits (R, Lambda, f at 1.4 Msun) used for validation

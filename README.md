# ML-NeutronStar-Classification
Code and data for the paper: Exploring the Limits of Machine-Learning Classification of Neutron-Star Matter Models.

This repository contains the computational framework and synthetic datasets for the paper: 
"Exploring the Limits of Machine-Learning Classification of Neutron-Star Matter Models".

Overview
We provide a machine-learning pipeline to classify four distinct families of neutron star equations of state (EoS) using macroscopic and oscillation observables (M, R, f, Q, z, \tau, h_c).

How to Reproduce Results
1. Install dependencies:
   `pip install -r requirements.txt`
2. Run the analysis:
   `python reproduce_all_figures.py`
3. Generate QMC data
   QMC.jl

## Revised study (`analysis/`, `population/`)
`analysis/` holds the audit of the single-sequence benchmark (leakage-free blocked validation, observational-error model
with an oracle Bayes bound, calibrated/conformal uncertainty) and the revised manuscript `analysis/paper_final.pdf`
(see `analysis/README.md`). `population/` holds the population study: 360 EoS (90 per composition scenario) solved with a
validated full-GR TOV / tidal / f-mode code and classified with whole-EoS hold-out (see `population/README.md`).
`reproduce_all_figures.py` reproduces the original random-split results only; those results are inflated by
near-duplicate neighbouring models (see the revised paper, Sec. 2).

Data notes: `DataML.xlsx` holds 770 models (222 nucleonic, 220 strange, 122 dark-matter, 206 hyperonic); the
`Surface_Redshift` column is inconsistent with (M, R) for the nucleonic and strange classes and is not used in the
revised analysis.

## Citation
If you use this code or data, please cite:
Husain, W. (2026). *Physics of the Dark Universe* (https://doi.org/10.1016/j.dark.2025.102208) and *Astronomy and Computing* (ML Framework).

"""External astrophysical consistency check for computed stellar f-mode damping times.

Compare the six 1.4 Msun TOV/f-mode results with the published empirical
relation of Benhar, Ferrari & Gualtieri, Phys. Rev. D 70, 124015 (2004).
The relation was fitted to distinct stellar models and is not an exact or
independent eigenfrequency calculation for these particular EoSs.
Its residuals must NOT be described as a validation of the full interior
complex-frequency eigenproblem.
"""
import json
from pathlib import Path
import numpy as np

SOURCE = "https://doi.org/10.1103/PhysRevD.70.124015"
M_SUN_KM = 1.47662504  # GM_sun/c^2 in km
C_KM_S = 299792.458


def benhar_tau(mass_solar, radius_km):
    """Relation tau = (R^4/M^3)/(c*(0.087 - 0.271*M/R)), M,R in km."""
    mass_km = float(mass_solar) * M_SUN_KM
    R = float(radius_km)
    coeff = 0.087 - 0.271 * mass_km / R
    if mass_km <= 0 or R <= 2 * mass_km or coeff <= 0:
        raise ValueError("Outside the physically intended compactness range")
    return R**4 / mass_km**3 / (C_KM_S * coeff)


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "results_validation.json").read_text())["six"]
    result = []
    for item in data:
        fit_tau = benhar_tau(1.4, item["R"])
        residual = 100 * (item["tau"] / fit_tau - 1.0)
        result.append({"eos":item["eos"], "M_Msun":1.4,
                       "R_km":item["R"], "tau_solver_s":item["tau"],
                       "tau_empirical_s":fit_tau, "signed_residual_percent":residual})
    absolute = [abs(r["signed_residual_percent"]) for r in result]
    output = {"source":SOURCE, "note":"Independent empirical-relation comparison, NOT stellar eigenvalue validation.",
              "n":len(result), "max_abs_percent":float(np.max(absolute)),
              "median_abs_percent":float(np.median(absolute)), "rows":result}
    (root / "results_tau_consistency.json").write_text(json.dumps(output,indent=2))
    assert len(result)==6
    # A broad sanity threshold, not a proof of solver correctness.
    if not all(np.isfinite(absolute)) or max(absolute) > 25:
        raise AssertionError("Damping-time empirical discrepancy exceeds sanity threshold")
    print(json.dumps(output,indent=2))


if __name__ == "__main__":
    main()

"""Independent-nuclear-EoS stress test for the f-mode DM shift.

These are six externally published *piecewise-polytropic representations* of
microphysical nuclear EoSs, not a fresh ensemble of microscopic QMC solutions.
The perturbative solver and the DM transformation are the ones used elsewhere
in this repository: this test explores sensitivity to the nuclear baseline,
not independence of the oscillation implementation.

Run: cd population && python microphysics_anchor.py [jobs]
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from families import read_pp, dm_admixed, max_mass, pc_for_mass, solve_star
from fmode import Background
import fmode2 as F
from test_tov import REF

MASS = 1.4
MCHI = 938.0
G_VALUES = (100.0, 300.0)
OUTPUT = Path(__file__).resolve().parent / "results_microphysics_anchor.json"


def calculate(name, g, reference):
    logP1, gam1, gam2, gam3, _, _, freq_ref = REF[name]
    nuclear = read_pp(logP1, gam1, gam2, gam3)
    admixed = dm_admixed(nuclear, g, MCHI)
    mx, lpmax = max_mass(admixed)
    lp = pc_for_mass(admixed, MASS, lpmax)
    pc = np.exp(lp)
    s = solve_star(admixed, pc)
    bg = Background(admixed, pc)
    last_error = None
    for factor in (1.005, 1.01, 0.998, 1.02, 0.985):
        try:
            f, tau, omega = F.find_mode(bg, reference["f"] * factor, 0.2)
            if not (0.7 * reference["f"] < f < 1.4 * reference["f"] and 0.01 < tau < 10):
                raise ValueError("mode outside astrophysical f-mode search window")
            return dict(eos=name, G_fm2=g, mchi_MeV=MCHI,
                        M_Msun=MASS, Mmax_Msun=float(mx), supports_1p97=bool(mx >= 1.97),
                        R_km=float(s["R"]), Lambda=float(s["Lambda"]),
                        f_Hz=float(f), tau_s=float(tau),
                        df_Hz=float(f - reference["f"]),
                        df_percent=float(100 * (f / reference["f"] - 1)),
                        dR_percent=float(100 * (s["R"] / reference["R"] - 1)),
                        dLambda_percent=float(100 * (s["Lambda"] / reference["Lambda"] - 1)))
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            last_error = str(exc)
    return dict(eos=name, G_fm2=g, mchi_MeV=MCHI, error=last_error or "mode not found")


def main():
    ref = json.loads(Path(__file__).with_name("results_validation.json").read_text())
    base = {r["eos"]: dict(R=r["R"], Lambda=r["Lam"], f=r["f"], tau=r["tau"])
            for r in ref["six"]}
    missing = set(REF) - set(base)
    if missing:
        raise RuntimeError(f"missing baseline f-mode frequencies: {missing}")
    work = [(name, g, base[name]) for name in sorted(REF) for g in G_VALUES]
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    rows = Parallel(n_jobs=jobs, verbose=5)(delayed(calculate)(*task) for task in work)
    failed = [r for r in rows if "error" in r]
    if failed:
        OUTPUT.write_text(json.dumps(dict(base=base, runs=rows, failures=failed), indent=2))
        raise RuntimeError(f"{len(failed)} stellar modes failed: {failed}")
    shift_by_g = {}
    for g in G_VALUES:
        data = [r for r in rows if r["G_fm2"] == g]
        vals = np.asarray([r["df_percent"] for r in data])
        shift_by_g[str(int(g))] = {
            "n": len(vals),
            "median_percent": float(np.median(vals)),
            "range_percent": [float(np.min(vals)), float(np.max(vals))],
            "viable_Mmax_count": int(sum(r["supports_1p97"] for r in data)),
        }
    freqs = np.asarray([r["f"] for r in base.values()])
    summary = dict(nuclear_baselines=len(base), number_of_admixed_modes=len(rows),
                   f_nuclear_range_Hz=[float(freqs.min()), float(freqs.max())],
                   f_nuclear_spread_Hz=float(freqs.max()-freqs.min()),
                   shift_by_G=shift_by_g)
    OUTPUT.write_text(json.dumps(dict(schema="microphysics-anchor-v1", description=__doc__,
                                      base=base, runs=rows, summary=summary), indent=2))
    assert len(rows) == 12
    print(json.dumps(summary, indent=2))
    print("OK: independently parameterised nuclear-baseline test (not two-fluid oscillation validation)")


if __name__ == "__main__":
    main()

"""Regenerate the paper's causal-only primary comparison, table, figure and macros.

This uses the fully audited 340-EoS subset (85 per class) with the existing
whole-EoS 3 x 5 validation output. The broader 360-EoS analysis remains
explicitly secondary because nine EoSs failed full-interior causality.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "population" / "results_causal_filtered_classification.json"
TABLE = ROOT / "tables" / "tab_causal_primary.tex"
FIGURE = ROOT / "figs" / "fig_causal_primary.pdf"
MACROS = ROOT / "results" / "macros_causal.json"
CLASSES = ["Nucleonic", "Strange", "DM-admixed", "Hyperonic-like"]
ORDER = (("ALL@0", "Noise-free"), ("NICER+f20", "NICER-like + 20% f"))


def main():
    raw = json.loads(SOURCE.read_text())
    selection = raw["selection"]
    assert selection["total"] == 340
    assert selection["used_per_class"] == 85
    assert selection["available_by_class"] == {"0": 85, "1": 90, "2": 86, "3": 90}
    assert len(raw["runs"]) == 30, "expected three repetitions x five folds x two scenarios"
    cells = []
    macros = {
        "causalTotal": str(selection["total"]),
        "causalPerClass": str(selection["used_per_class"]),
        "causalRemoved": "9",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.6), sharex=True, sharey=True)
    for axis, (scenario, label) in zip(axes, ORDER):
        for model, name in (("MLP", "MLP ensemble"), ("LogReg", "Logistic regression")):
            stats = raw["scenarios"][scenario][model]
            reps = np.asarray(stats["repetition_means"], float)
            assert len(reps) == 3
            mean = float(stats["accuracy_mean"])
            assert np.isclose(mean, reps.mean(), rtol=0, atol=1e-10)
            cm = np.asarray(stats["pooled_confusion_matrix"], float)
            assert cm.shape == (4, 4) and np.all(cm.sum(axis=1) > 0)
            recall = np.diag(cm) / cm.sum(axis=1)
            pair = float(stats["pair_accuracy_conditional"])
            assert 0 <= pair <= 1
            scenkey = "zero" if scenario == "ALL@0" else "noisy"
            modelkey = "mlp" if model == "MLP" else "lr"
            macros[f"causal_{scenkey}_{modelkey}"] = f"{100*mean:.1f}"
            macros[f"causal_{scenkey}_{modelkey}_pair"] = f"{100*pair:.1f}"
            macros[f"causal_{scenkey}_{modelkey}_strange"] = f"{100*recall[1]:.1f}"
            # Cross-validation repetitions overlap in underlying EoSs, so do
            # not turn their standard deviation into an independent-sample CI.
            cells.append(
                f"{label} & {name} & {100*mean:.1f} & "
                f"{100*recall[1]:.1f} & {100*pair:.1f} "+r"\\"
            )
            if model != "MLP":
                continue
            normalized = cm / cm.sum(axis=1, keepdims=True)
            axis.imshow(normalized, vmin=0, vmax=1, cmap="viridis")
            for i in range(4):
                for j in range(4):
                    axis.text(j, i, f"{100*normalized[i,j]:.0f}",
                              ha="center", va="center",
                              color="white" if normalized[i,j] < .60 else "black", fontsize=8)
            axis.set_title(f"{label} ({100*mean:.1f}% overall)")
            axis.set_xticks(range(4), ["Nuc.", "Str.", "DM", "Hyp."])
            axis.set_yticks(range(4), CLASSES, fontsize=8)
            axis.set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    fig.suptitle("Fully causal 340-EoS primary evaluation: row-normalized confusion (%)", fontsize=10)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    MACROS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    plt.close(fig)
    # The surrounding TeX table supplies the row terminator for its final row.
    TABLE.write_text("\n".join(cells).rstrip()[:-2], encoding="utf-8")
    MACROS.write_text(json.dumps(macros, indent=2)+"\n", encoding="utf-8")
    print("CAUSAL_PRIMARY_VALIDATED", json.dumps(macros, sort_keys=True))
    assert abs(float(macros["causal_zero_mlp"]) - 63.1) < .11
    assert abs(float(macros["causal_noisy_mlp"]) - 36.7) < .11


if __name__ == "__main__":
    main()

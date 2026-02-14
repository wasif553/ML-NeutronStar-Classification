#!/usr/bin/env python3
"""
Reproduce ALL ML figures and outputs for EOS classification (scikit-learn MLP).

Input:
  - DataML_shuffled.xlsx  (must be in the same folder as this script, or edit the path below)

Label column:
  - 'EoS' with mapping:
      0 = nucleon only
      1 = strange matter
      2 = neutron decay -> non-interacting dark matter
      3 = hyperons

Features used (all non-label columns):
  - '# Mass_Msun'
  - 'Radius_km'
  - 'f_mode_Hz'
  - 'Quadrupole_...Surface_Redshift'  (exact name read from file)
  - 'Damping_time_s'
  - 'Characteristic_Strain_hc'

Outputs (saved in current folder):
  - DataML_ready_for_training.csv
  - fig_mlp_training_loss.png
  - fig_confusion_matrix.png
  - fig_confusion_matrix_normalized.png
  - fig_feature_importance.png
  - fig_accuracy_vs_mass_bins.png
  - accuracy_vs_mass_bins_table.csv
  - fig_multiclass_roc.png

Notes:
  - TensorFlow was not available in the execution environment, so we use scikit-learn's MLPClassifier.
  - Splits are stratified and fixed with random_state=42 for reproducibility.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_curve,
    auc
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt


def main():
    # -----------------------------
    # 1) Load data
    # -----------------------------
    xlsx_path = "C:/Users/maste/Documents/Python learning/Ml/DataML_shuffled.xlsx"  # change if needed
    df = pd.read_excel(xlsx_path)

    label_col = "EoS"
    feature_cols = [c for c in df.columns if c != label_col]

    # Basic cleaning: drop rows with missing values in relevant columns
    df_clean = df.dropna(subset=feature_cols + [label_col]).copy()

    X = df_clean[feature_cols].values.astype(float)
    y = df_clean[label_col].values.astype(int)

    print("Clean dataset shape:", df_clean.shape)
    print("Features:", feature_cols)
    print("Class distribution:\n", pd.Series(y).value_counts().sort_index())

    # Save ML-ready CSV for transparency/reuse
    df_clean.to_csv("DataML_ready_for_training.csv", index=False)
    print("Saved: DataML_ready_for_training.csv")

    # -----------------------------
    # 2) Train/Val/Test split (stratified)
    # -----------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval
    )

    print("\nSplit sizes: Train", len(y_train), "Val", len(y_val), "Test", len(y_test))

    # -----------------------------
    # 3) Standardize
    # -----------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # -----------------------------
    # 4) Class weights (computed; may not be used depending on sklearn version)
    # -----------------------------
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("\nClass weights:", class_weight)

    sample_weight = np.array([class_weight[int(label)] for label in y_train], dtype=float)

    # -----------------------------
    # 5) Train MLP
    # -----------------------------
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,  # internal validation for early stopping
        random_state=42
    )

    # Try fit with sample_weight if supported by your sklearn version
    try:
        mlp.fit(X_train_s, y_train, sample_weight=sample_weight)
        used_weights = True
    except TypeError:
        mlp.fit(X_train_s, y_train)
        used_weights = False

    print("\nTrained MLP. Used sample weights:", used_weights)
    print("Iterations:", mlp.n_iter_)

    # -----------------------------
    # 6) Evaluate
    # -----------------------------
    y_train_pred = mlp.predict(X_train_s)
    y_val_pred   = mlp.predict(X_val_s)
    y_test_pred  = mlp.predict(X_test_s)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc   = accuracy_score(y_val, y_val_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)

    print(f"\nAccuracy  Train: {train_acc:.4f}  Val: {val_acc:.4f}  Test: {test_acc:.4f}")
    print("\nClassification report (test):\n")
    print(classification_report(y_test, y_test_pred, digits=3))

    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_test, y_test_pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred) [0,1,2,3]:\n", cm)

    # -----------------------------
    # 7A) Training loss curve
    # -----------------------------
    plt.figure()
    plt.plot(mlp.loss_curve_, label="training loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_mlp_training_loss.png", dpi=200)
    plt.close()
    print("Saved: fig_mlp_training_loss.png")

    # -----------------------------
    # 7B) Confusion matrix (counts)
    # -----------------------------
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ["0 (nuc)", "1 (str)", "2 (DM)", "3 (hyp)"], rotation=30, ha="right")
    plt.yticks(tick_marks, ["0 (nuc)", "1 (str)", "2 (DM)", "3 (hyp)"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("fig_confusion_matrix.png", dpi=200)
    plt.close()
    print("Saved: fig_confusion_matrix.png")

    # -----------------------------
    # 7C) Normalized confusion matrix (row-normalized)
    # -----------------------------
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure()
    plt.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=1)
    plt.title("Normalized Confusion Matrix (Test Set)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar(label="Fraction")
    plt.xticks(tick_marks, ["0 (nuc)", "1 (str)", "2 (DM)", "3 (hyp)"], rotation=30, ha="right")
    plt.yticks(tick_marks, ["0 (nuc)", "1 (str)", "2 (DM)", "3 (hyp)"])
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig("fig_confusion_matrix_normalized.png", dpi=200)
    plt.close()
    print("Saved: fig_confusion_matrix_normalized.png")

    # -----------------------------
    # 7D) Permutation feature importance (manual)
    # -----------------------------
    baseline_acc = test_acc
    rng = np.random.RandomState(42)
    drops = []
    for k in range(X_test_s.shape[1]):
        Xp = X_test_s.copy()
        rng.shuffle(Xp[:, k])
        acc_k = accuracy_score(y_test, mlp.predict(Xp))
        drops.append(baseline_acc - acc_k)
    drops = np.array(drops)

    plt.figure()
    plt.bar(range(len(feature_cols)), drops)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=30, ha="right")
    plt.ylabel("Accuracy drop")
    plt.title("Permutation Feature Importance (Test Set)")
    plt.tight_layout()
    plt.savefig("fig_feature_importance.png", dpi=200)
    plt.close()
    print("Saved: fig_feature_importance.png")

    # -----------------------------
    # 7E) Accuracy vs mass bins (test set)
    # -----------------------------
    # Identify the mass column
    mass_col = "# Mass_Msun"
    if mass_col not in feature_cols:
        candidates = [c for c in feature_cols if "Mass" in c or "mass" in c]
        mass_col = candidates[0] if candidates else feature_cols[0]

    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    masses = X_test_df[mass_col].values

    nbins = 8
    bins = np.linspace(masses.min(), masses.max(), nbins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    acc_bins = []
    counts = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        idx = (masses >= b0) & (masses < b1)
        counts.append(int(idx.sum()))
        if idx.sum() >= 5:
            acc_bins.append(accuracy_score(y_test[idx], y_test_pred[idx]))
        else:
            acc_bins.append(np.nan)

    plt.figure()
    plt.plot(bin_centers, acc_bins, marker="o")
    plt.xlabel("Mass bin center (M_sun)")
    plt.ylabel("Accuracy (test set)")
    plt.title("Classification Accuracy vs Mass (Binned)")
    plt.tight_layout()
    plt.savefig("fig_accuracy_vs_mass_bins.png", dpi=200)
    plt.close()
    print("Saved: fig_accuracy_vs_mass_bins.png")

    bin_table = pd.DataFrame({
        "bin_left": bins[:-1],
        "bin_right": bins[1:],
        "bin_center": bin_centers,
        "count": counts,
        "accuracy": acc_bins
    })
    bin_table.to_csv("accuracy_vs_mass_bins_table.csv", index=False)
    print("Saved: accuracy_vs_mass_bins_table.csv")

    # -----------------------------
    # 7F) Multiclass ROC curves (one-vs-rest)
    # -----------------------------
    y_score = mlp.predict_proba(X_test_s)
    y_test_bin = label_binarize(y_test, classes=labels)

    fpr = {}
    tpr = {}
    roc_auc = {}
    for i, cls in enumerate(labels):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    for i, cls in enumerate(labels):
        plt.plot(fpr[i], tpr[i], label=f"Class {cls} (AUC={roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC={roc_auc['micro']:.3f})", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves (One-vs-Rest, Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("fig_multiclass_roc.png", dpi=200)
    plt.close()
    print("Saved: fig_multiclass_roc.png")

    print("\nAll outputs saved in the current working directory.")


if __name__ == "__main__":
    main()

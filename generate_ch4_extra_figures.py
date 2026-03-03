"""
===============================================================================
CHAPTER 4 EXTRA VISUALISATIONS GENERATOR (Random Forest ONLY)
===============================================================================
Purpose:
    Generates additional evaluation visualisations required by typical AI
    coursework rubrics, while using ONLY the trained Random Forest model.

Generates (saved into report_outputs/figures/):
    Figure 10: Learning Curve (Training vs CV Accuracy)   [RF equivalent of "accuracy/loss curve"]
    Figure 11: Multi-class ROC Curves (One-vs-Rest)
    Figure 12: Multi-class Precision–Recall Curves (One-vs-Rest)
    Figure 13: Decision Boundary (2D slice: stress_level vs sleep_hours)
    Figure 14: Per-class Precision/Recall/F1 bar chart (test set)

Also generates (optional):
    Figure 10b: OOB Error vs Number of Trees (RF-specific training stability curve)

How to run:
    python3 generate_ch4_extra_figures.py

Requirements:
    - Run python3 train_pipeline.py first (model + scaler + encoder saved in models/)
===============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.ensemble import RandomForestClassifier


DATASET_PATH = "data/student_burnout_dataset.csv"
MODEL_PATH = "models/random_forest_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
FEATURES_PATH = "models/feature_names.pkl"

OUT_DIR = "report_outputs/figures"


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run: python3 train_pipeline.py")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    return model, scaler, label_encoder, feature_names


def load_and_prepare_data(label_encoder, feature_names):
    """
    Load raw CSV, impute missing values (median), encode target using SAVED encoder,
    then split and scale using SAVED scaler later.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Dataset not found. Run: python3 train_pipeline.py (it generates the CSV).")

    df = pd.read_csv(DATASET_PATH)

    # Median imputation for numeric columns (same approach as training pipeline)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Encode target with the SAVED label encoder to guarantee identical mapping
    y = label_encoder.transform(df["burnout_risk"])
    X = df[feature_names].copy()

    # Use the same split settings as your pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def fig10_learning_curve(model, X_train_scaled, y_train, save_path):
    """
    RF does not have epoch-by-epoch loss curves. A learning curve is the correct
    equivalent: it shows training vs cross-validation accuracy as training size grows.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 8)

    # Use a fresh estimator with same hyperparameters (still Random Forest only).
    params = model.get_params()
    estimator = RandomForestClassifier(**params)

    sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X_train_scaled,
        y=y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, train_mean, marker="o", linewidth=2.2, label="Training Accuracy")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)

    ax.plot(sizes, val_mean, marker="o", linewidth=2.2, label="Cross-Validation Accuracy")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)

    ax.set_title("Figure 10: Learning Curve (Random Forest)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Set Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig10b_oob_curve(model, X_train_scaled, y_train, save_path):
    """
    Optional RF-specific curve: OOB error vs number of trees.
    Still Random Forest only (a hyperparameter sweep on n_estimators).
    """
    params = model.get_params()
    # Ensure OOB enabled
    params["oob_score"] = True
    params["bootstrap"] = True
    params["n_jobs"] = -1
    params["random_state"] = 42

    tree_counts = [10, 25, 50, 75, 100, 150, 200, 300]
    oob_errors = []

    for n in tree_counts:
        params["n_estimators"] = n
        rf = RandomForestClassifier(**params)
        rf.fit(X_train_scaled, y_train)
        oob_error = 1 - rf.oob_score_
        oob_errors.append(oob_error)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tree_counts, oob_errors, marker="o", linewidth=2.2, color="#c0392b")
    ax.set_title("Figure 10b (Optional): OOB Error vs Number of Trees (Random Forest)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Trees (n_estimators)", fontsize=12, fontweight="bold")
    ax.set_ylabel("OOB Error (1 - OOB Accuracy)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig11_multiclass_roc(model, X_test_scaled, y_test, label_encoder, save_path):
    """
    Multi-class ROC using One-vs-Rest (OvR).
    """
    proba = model.predict_proba(X_test_scaled)
    classes = model.classes_  # numeric labels used in training
    class_names = label_encoder.inverse_transform(classes)

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve((y_test == cls).astype(int), proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    # chance line
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Chance")

    macro_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")

    ax.set_title(f"Figure 11: Multi-Class ROC Curves (OvR) — Macro AUC={macro_auc:.3f}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig12_multiclass_pr(model, X_test_scaled, y_test, label_encoder, save_path):
    """
    Multi-class Precision–Recall curves using One-vs-Rest (OvR).
    """
    proba = model.predict_proba(X_test_scaled)
    classes = model.classes_
    class_names = label_encoder.inverse_transform(classes)

    fig, ax = plt.subplots(figsize=(10, 7))

    ap_scores = []
    for i, cls in enumerate(classes):
        y_true_bin = (y_test == cls).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_bin, proba[:, i])
        ap = average_precision_score(y_true_bin, proba[:, i])
        ap_scores.append(ap)
        ax.plot(recall, precision, linewidth=2, label=f"{class_names[i]} (AP={ap:.3f})")

    macro_ap = float(np.mean(ap_scores))

    ax.set_title(f"Figure 12: Multi-Class Precision–Recall Curves (OvR) — Macro AP={macro_ap:.3f}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig13_decision_boundary_2d(
    model, scaler, label_encoder, feature_names,
    X_train, X_test, y_test,
    x_feature="stress_level",
    y_feature="sleep_hours",
    save_path="report_outputs/figures/fig13_decision_boundary.png"
):
    """
    Decision boundary for a 2D slice: vary two features, hold others at training medians.
    Uses the SAME saved scaler + SAME trained RF model.
    """
    if x_feature not in feature_names or y_feature not in feature_names:
        raise ValueError("Chosen decision-boundary features must exist in feature_names.")

    # Medians from training data (original scale)
    medians = X_train.median(numeric_only=True).to_dict()

    # Set axis ranges based on realistic domain constraints
    ranges = {
        "stress_level": (1, 10),
        "sleep_hours": (3, 10),
        "attendance": (10, 100),
        "study_hours": (0, 14),
        "assignment_completion": (0, 100),
        "gpa": (0, 4),
        "screen_time": (1, 16),
        "part_time_hours": (0, 30),
        "social_hours": (0, 8),
    }

    x_min, x_max = ranges.get(x_feature, (X_train[x_feature].min(), X_train[x_feature].max()))
    y_min, y_max = ranges.get(y_feature, (X_train[y_feature].min(), X_train[y_feature].max()))

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid_df = pd.DataFrame({f: np.full(xx.size, medians[f]) for f in feature_names})
    grid_df[x_feature] = xx.ravel()
    grid_df[y_feature] = yy.ravel()

    grid_scaled = scaler.transform(grid_df)
    grid_pred = model.predict(grid_scaled).reshape(xx.shape)

    # Prepare plot colors
    classes = model.classes_
    class_names = label_encoder.inverse_transform(classes)
    palette = {name: color for name, color in zip(class_names, ["#e74c3c", "#2ecc71", "#f39c12"])}

    # Map numeric predictions to indices 0..K-1 for contour plot
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    Z = np.vectorize(class_to_idx.get)(grid_pred)

    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = matplotlib.colors.ListedColormap([palette[class_names[i]] for i in range(len(class_names))])

    ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap, levels=len(classes))

    # Scatter test samples projected into 2D
    for cls, name in zip(classes, class_names):
        mask = (y_test == cls)
        ax.scatter(
            X_test.loc[mask, x_feature],
            X_test.loc[mask, y_feature],
            s=35,
            alpha=0.8,
            label=f"{name} (test points)",
            edgecolor="white",
            linewidth=0.6
        )

    ax.set_title(f"Figure 13: Decision Boundary (2D Slice) — {x_feature} vs {y_feature}",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel(x_feature, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_feature, fontsize=12, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig14_per_class_metrics_bar(y_test, y_pred, label_encoder, save_path):
    """
    Bar chart comparing per-class Precision/Recall/F1.
    This satisfies the rubric's "bar chart comparing results" without comparing models.
    """
    classes = np.unique(y_test)
    class_names = label_encoder.inverse_transform(classes)

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    metrics = ["precision", "recall", "f1-score"]
    values = {m: [report[name][m] for name in class_names] for m in metrics}

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, values["precision"], width, label="Precision")
    ax.bar(x, values["recall"], width, label="Recall")
    ax.bar(x + width, values["f1-score"], width, label="F1-score")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Figure 14: Per-Class Metric Comparison (Test Set)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    # Add numeric labels
    for i, name in enumerate(class_names):
        ax.text(x[i] - width, values["precision"][i] + 0.02, f"{values['precision'][i]:.2f}", ha="center", fontsize=10)
        ax.text(x[i],         values["recall"][i] + 0.02,    f"{values['recall'][i]:.2f}", ha="center", fontsize=10)
        ax.text(x[i] + width, values["f1-score"][i] + 0.02,  f"{values['f1-score'][i]:.2f}", ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model, scaler, label_encoder, feature_names = load_artifacts()
    X_train, X_test, y_train, y_test = load_and_prepare_data(label_encoder, feature_names)

    # Scale using saved scaler (must match training)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Predictions for some plots
    y_pred = model.predict(X_test_scaled)

    # --- Generate required Chapter 4 visuals ---
    fig10_learning_curve(
        model, X_train_scaled, y_train,
        save_path=os.path.join(OUT_DIR, "fig10_learning_curve.png")
    )

    # Optional but useful appendix figure (RF-specific)
    fig10b_oob_curve(
        model, X_train_scaled, y_train,
        save_path=os.path.join(OUT_DIR, "fig10b_oob_error_vs_trees.png")
    )

    fig11_multiclass_roc(
        model, X_test_scaled, y_test, label_encoder,
        save_path=os.path.join(OUT_DIR, "fig11_multiclass_roc_ovr.png")
    )

    fig12_multiclass_pr(
        model, X_test_scaled, y_test, label_encoder,
        save_path=os.path.join(OUT_DIR, "fig12_multiclass_pr_ovr.png")
    )

    fig13_decision_boundary_2d(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        feature_names=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        x_feature="stress_level",
        y_feature="sleep_hours",
        save_path=os.path.join(OUT_DIR, "fig13_decision_boundary_stress_vs_sleep.png")
    )

    fig14_per_class_metrics_bar(
        y_test=y_test,
        y_pred=y_pred,
        label_encoder=label_encoder,
        save_path=os.path.join(OUT_DIR, "fig14_per_class_metrics_bar.png")
    )

    print("[INFO] Chapter 4 extra figures generated:")
    for f in [
        "fig10_learning_curve.png",
        "fig10b_oob_error_vs_trees.png (optional)",
        "fig11_multiclass_roc_ovr.png",
        "fig12_multiclass_pr_ovr.png",
        "fig13_decision_boundary_stress_vs_sleep.png",
        "fig14_per_class_metrics_bar.png",
    ]:
        print(f"  - {OUT_DIR}/{f}")


if __name__ == "__main__":
    main()
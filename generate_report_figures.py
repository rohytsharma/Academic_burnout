"""
===============================================================================
REPORT FIGURES AND TABLES GENERATOR
===============================================================================
Purpose:
    Generates ALL figures and tables needed for the coursework report.
    Saves everything as PNG images and CSV files in a 'report_outputs/' folder.

    Run AFTER train_pipeline.py:
        python generate_report_figures.py

Output Folder Structure:
    report_outputs/
    ├── figures/
    │   ├── fig1_class_distribution.png
    │   ├── fig2_correlation_heatmap.png
    │   ├── fig3_confusion_matrix.png
    │   ├── fig4_feature_importance.png
    │   ├── fig5_cv_scores_boxplot.png
    │   ├── fig6_feature_distributions.png
    │   ├── fig7_feature_boxplots_by_class.png
    │   ├── fig8_prediction_probability_example.png
    │   └── fig9_bias_variance_diagram.png
    └── tables/
        ├── table1_dataset_statistics.csv
        ├── table2_missing_values.csv
        ├── table3_class_distribution.csv
        ├── table4_evaluation_metrics.csv
        ├── table5_classification_report.csv
        ├── table6_cross_validation_scores.csv
        ├── table7_feature_importance.csv
        ├── table8_confusion_matrix.csv
        └── table9_hyperparameters.csv

Author: Student
Course: Undergraduate AI Coursework
===============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_preprocessing import DataPreprocessor
from modules.model_training import BurnoutModelTrainer


def create_output_dirs():
    """Create output directories for report figures and tables."""
    os.makedirs("report_outputs/figures", exist_ok=True)
    os.makedirs("report_outputs/tables", exist_ok=True)
    print("[INFO] Output directories created: report_outputs/figures/ and report_outputs/tables/")


def load_all_data():
    """Load dataset, preprocess, and load trained model."""
    # Load raw dataset
    raw_df = pd.read_csv("data/student_burnout_dataset.csv")

    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline("data/student_burnout_dataset.csv")

    # Load trained model
    trainer = BurnoutModelTrainer()
    trainer.load_model("models/random_forest_model.pkl")

    # Get predictions
    y_pred = trainer.predict(data["X_test_scaled"])
    y_pred_proba = trainer.predict_proba(data["X_test_scaled"])

    return raw_df, data, preprocessor, trainer, y_pred, y_pred_proba


# =============================================================================
# FIGURE 1: CLASS DISTRIBUTION
# =============================================================================
def generate_fig1_class_distribution(raw_df):
    """Bar chart showing the distribution of burnout risk classes."""
    print("\n[FIGURE 1] Generating class distribution chart...")

    class_counts = raw_df["burnout_risk"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    bar_colors = [colors.get(c, "blue") for c in class_counts.index]

    bars = axes[0].bar(class_counts.index, class_counts.values,
                       color=bar_colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     str(val), ha="center", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Burnout Risk Level", fontsize=12)
    axes[0].set_ylabel("Number of Students", fontsize=12)
    axes[0].set_title("(a) Class Distribution - Bar Chart", fontsize=13, fontweight="bold")

    # Pie chart
    axes[1].pie(class_counts.values, labels=class_counts.index,
                colors=bar_colors, autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 11, "fontweight": "bold"},
                wedgeprops={"edgecolor": "black", "linewidth": 0.8})
    axes[1].set_title("(b) Class Distribution - Pie Chart", fontsize=13, fontweight="bold")

    plt.suptitle("Figure 1: Distribution of Burnout Risk Classes in Dataset",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig1_class_distribution.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig1_class_distribution.png")


# =============================================================================
# FIGURE 2: CORRELATION HEATMAP
# =============================================================================
def generate_fig2_correlation_heatmap(raw_df):
    """Correlation heatmap of all numeric features."""
    print("\n[FIGURE 2] Generating correlation heatmap...")

    numeric_df = raw_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.triu(np.ones_like(corr, dtype=bool))  # Upper triangle mask
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1,
        square=True, cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    ax.set_title("Figure 2: Feature Correlation Heatmap",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig2_correlation_heatmap.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig2_correlation_heatmap.png")


# =============================================================================
# FIGURE 3: CONFUSION MATRIX
# =============================================================================
def generate_fig3_confusion_matrix(data, y_pred):
    """Confusion matrix heatmap with counts and percentages."""
    print("\n[FIGURE 3] Generating confusion matrix...")

    class_names = list(data["label_encoder"].classes_)
    cm = confusion_matrix(data["y_test"], y_pred)
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Count"})
    axes[0].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Actual Label", fontsize=11, fontweight="bold")
    axes[0].set_title("(a) Confusion Matrix - Counts", fontsize=12, fontweight="bold")

    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, linecolor="gray",
                cbar_kws={"label": "Percentage (%)"})
    axes[1].set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Actual Label", fontsize=11, fontweight="bold")
    axes[1].set_title("(b) Confusion Matrix - Percentages", fontsize=12, fontweight="bold")

    plt.suptitle("Figure 3: Confusion Matrix — Random Forest Classifier",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig3_confusion_matrix.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig3_confusion_matrix.png")


# =============================================================================
# FIGURE 4: FEATURE IMPORTANCE
# =============================================================================
def generate_fig4_feature_importance(trainer, data):
    """Horizontal bar chart of feature importances."""
    print("\n[FIGURE 4] Generating feature importance chart...")

    importances = trainer.model.feature_importances_
    feature_names = data["feature_names"]
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    names = [p[0] for p in pairs]
    scores = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1],
                   edgecolor="black", linewidth=0.5)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Importance Score (Mean Decrease in Impurity)", fontsize=12)
    ax.set_title("Figure 4: Feature Importance — Random Forest Classifier",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(scores) * 1.25)
    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig4_feature_importance.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig4_feature_importance.png")

    return pairs


# =============================================================================
# FIGURE 5: CROSS-VALIDATION BOX PLOT
# =============================================================================
def generate_fig5_cv_boxplot(data, trainer):
    """Box plot showing cross-validation score distributions."""
    print("\n[FIGURE 5] Generating cross-validation box plot...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring_metrics = {
        "Accuracy": "accuracy",
        "Precision\n(Macro)": "precision_macro",
        "Recall\n(Macro)": "recall_macro",
        "F1-Score\n(Macro)": "f1_macro"
    }

    all_scores = {}
    cv_results_for_table = {}

    for display_name, scoring in scoring_metrics.items():
        scores = cross_val_score(
            trainer.model, data["X_train_scaled"], data["y_train"],
            cv=skf, scoring=scoring, n_jobs=-1
        )
        all_scores[display_name] = scores
        clean_name = display_name.replace("\n", " ")
        cv_results_for_table[clean_name] = {
            "Fold 1": scores[0], "Fold 2": scores[1], "Fold 3": scores[2],
            "Fold 4": scores[3], "Fold 5": scores[4],
            "Mean": scores.mean(), "Std": scores.std()
        }

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(all_scores.values(), labels=all_scores.keys(),
                    patch_artist=True, widths=0.5,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color="red", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    for i, (name, scores) in enumerate(all_scores.items()):
        ax.scatter(i + 1, scores.mean(), color="black", zorder=5, s=80,
                   marker="D", label="Mean" if i == 0 else "")
        ax.annotate(f"{scores.mean():.3f}", (i + 1, scores.mean()),
                    textcoords="offset points", xytext=(15, 5),
                    fontsize=10, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Figure 5: 5-Fold Stratified Cross-Validation Scores",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig5_cv_scores_boxplot.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig5_cv_scores_boxplot.png")

    return cv_results_for_table


# =============================================================================
# FIGURE 6: FEATURE DISTRIBUTIONS (HISTOGRAMS)
# =============================================================================
def generate_fig6_feature_distributions(raw_df):
    """Histograms of all numeric features, colored by burnout risk."""
    print("\n[FIGURE 6] Generating feature distribution histograms...")

    feature_cols = [c for c in raw_df.columns if c != "burnout_risk"]
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

    for i, col in enumerate(feature_cols):
        for risk_level in ["Low", "Medium", "High"]:
            subset = raw_df[raw_df["burnout_risk"] == risk_level][col].dropna()
            axes[i].hist(subset, bins=20, alpha=0.5, label=risk_level,
                         color=colors[risk_level], edgecolor="white")
        axes[i].set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel("Frequency", fontsize=9)
        axes[i].legend(fontsize=8)

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Figure 6: Feature Distributions by Burnout Risk Level",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig6_feature_distributions.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig6_feature_distributions.png")


# =============================================================================
# FIGURE 7: BOX PLOTS BY CLASS
# =============================================================================
def generate_fig7_feature_boxplots(raw_df):
    """Box plots of each feature grouped by burnout risk class."""
    print("\n[FIGURE 7] Generating feature box plots by class...")

    feature_cols = [c for c in raw_df.columns if c != "burnout_risk"]
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    order = ["Low", "Medium", "High"]
    palette = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}

    for i, col in enumerate(feature_cols):
        sns.boxplot(x="burnout_risk", y=col, data=raw_df, ax=axes[i],
                    order=order, palette=palette, width=0.5,
                    linewidth=1.2, fliersize=3)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=11, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel(col, fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Figure 7: Feature Distributions by Burnout Risk Level (Box Plots)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig7_feature_boxplots_by_class.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig7_feature_boxplots_by_class.png")


# =============================================================================
# FIGURE 8: EXAMPLE PREDICTION PROBABILITY
# =============================================================================
def generate_fig8_prediction_example(trainer, preprocessor):
    """Show a sample prediction with probability bars."""
    print("\n[FIGURE 8] Generating example prediction probability chart...")

    sample_input = {
        "attendance": 55.0, "study_hours": 2.0, "sleep_hours": 4.5,
        "assignment_completion": 40.0, "gpa": 1.8, "stress_level": 8,
        "screen_time": 10.0, "part_time_hours": 20.0, "social_hours": 6.0
    }

    input_scaled = preprocessor.preprocess_single_input(sample_input)
    prediction = trainer.predict(input_scaled)[0]
    probabilities = trainer.predict_proba(input_scaled)[0]
    predicted_label = preprocessor.label_encoder.inverse_transform([prediction])[0]
    class_names = preprocessor.label_encoder.classes_

    fig, ax = plt.subplots(figsize=(8, 4))

    color_map = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
    bar_colors = [color_map.get(c, "blue") for c in class_names]

    bars = ax.barh(class_names, probabilities * 100, color=bar_colors,
                   edgecolor="black", height=0.5, linewidth=0.8)

    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.1f}%", va="center", fontweight="bold", fontsize=12)

    ax.set_xlabel("Probability (%)", fontsize=12)
    ax.set_title(f"Figure 8: Example Prediction — Predicted: {predicted_label}",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 110)

    # Add sample input as text annotation
    input_text = "Input: " + ", ".join([f"{k}={v}" for k, v in sample_input.items()])
    ax.annotate(input_text, xy=(0.5, -0.15), xycoords="axes fraction",
                ha="center", fontsize=8, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="gray"))

    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig8_prediction_probability_example.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig8_prediction_probability_example.png")


# =============================================================================
# FIGURE 9: BIAS-VARIANCE TRADEOFF DIAGRAM
# =============================================================================
def generate_fig9_bias_variance_diagram():
    """Conceptual diagram illustrating bias-variance tradeoff."""
    print("\n[FIGURE 9] Generating bias-variance tradeoff diagram...")

    complexity = np.linspace(0.5, 10, 100)

    # Simulated curves
    bias_squared = 5 * np.exp(-0.5 * complexity) + 0.2
    variance = 0.05 * complexity ** 2
    total_error = bias_squared + variance + 0.3  # irreducible error

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(complexity, bias_squared, "b-", linewidth=2.5, label="Bias²")
    ax.plot(complexity, variance, "r-", linewidth=2.5, label="Variance")
    ax.plot(complexity, total_error, "k--", linewidth=2.5, label="Total Error")
    ax.axhline(y=0.3, color="gray", linestyle=":", linewidth=1.5,
               label="Irreducible Error")

    # Mark optimal point
    optimal_idx = np.argmin(total_error)
    ax.axvline(x=complexity[optimal_idx], color="green", linestyle="--",
               linewidth=1.5, alpha=0.7)
    ax.scatter(complexity[optimal_idx], total_error[optimal_idx],
               color="green", s=100, zorder=5, marker="*")
    ax.annotate("Optimal\nComplexity",
                xy=(complexity[optimal_idx], total_error[optimal_idx]),
                xytext=(complexity[optimal_idx] + 1.5, total_error[optimal_idx] + 0.5),
                fontsize=11, fontweight="bold", color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.5))

    # Mark regions
    ax.annotate("UNDERFITTING\n(High Bias,\nLow Variance)",
                xy=(1.5, 3.5), fontsize=10, ha="center", color="blue",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    ax.annotate("OVERFITTING\n(Low Bias,\nHigh Variance)",
                xy=(8.5, 3.5), fontsize=10, ha="center", color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5))

    # Mark Random Forest position
    rf_x = complexity[optimal_idx] + 0.3
    ax.annotate("Random Forest\n(Low Bias + Low Variance\nvia Ensemble Averaging)",
                xy=(rf_x, total_error[optimal_idx] + 0.1),
                xytext=(rf_x + 1.5, total_error[optimal_idx] + 1.8),
                fontsize=10, ha="center", color="purple", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.5),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.5))

    ax.set_xlabel("Model Complexity →", fontsize=12, fontweight="bold")
    ax.set_ylabel("Error →", fontsize=12, fontweight="bold")
    ax.set_title("Figure 9: Bias-Variance Tradeoff in Machine Learning",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper center", fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 6)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig("report_outputs/figures/fig9_bias_variance_diagram.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("       Saved: report_outputs/figures/fig9_bias_variance_diagram.png")


# =============================================================================
# TABLE 1: DATASET STATISTICS
# =============================================================================
def generate_table1_statistics(raw_df):
    """Descriptive statistics for all features."""
    print("\n[TABLE 1] Generating dataset statistics...")

    stats = raw_df.describe().round(3).T
    stats.to_csv("report_outputs/tables/table1_dataset_statistics.csv")
    print("       Saved: report_outputs/tables/table1_dataset_statistics.csv")
    return stats


# =============================================================================
# TABLE 2: MISSING VALUES
# =============================================================================
def generate_table2_missing_values(raw_df):
    """Missing value count per feature."""
    print("\n[TABLE 2] Generating missing values table...")

    missing = raw_df.isnull().sum()
    missing_df = pd.DataFrame({
        "Feature": missing.index,
        "Missing Count": missing.values,
        "Missing Percentage (%)": (missing.values / len(raw_df) * 100).round(2)
    })
    missing_df.to_csv("report_outputs/tables/table2_missing_values.csv", index=False)
    print("       Saved: report_outputs/tables/table2_missing_values.csv")
    return missing_df


# =============================================================================
# TABLE 3: CLASS DISTRIBUTION
# =============================================================================
def generate_table3_class_distribution(raw_df):
    """Target variable class distribution."""
    print("\n[TABLE 3] Generating class distribution table...")

    counts = raw_df["burnout_risk"].value_counts()
    dist_df = pd.DataFrame({
        "Burnout Risk Level": counts.index,
        "Count": counts.values,
        "Percentage (%)": (counts.values / len(raw_df) * 100).round(2)
    })
    dist_df.to_csv("report_outputs/tables/table3_class_distribution.csv", index=False)
    print("       Saved: report_outputs/tables/table3_class_distribution.csv")
    return dist_df


# =============================================================================
# TABLE 4: EVALUATION METRICS
# =============================================================================
def generate_table4_evaluation_metrics(data, y_pred):
    """Test set evaluation metrics."""
    print("\n[TABLE 4] Generating evaluation metrics table...")

    metrics = {
        "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"],
        "Score": [
            accuracy_score(data["y_test"], y_pred),
            precision_score(data["y_test"], y_pred, average="macro", zero_division=0),
            recall_score(data["y_test"], y_pred, average="macro", zero_division=0),
            f1_score(data["y_test"], y_pred, average="macro", zero_division=0)
        ]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_df["Score"] = metrics_df["Score"].round(4)
    metrics_df["Percentage (%)"] = (metrics_df["Score"] * 100).round(2)
    metrics_df.to_csv("report_outputs/tables/table4_evaluation_metrics.csv", index=False)
    print("       Saved: report_outputs/tables/table4_evaluation_metrics.csv")
    return metrics_df


# =============================================================================
# TABLE 5: CLASSIFICATION REPORT
# =============================================================================
def generate_table5_classification_report(data, y_pred):
    """Per-class classification report as a table."""
    print("\n[TABLE 5] Generating classification report table...")

    class_names = list(data["label_encoder"].classes_)
    report_dict = classification_report(
        data["y_test"], y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).T.round(4)
    report_df.to_csv("report_outputs/tables/table5_classification_report.csv")
    print("       Saved: report_outputs/tables/table5_classification_report.csv")
    return report_df


# =============================================================================
# TABLE 6: CROSS-VALIDATION SCORES
# =============================================================================
def generate_table6_cv_scores(cv_results):
    """Cross-validation scores table."""
    print("\n[TABLE 6] Generating cross-validation scores table...")

    cv_df = pd.DataFrame(cv_results).T
    cv_df = cv_df.round(4)
    cv_df.index.name = "Metric"
    cv_df.to_csv("report_outputs/tables/table6_cross_validation_scores.csv")
    print("       Saved: report_outputs/tables/table6_cross_validation_scores.csv")
    return cv_df


# =============================================================================
# TABLE 7: FEATURE IMPORTANCE
# =============================================================================
def generate_table7_feature_importance(feature_importances):
    """Feature importance scores table."""
    print("\n[TABLE 7] Generating feature importance table...")

    fi_df = pd.DataFrame(feature_importances, columns=["Feature", "Importance Score"])
    fi_df["Rank"] = range(1, len(fi_df) + 1)
    fi_df["Importance (%)"] = (fi_df["Importance Score"] * 100).round(2)
    fi_df = fi_df[["Rank", "Feature", "Importance Score", "Importance (%)"]]
    fi_df.to_csv("report_outputs/tables/table7_feature_importance.csv", index=False)
    print("       Saved: report_outputs/tables/table7_feature_importance.csv")
    return fi_df


# =============================================================================
# TABLE 8: CONFUSION MATRIX
# =============================================================================
def generate_table8_confusion_matrix(data, y_pred):
    """Confusion matrix as a labelled table."""
    print("\n[TABLE 8] Generating confusion matrix table...")

    class_names = list(data["label_encoder"].classes_)
    cm = confusion_matrix(data["y_test"], y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {c}" for c in class_names],
        columns=[f"Predicted: {c}" for c in class_names]
    )
    cm_df.to_csv("report_outputs/tables/table8_confusion_matrix.csv")
    print("       Saved: report_outputs/tables/table8_confusion_matrix.csv")
    return cm_df


# =============================================================================
# TABLE 9: MODEL HYPERPARAMETERS
# =============================================================================
def generate_table9_hyperparameters(trainer):
    """Model hyperparameters table."""
    print("\n[TABLE 9] Generating hyperparameters table...")

    params = trainer.model.get_params()
    important_params = {
        "n_estimators": "Number of trees in the forest",
        "max_depth": "Maximum depth of each tree",
        "min_samples_split": "Minimum samples to split a node",
        "min_samples_leaf": "Minimum samples in a leaf node",
        "max_features": "Features considered per split",
        "class_weight": "Class weighting strategy",
        "random_state": "Random seed for reproducibility",
        "oob_score": "Out-of-bag score enabled",
        "n_jobs": "CPU cores used (-1 = all)"
    }

    hp_data = []
    for param, description in important_params.items():
        hp_data.append({
            "Parameter": param,
            "Value": str(params.get(param, "N/A")),
            "Description": description
        })

    hp_df = pd.DataFrame(hp_data)
    hp_df.to_csv("report_outputs/tables/table9_hyperparameters.csv", index=False)
    print("       Saved: report_outputs/tables/table9_hyperparameters.csv")
    return hp_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """Generate all report figures and tables."""
    print("\n" + "=" * 70)
    print("   REPORT FIGURES AND TABLES GENERATOR")
    print("   Generating all outputs for coursework report...")
    print("=" * 70)

    create_output_dirs()

    # Load everything
    raw_df, data, preprocessor, trainer, y_pred, y_pred_proba = load_all_data()

    # ==================== FIGURES ====================
    generate_fig1_class_distribution(raw_df)
    generate_fig2_correlation_heatmap(raw_df)
    generate_fig3_confusion_matrix(data, y_pred)
    feature_importances = generate_fig4_feature_importance(trainer, data)
    cv_results = generate_fig5_cv_boxplot(data, trainer)
    generate_fig6_feature_distributions(raw_df)
    generate_fig7_feature_boxplots(raw_df)
    generate_fig8_prediction_example(trainer, preprocessor)
    generate_fig9_bias_variance_diagram()

    # ==================== TABLES ====================
    generate_table1_statistics(raw_df)
    generate_table2_missing_values(raw_df)
    generate_table3_class_distribution(raw_df)
    generate_table4_evaluation_metrics(data, y_pred)
    generate_table5_classification_report(data, y_pred)
    generate_table6_cv_scores(cv_results)
    generate_table7_feature_importance(feature_importances)
    generate_table8_confusion_matrix(data, y_pred)
    generate_table9_hyperparameters(trainer)

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("   ALL REPORT OUTPUTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    print("\n   📁 Figures (9 PNG files):")
    for f in sorted(os.listdir("report_outputs/figures")):
        print(f"       ✅ report_outputs/figures/{f}")
    print("\n   📁 Tables (9 CSV files):")
    for f in sorted(os.listdir("report_outputs/tables")):
        print(f"       ✅ report_outputs/tables/{f}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
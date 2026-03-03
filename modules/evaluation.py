"""
===============================================================================
MODEL EVALUATION MODULE
===============================================================================
Purpose:
    Provides comprehensive evaluation of the trained Random Forest model
    using standard classification metrics and visualisations.

Evaluation Metrics Included:
    1. Accuracy — overall proportion of correct predictions
    2. Precision — of all predicted positives, how many are truly positive
    3. Recall — of all actual positives, how many were correctly identified
    4. F1-Score — harmonic mean of Precision and Recall
    5. Confusion Matrix — detailed breakdown of predictions vs actual classes
    6. Classification Report — per-class precision, recall, F1

AI Syllabus Relevance:
    - These metrics are fundamental to evaluating ANY classification model.
    - Accuracy alone is misleading for imbalanced datasets.
    - Confusion Matrix reveals which classes the model confuses.
    - F1-Score balances precision and recall into a single metric.

Author: Rohit Sharma

===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import os


class ModelEvaluator:


    def __init__(self, y_test, y_pred, class_names=None):

        self.y_test = y_test
        self.y_pred = y_pred
        self.class_names = class_names if class_names is not None else \
            [str(c) for c in sorted(np.unique(y_test))]
        self.metrics = {}

    def compute_metrics(self):

        self.metrics = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "precision_macro": precision_score(
                self.y_test, self.y_pred, average="macro", zero_division=0
            ),
            "recall_macro": recall_score(
                self.y_test, self.y_pred, average="macro", zero_division=0
            ),
            "f1_macro": f1_score(
                self.y_test, self.y_pred, average="macro", zero_division=0
            )
        }

        print("\n" + "=" * 60)
        print("MODEL EVALUATION METRICS (TEST SET)")
        print("=" * 60)
        for metric, value in self.metrics.items():
            print(f"  {metric:20s}: {value:.4f} ({value * 100:.2f}%)")
        print("=" * 60)

        return self.metrics

    def generate_confusion_matrix(self):

        cm = confusion_matrix(self.y_test, self.y_pred)
        print("\n[INFO] Confusion Matrix:")
        print(cm)
        return cm

    def plot_confusion_matrix(self, save_path="modules/confusion_matrix.png"):

        cm = confusion_matrix(self.y_test, self.y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": "Count"}
        )
        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
        ax.set_title(
            "Confusion Matrix — Random Forest Classifier",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        # Save figure
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix plot saved to: {save_path}")

        return fig

    def print_classification_report(self):

        print("\n" + "=" * 60)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 60)
        report = classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        print(report)
        return report

    def plot_feature_importance(
        self,
        feature_importances,
        save_path="modules/feature_importance.png"
    ):

        names = [fi[0] for fi in feature_importances]
        scores = [fi[1] for fi in feature_importances]

        # Reverse for horizontal bar chart (highest at top)
        names_reversed = names[::-1]
        scores_reversed = scores[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))[::-1]
        bars = ax.barh(names_reversed, scores_reversed, color=colors, edgecolor="white")

        # Add value labels on bars
        for bar, score in zip(bars, scores_reversed):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                va="center",
                fontsize=10,
                fontweight="bold"
            )

        ax.set_xlabel("Importance Score (Gini / MDI)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Feature Importances — Random Forest Classifier",
            fontsize=14, fontweight="bold"
        )
        ax.set_xlim(0, max(scores) * 1.2)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Feature importance plot saved to: {save_path}")

        return fig

    def get_evaluation_summary(self):

        if not self.metrics:
            self.compute_metrics()

        summary = "MODEL EVALUATION SUMMARY\n"
        summary += "=" * 40 + "\n"
        for metric, value in self.metrics.items():
            summary += f"  {metric:20s}: {value:.4f}\n"
        summary += "=" * 40
        return summary
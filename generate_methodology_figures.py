import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT_DIR = "report_outputs/methodology"


def _box(ax, x, y, w, h, text, fontsize=10):
    rect = Rectangle((x, y), w, h, fill=False, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def _arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=15, linewidth=1.6)
    ax.add_patch(arr)


def figM1_training_pipeline():
    """Training pipeline flow chart."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)

    _box(ax, 0.3, 2.2, 1.7, 1.0, "Load CSV\nDataset")
    _box(ax, 2.3, 2.2, 1.9, 1.0, "Missing Value\nImputation\n(Median)")
    _box(ax, 4.5, 2.2, 1.7, 1.0, "Encode Target\n(LabelEncoder)")
    _box(ax, 6.5, 2.2, 1.7, 1.0, "Train/Test Split\n(80/20, Stratified)")
    _box(ax, 8.5, 2.2, 1.4, 1.0, "Scale\n(StandardScaler)")
    _box(ax, 10.2, 2.2, 1.6, 1.0, "RF Train\n+ Save\nArtifacts")

    _box(ax, 6.5, 0.6, 2.7, 1.0, "SKFCV (K=5)\non Training Set\n(Validation)")

    # arrows top row
    _arrow(ax, 2.0, 2.7, 2.3, 2.7)
    _arrow(ax, 4.2, 2.7, 4.5, 2.7)
    _arrow(ax, 6.2, 2.7, 6.5, 2.7)
    _arrow(ax, 8.2, 2.7, 8.5, 2.7)
    _arrow(ax, 9.9, 2.7, 10.2, 2.7)

    # arrow down to CV from split
    _arrow(ax, 7.35, 2.2, 7.35, 1.6)
    # arrow from CV to train
    _arrow(ax, 9.2, 1.1, 10.2, 2.2)

    ax.set_title("Figure M1: End-to-End Training Pipeline (Methodology)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figM1_training_pipeline.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figM2_stratified_kfold():
    """Diagram of 5-fold stratified CV (conceptual)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)

    ax.text(0.3, 3.5, "Stratified 5-Fold Cross-Validation (each fold preserves class proportions)", fontsize=12, fontweight="bold")

    folds = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    x0 = 0.8
    w = 1.9
    gap = 0.25
    y = 2.3
    h = 0.8

    for i, f in enumerate(folds):
        _box(ax, x0 + i*(w+gap), y, w, h, f)

    ax.text(0.3, 1.55, "Iteration example:", fontsize=11, fontweight="bold")
    ax.text(0.3, 1.15, "• Train on 4 folds, validate on 1 fold (rotates)\n• Repeat 5 times → average score + std (stability/variance evidence)", fontsize=10)

    # highlight "validation fold" example (Fold 3)
    val_idx = 2
    rect = Rectangle((x0 + val_idx*(w+gap), y), w, h, fill=True, alpha=0.15, linewidth=2.2)
    ax.add_patch(rect)
    ax.text(x0 + val_idx*(w+gap) + w/2, y-0.25, "Validation fold\n(example)", ha="center", va="top", fontsize=9)

    ax.set_title("Figure M2: Stratified K-Fold Cross-Validation (Methodology)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figM2_stratified_kfold.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figM3_random_forest_concept():
    """Random Forest concept diagram: bootstrap samples -> trees -> vote."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)

    _box(ax, 0.4, 1.7, 2.2, 1.0, "Training Data\n(N samples)")
    _box(ax, 3.0, 2.6, 2.0, 0.8, "Bootstrap\nSample #1")
    _box(ax, 3.0, 1.6, 2.0, 0.8, "Bootstrap\nSample #2")
    _box(ax, 3.0, 0.6, 2.0, 0.8, "Bootstrap\nSample #B")

    _box(ax, 5.6, 2.6, 1.6, 0.8, "Tree 1")
    _box(ax, 5.6, 1.6, 1.6, 0.8, "Tree 2")
    _box(ax, 5.6, 0.6, 1.6, 0.8, "Tree B")

    _box(ax, 8.0, 1.7, 3.6, 1.0, "Aggregate Votes\nMajority Class\n+ Probabilities\n(predict_proba)")

    # arrows
    _arrow(ax, 2.6, 2.2, 3.0, 3.0)
    _arrow(ax, 2.6, 2.2, 3.0, 2.0)
    _arrow(ax, 2.6, 2.2, 3.0, 1.0)

    _arrow(ax, 5.0, 3.0, 5.6, 3.0)
    _arrow(ax, 5.0, 2.0, 5.6, 2.0)
    _arrow(ax, 5.0, 1.0, 5.6, 1.0)

    _arrow(ax, 7.2, 3.0, 8.0, 2.2)
    _arrow(ax, 7.2, 2.0, 8.0, 2.2)
    _arrow(ax, 7.2, 1.0, 8.0, 2.2)

    ax.set_title("Figure M3: Random Forest Concept (Bagging + Voting)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figM3_random_forest_concept.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figM4_hyperparameter_bvt():
    """Conceptual table/diagram of hyperparameters vs bias/variance."""
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_axis_off()

    rows = [
        ("n_estimators ↑", "Variance ↓ (more averaging)", "Bias ~ (usually unchanged)", "More stable; diminishing returns"),
        ("max_depth ↑", "Variance ↑", "Bias ↓", "Deeper trees fit training more"),
        ("min_samples_leaf ↑", "Variance ↓", "Bias ↑", "Regularisation; smoother boundaries"),
        ("max_features ↓", "Variance ↓ (decorrelation)", "Bias ↑ (if too small)", "Trees become more diverse"),
    ]

    col_labels = ["Change", "Effect on Variance", "Effect on Bias", "Interpretation"]
    table_data = [list(r) for r in rows]

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    ax.set_title("Figure M4: Conceptual Hyperparameter Effects on Bias–Variance", fontsize=14, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figM4_hyperparameter_bvt.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    figM1_training_pipeline()
    figM2_stratified_kfold()
    figM3_random_forest_concept()
    figM4_hyperparameter_bvt()
    print(f"[INFO] Methodology figures generated in: {OUT_DIR}")


if __name__ == "__main__":
    main()
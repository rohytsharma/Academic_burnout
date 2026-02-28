"""
===============================================================================
STREAMLIT WEB APPLICATION
===============================================================================
Purpose:
    Interactive web interface for the Academic Burnout Risk Prediction System.
    Allows users to:
        1. Input student academic/lifestyle features via sliders and inputs
        2. Receive a burnout risk prediction (Low / Medium / High)
        3. View prediction probability scores
        4. Explore feature importance visualisation
        5. Review model evaluation metrics and confusion matrix

    Launch with:
        streamlit run streamlit_app.py

Author: Student
Course: Undergraduate AI Coursework
===============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_preprocessing import DataPreprocessor
from modules.model_training import BurnoutModelTrainer
from modules.evaluation import ModelEvaluator


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Burnout Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# LOAD MODEL AND ARTIFACTS (cached to avoid reloading on every interaction)
# =============================================================================
@st.cache_resource
def load_model_and_artifacts():
    """
    Load the trained Random Forest model, scaler, and label encoder.
    Uses Streamlit's caching to load only once.

    Returns
    -------
    tuple
        (BurnoutModelTrainer, DataPreprocessor) with loaded artifacts.
    """
    trainer = BurnoutModelTrainer()
    trainer.load_model("models/random_forest_model.pkl")

    preprocessor = DataPreprocessor()
    preprocessor.load_artifacts("models")

    return trainer, preprocessor


@st.cache_data
def load_dataset():
    """Load the dataset for display and analysis."""
    return pd.read_csv("data/student_burnout_dataset.csv")


# =============================================================================
# CHECK IF MODEL EXISTS
# =============================================================================
if not os.path.exists("models/random_forest_model.pkl"):
    st.error(
        "⚠️ **Trained model not found!**\n\n"
        "Please run the training pipeline first:\n\n"
        "```\npython train_pipeline.py\n```"
    )
    st.stop()


# Load model and preprocessor
trainer, preprocessor = load_model_and_artifacts()


# =============================================================================
# SIDEBAR — NAVIGATION
# =============================================================================
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio(
    "Select Page:",
    [
        "🏠 Home",
        "🔮 Predict Burnout Risk",
        "📊 Model Evaluation",
        "📈 Feature Importance",
        "📚 Dataset Explorer",
        "🧠 About the Model"
    ]
)


# =============================================================================
# PAGE 1: HOME
# =============================================================================
if page == "🏠 Home":
    st.title("🎓 AI-Based Academic Burnout Risk Prediction System")
    st.markdown("---")

    st.markdown("""
    ### Welcome

    This system uses a **Random Forest Classifier** — an ensemble machine 
    learning model — to predict whether a student is at **Low**, **Medium**, 
    or **High** risk of academic burnout based on their academic performance 
    and lifestyle factors.

    ### 📋 Features Used for Prediction

    | Feature | Description | Range |
    |---------|-------------|-------|
    | Attendance | Class attendance percentage | 0–100% |
    | Study Hours | Daily study hours | 0–14 hours |
    | Sleep Hours | Daily sleep hours | 3–10 hours |
    | Assignment Completion | Assignment completion rate | 0–100% |
    | GPA | Grade Point Average | 0.0–4.0 |
    | Stress Level | Self-reported stress | 1–10 |
    | Screen Time | Daily recreational screen time | 1–16 hours |
    | Part-time Hours | Weekly part-time work hours | 0–30 hours |
    | Social Hours | Daily social/leisure hours | 0–8 hours |

    ### 🔮 How to Use
    1. Navigate to **Predict Burnout Risk** in the sidebar
    2. Adjust the sliders to match a student's profile
    3. Click **Predict** to see the burnout risk level and probabilities
    4. Explore **Model Evaluation** and **Feature Importance** for deeper insights
    """)

    st.markdown("---")

    # Quick stats
    try:
        df = load_dataset()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Records", f"{len(df)}")
        col2.metric("🔢 Features", "9")
        col3.metric("🎯 Classes", "3")
        col4.metric("🌲 Trees in Forest", "200")
    except Exception:
        pass


# =============================================================================
# PAGE 2: PREDICT BURNOUT RISK
# =============================================================================
elif page == "🔮 Predict Burnout Risk":
    st.title("🔮 Burnout Risk Prediction")
    st.markdown("Adjust the sliders below to input a student's profile, "
                "then click **Predict** to see their burnout risk level.")
    st.markdown("---")

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Academic Factors")
        attendance = st.slider(
            "Attendance (%)", 10.0, 100.0, 75.0, 0.5,
            help="Class attendance percentage"
        )
        study_hours = st.slider(
            "Study Hours (daily)", 0.0, 14.0, 5.0, 0.5,
            help="Average daily study hours"
        )
        assignment_completion = st.slider(
            "Assignment Completion (%)", 0.0, 100.0, 70.0, 1.0,
            help="Percentage of assignments completed"
        )
        gpa = st.slider(
            "GPA", 0.0, 4.0, 2.8, 0.05,
            help="Current Grade Point Average"
        )
        stress_level = st.slider(
            "Stress Level", 1, 10, 5,
            help="Self-reported stress level (1=low, 10=high)"
        )

    with col2:
        st.subheader("🏃 Lifestyle Factors")
        sleep_hours = st.slider(
            "Sleep Hours (daily)", 3.0, 10.0, 6.5, 0.5,
            help="Average daily sleep hours"
        )
        screen_time = st.slider(
            "Screen Time (daily hours)", 1.0, 16.0, 5.0, 0.5,
            help="Daily recreational screen time"
        )
        part_time_hours = st.slider(
            "Part-time Work (weekly hours)", 0.0, 30.0, 8.0, 0.5,
            help="Weekly part-time work hours"
        )
        social_hours = st.slider(
            "Social Hours (daily)", 0.0, 8.0, 3.0, 0.5,
            help="Daily social and leisure hours"
        )

    st.markdown("---")

    # Predict button
    if st.button("🔮 Predict Burnout Risk", type="primary", use_container_width=True):
        # Prepare input
        input_dict = {
            "attendance": attendance,
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "assignment_completion": assignment_completion,
            "gpa": gpa,
            "stress_level": stress_level,
            "screen_time": screen_time,
            "part_time_hours": part_time_hours,
            "social_hours": social_hours
        }

        # Preprocess and predict
        input_scaled = preprocessor.preprocess_single_input(input_dict)
        prediction = trainer.predict(input_scaled)[0]
        probabilities = trainer.predict_proba(input_scaled)[0]

        # Decode prediction
        predicted_label = preprocessor.label_encoder.inverse_transform([prediction])[0]
        class_names = preprocessor.label_encoder.classes_

        # Display result
        st.markdown("---")
        st.subheader("📋 Prediction Result")

        # Color-coded result
        color_map = {"Low": "green", "Medium": "orange", "High": "red"}
        emoji_map = {"Low": "✅", "Medium": "⚠️", "High": "🚨"}

        result_color = color_map.get(predicted_label, "blue")
        result_emoji = emoji_map.get(predicted_label, "")

        st.markdown(
            f"<h2 style='text-align: center; color: {result_color};'>"
            f"{result_emoji} Burnout Risk: {predicted_label} {result_emoji}"
            f"</h2>",
            unsafe_allow_html=True
        )

        # Probability display
        st.subheader("📊 Prediction Probabilities")
        st.markdown(
            "These probabilities represent the proportion of trees in the "
            "Random Forest that voted for each class."
        )

        prob_cols = st.columns(len(class_names))
        for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
            with prob_cols[i]:
                cls_emoji = emoji_map.get(cls, "")
                st.metric(
                    label=f"{cls_emoji} {cls}",
                    value=f"{prob * 100:.1f}%"
                )

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        bar_colors = [color_map.get(c, "blue") for c in class_names]
        bars = ax.barh(class_names, probabilities * 100, color=bar_colors,
                       edgecolor="white", height=0.5)
        for bar, prob in zip(bars, probabilities):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{prob * 100:.1f}%", va="center", fontweight="bold")
        ax.set_xlabel("Probability (%)")
        ax.set_title("Prediction Probability Distribution")
        ax.set_xlim(0, 110)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Input summary table
        st.subheader("📝 Input Summary")
        input_df = pd.DataFrame([input_dict])
        st.dataframe(input_df, use_container_width=True)


# =============================================================================
# PAGE 3: MODEL EVALUATION
# =============================================================================
elif page == "📊 Model Evaluation":
    st.title("📊 Model Evaluation Metrics")
    st.markdown(
        "Evaluation results from the trained Random Forest model on the "
        "**held-out test set** (20% of data, not used during training)."
    )
    st.markdown("---")

    # Load test data and evaluate
    try:
        dataset_path = "data/student_burnout_dataset.csv"
        preprocessor_eval = DataPreprocessor()
        data = preprocessor_eval.preprocess_pipeline(dataset_path)

        trainer_eval = BurnoutModelTrainer()
        trainer_eval.load_model("models/random_forest_model.pkl")

        y_pred = trainer_eval.predict(data["X_test_scaled"])

        evaluator = ModelEvaluator(
            y_test=data["y_test"],
            y_pred=y_pred,
            class_names=list(data["label_encoder"].classes_)
        )

        metrics = evaluator.compute_metrics()

        # Display metrics in columns
        st.subheader("🎯 Performance Metrics")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        mcol2.metric("Precision", f"{metrics['precision_macro']:.4f}")
        mcol3.metric("Recall", f"{metrics['recall_macro']:.4f}")
        mcol4.metric("F1-Score", f"{metrics['f1_macro']:.4f}")

        # Metric explanations
        with st.expander("📖 What do these metrics mean?"):
            st.markdown("""
            - **Accuracy**: Proportion of all predictions that are correct.
              *Simple but can be misleading for imbalanced classes.*

            - **Precision (Macro)**: Of all students predicted as a certain risk 
              level, how many truly belong to that level? 
              *Averaged equally across all classes.*

            - **Recall (Macro)**: Of all students who truly belong to a risk 
              level, how many did the model correctly identify? 
              *Averaged equally across all classes.*

            - **F1-Score (Macro)**: Harmonic mean of Precision and Recall. 
              *Provides a single balanced metric. Range: 0 (worst) to 1 (best).*
            """)

        st.markdown("---")

        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        st.markdown(
            "Rows = Actual class, Columns = Predicted class. "
            "Diagonal values are correct predictions."
        )

        fig_cm = evaluator.plot_confusion_matrix("models/confusion_matrix.png")
        st.pyplot(fig_cm)
        plt.close()

        with st.expander("📖 How to read the Confusion Matrix"):
            st.markdown("""
            The confusion matrix is a table showing prediction outcomes:
            - **Diagonal cells** (top-left to bottom-right): Correct predictions 
              (True Positives for each class)
            - **Off-diagonal cells**: Misclassifications
            - **Row totals**: Actual count of each class in the test set
            - **Column totals**: Predicted count of each class

            For example, if cell (High, Medium) = 3, it means 3 students who 
            were actually High risk were incorrectly predicted as Medium risk.
            """)

        st.markdown("---")

        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        report = evaluator.print_classification_report()
        st.code(report, language="text")

        st.markdown("---")

        # Cross-validation scores
        st.subheader("🔄 Cross-Validation Results")
        st.markdown(
            "5-Fold Stratified Cross-Validation scores on the training set. "
            "Each fold uses 80% for training and 20% for validation."
        )

        cv_scores = trainer_eval.cross_validate(
            data["X_train_scaled"], data["y_train"], cv_folds=5
        )

        cv_data = []
        for metric_name, scores in cv_scores.items():
            cv_data.append({
                "Metric": metric_name,
                "Fold 1": f"{scores[0]:.4f}",
                "Fold 2": f"{scores[1]:.4f}",
                "Fold 3": f"{scores[2]:.4f}",
                "Fold 4": f"{scores[3]:.4f}",
                "Fold 5": f"{scores[4]:.4f}",
                "Mean": f"{scores.mean():.4f}",
                "Std": f"{scores.std():.4f}"
            })

        cv_df = pd.DataFrame(cv_data)
        st.dataframe(cv_df, use_container_width=True)

        with st.expander("📖 Understanding Cross-Validation"):
            st.markdown("""
            **K-Fold Cross-Validation** divides training data into K folds:
            - The model is trained K times, each time holding out a different fold
            - This produces K performance estimates
            - The mean gives a reliable performance estimate
            - The standard deviation (Std) indicates model stability

            **Low Std** → Model performs consistently across different data subsets 
            (low variance, good generalisation)

            **High Std** → Model is sensitive to which data it trains on 
            (high variance, potential overfitting)
            """)

    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")


# =============================================================================
# PAGE 4: FEATURE IMPORTANCE
# =============================================================================
elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance Analysis")
    st.markdown(
        "Feature importance scores from the Random Forest model, calculated "
        "using **Mean Decrease in Impurity (Gini Importance)**."
    )
    st.markdown("---")

    try:
        # Get feature importances
        feature_importances = trainer.get_feature_importance(
            preprocessor.feature_names
        )

        # Create and display the plot
        evaluator_fi = ModelEvaluator(
            y_test=[0], y_pred=[0],
            class_names=list(preprocessor.label_encoder.classes_)
        )
        fig = evaluator_fi.plot_feature_importance(
            feature_importances,
            "models/feature_importance.png"
        )
        st.pyplot(fig)
        plt.close()

        # Table
        st.subheader("📋 Importance Scores Table")
        fi_df = pd.DataFrame(
            feature_importances,
            columns=["Feature", "Importance Score"]
        )
        fi_df["Rank"] = range(1, len(fi_df) + 1)
        fi_df = fi_df[["Rank", "Feature", "Importance Score"]]
        st.dataframe(fi_df, use_container_width=True)

        with st.expander("📖 How is Feature Importance calculated?"):
            st.markdown("""
            **Mean Decrease in Impurity (MDI) / Gini Importance:**

            1. For each tree in the forest, at each split node, the algorithm 
               calculates how much the **Gini impurity** decreases due to that split.

            2. For each feature, the total decrease in impurity across ALL nodes 
               that use that feature is summed up, across ALL trees.

            3. These values are then **normalised** to sum to 1.0.

            **Interpretation:**
            - Higher importance → the feature is more useful for distinguishing 
              between burnout risk levels
            - A feature with importance 0.0 is never used for splitting
            - Feature importance does NOT indicate direction (positive/negative 
              effect) — only discriminative power
            """)

    except Exception as e:
        st.error(f"Error loading feature importance: {str(e)}")


# =============================================================================
# PAGE 5: DATASET EXPLORER
# =============================================================================
elif page == "📚 Dataset Explorer":
    st.title("📚 Dataset Explorer")
    st.markdown("Explore the synthetic student burnout dataset used for training.")
    st.markdown("---")

    try:
        df = load_dataset()

        # Basic stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Features", len(df.columns) - 1)
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("---")

        # Class distribution
        st.subheader("🎯 Burnout Risk Distribution")
        class_counts = df["burnout_risk"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
        bar_colors = [colors.get(c, "blue") for c in class_counts.index]
        ax.bar(class_counts.index, class_counts.values, color=bar_colors,
               edgecolor="white")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Burnout Risk Classes")
        for i, (idx, val) in enumerate(zip(class_counts.index, class_counts.values)):
            ax.text(i, val + 2, str(val), ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # Descriptive statistics
        st.subheader("📊 Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)

        # Correlation heatmap
        st.subheader("🔗 Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5,
            vmin=-1, vmax=1
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")


# =============================================================================
# PAGE 6: ABOUT THE MODEL
# =============================================================================
elif page == "🧠 About the Model":
    st.title("🧠 About the Random Forest Model")
    st.markdown("---")

    st.markdown("""
    ## What is Random Forest?

    Random Forest is an **ensemble learning** method that builds multiple 
    **Decision Trees** during training and combines their predictions through 
    **majority voting** (for classification).

    It was introduced by **Leo Breiman in 2001** and is one of the most 
    widely used machine learning algorithms due to its accuracy, robustness, 
    and interpretability.

    ---

    ## How Does It Work?

    ### Step 1: Bootstrap Sampling (Bagging)
    From the original training dataset of N samples, create B random samples 
    **with replacement** (bootstrap samples), each of size N.

    Some samples appear multiple times; others may not appear at all in a 
    given bootstrap sample (~37% are left out on average — these form the 
    **Out-of-Bag (OOB)** samples).

    ### Step 2: Growing Trees with Random Feature Subsets
    For each bootstrap sample, grow a Decision Tree:
    - At each node, instead of considering **all** features for the best split, 
      randomly select a **subset** of features (typically √p for classification)
    - Find the best split among this random subset
    - Grow the tree fully (no pruning by default)

    ### Step 3: Ensemble Prediction
    To predict a new sample:
    - Pass it through **all B trees**
    - Each tree outputs a class prediction
    - The final prediction is the **majority vote** across all trees
    - **Probability** = proportion of trees voting for each class

    ---

    ## Why Random Forest Reduces Variance (Not Bias)

    ### The Problem with Single Decision Trees
    A single deep Decision Tree has **low bias** (it can fit complex patterns) 
    but **high variance** (small changes in data → very different tree structure 
    → different predictions).

    ### How Random Forest Solves This
    1. **Bagging** (averaging many models) reduces variance:
       - Var(average) = Var(individual) / B + correlation term
    2. **Random feature subsets** reduce correlation between trees:
       - If one feature dominates, not every tree uses it first
       - Trees learn different patterns → more diverse → lower correlation
    3. The **bias stays low** because each tree is still deep and expressive

    ### Result:
    > Random Forest = Low Bias + Low Variance = Good Generalisation

    ---

    ## Bias-Variance Tradeoff

    | Aspect | Single Decision Tree | Random Forest |
    |--------|---------------------|---------------|
    | Bias | Low (deep tree) | Low (deep trees) |
    | Variance | **High** (overfitting) | **Low** (ensemble averaging) |
    | Risk | Overfitting | Well-balanced |

    ### How Hyperparameters Affect the Tradeoff:

    | Parameter | ↑ Value Effect on Bias | ↑ Value Effect on Variance |
    |-----------|----------------------|---------------------------|
    | n_estimators | No change | ↓ Decreases (more averaging) |
    | max_depth | ↓ Decreases | ↑ Increases per tree |
    | min_samples_split | ↑ Increases | ↓ Decreases |
    | max_features | ↑ Increases | ↓ Decreases (more decorrelation) |

    ---

    ## Model Configuration Used in This Project

    | Parameter | Value | Justification |
    |-----------|-------|---------------|
    | n_estimators | 200 | Sufficient trees for stable ensemble |
    | max_depth | 15 | Deep enough for 9 features, with limit to prevent overfitting |
    | min_samples_split | 5 | Regularisation against noisy splits |
    | min_samples_leaf | 2 | Ensures leaf nodes are meaningful |
    | max_features | √p = 3 | Decorrelates trees effectively |
    | class_weight | balanced | Handles potential class imbalance |

    ---

    ## Cross-Validation

    **5-Fold Stratified Cross-Validation** is used to evaluate the model:
    1. Training data is split into 5 equal folds
    2. Model is trained 5 times, each time using 4 folds for training 
       and 1 fold for validation
    3. The mean score across folds estimates true generalisation performance
    4. The standard deviation indicates model stability

    **Stratification** ensures each fold maintains the same proportion 
    of Low/Medium/High samples as the full dataset.

    ---

    ## Key Advantages of Random Forest
    - ✅ Handles non-linear relationships without feature engineering
    - ✅ Robust to outliers (tree-based splits are invariant to scaling)
    - ✅ Provides feature importance rankings
    - ✅ Rarely overfits when n_estimators is sufficiently large
    - ✅ Works well with mixed feature types and missing values
    - ✅ Provides probability estimates via tree voting proportions
    """)

    st.markdown("---")
    st.info(
        "💡 This project uses ONLY Random Forest Classifier. "
        "No model comparison is performed as per project requirements."
    )


# =============================================================================
# FOOTER
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**AI Coursework Project**\n\n"
    "Random Forest Classifier\n\n"
    "Built with scikit-learn & Streamlit"
)
"""
===============================================================================
MODEL TRAINING MODULE — RANDOM FOREST CLASSIFIER
===============================================================================
Purpose:
    Trains a Random Forest Classifier for predicting student burnout risk.
    This module handles model instantiation, training, cross-validation,
    hyperparameter configuration, and model persistence.

===============================================================================
RANDOM FOREST — THEORETICAL BACKGROUND
===============================================================================

What is Random Forest?
    Random Forest is an ENSEMBLE learning method that constructs multiple
    Decision Trees during training and outputs the MODE (most frequent class)
    of the individual trees' predictions for classification tasks.

    It was proposed by Leo Breiman in 2001 and combines two key ideas:
        1. BAGGING (Bootstrap Aggregating)
        2. RANDOM FEATURE SUBSETS

How Does Random Forest Work? (Step-by-Step)
    1. From the original training dataset of N samples, create B bootstrap
       samples (random samples WITH replacement, each of size N).
    2. For each bootstrap sample, grow a Decision Tree:
       a. At each node, instead of considering ALL features for the best
          split, randomly select a SUBSET of √p features (where p is the
          total number of features).
       b. Choose the best split among this random subset.
       c. Grow the tree fully (no pruning by default).
    3. To make a prediction, pass the input through ALL B trees and take
       a MAJORITY VOTE across their predictions.

Why Does Random Forest Reduce Variance?
    - A single Decision Tree is a HIGH VARIANCE model: small changes in
      training data can produce very different trees (overfitting).
    - By training many trees on different bootstrap samples and averaging
      their predictions, Random Forest REDUCES VARIANCE without
      significantly increasing bias.
    - The random feature subset selection DECORRELATES the trees. If one
      feature is very strong, not every tree will use it at the root,
      forcing trees to learn different patterns.

    Mathematically:
        Var(average of B trees) = (1/B) * σ² + ρ * σ²
        where ρ is the pairwise correlation between trees.
        By decorrelating trees (reducing ρ), the overall variance decreases.

===============================================================================
BIAS-VARIANCE TRADEOFF IN RANDOM FOREST
===============================================================================

Bias:
    - Bias measures how far the model's average prediction is from the true
      value. High bias means the model is too simple (underfitting).
    - Random Forest has LOW BIAS because each individual tree is a complex,
      fully-grown model that can capture non-linear relationships.

Variance:
    - Variance measures how much predictions change across different training
      sets. High variance means the model is too sensitive to training data.
    - A single Decision Tree has HIGH VARIANCE (overfitting).
    - Random Forest REDUCES VARIANCE through ensemble averaging and feature
      randomisation, while maintaining the low bias of individual trees.

The Tradeoff:
    - Ideal model: Low bias AND low variance.
    - Random Forest achieves this by:
        * Keeping bias low → each tree is deep and expressive
        * Reducing variance → averaging many decorrelated trees

Key Hyperparameters Affecting Bias-Variance:
    - n_estimators (number of trees):
        * More trees → lower variance (more averaging) → no increase in bias
        * Diminishing returns beyond ~100-200 trees
    - max_depth (maximum tree depth):
        * Deeper trees → lower bias, higher variance per tree
        * Shallower trees → higher bias, lower variance per tree
    - min_samples_split / min_samples_leaf:
        * Higher values → more regularisation → higher bias, lower variance
    - max_features:
        * Fewer features per split → more decorrelated trees → lower variance
        * Too few features → higher bias

===============================================================================

Author: ROhit Sharma

===============================================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os


class BurnoutModelTrainer:


    def __init__(self):
        """Initialise the trainer with no model."""
        self.model = None
        self.cv_scores = {}
        self.is_trained = False

    def create_model(
        self,
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    ):

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            class_weight=class_weight,
            # Out-of-bag score: uses samples NOT in each bootstrap to
            # estimate generalisation accuracy without separate validation set
            oob_score=True,
            n_jobs=-1  # Use all CPU cores for parallel tree construction
        )
        print("[INFO] Random Forest Classifier created with parameters:")
        print(f"       n_estimators     = {n_estimators}")
        print(f"       max_depth        = {max_depth}")
        print(f"       min_samples_split= {min_samples_split}")
        print(f"       min_samples_leaf = {min_samples_leaf}")
        print(f"       max_features     = {max_features}")
        print(f"       class_weight     = {class_weight}")
        return self.model

    def train(self, X_train, y_train):

        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")

        print("\n[INFO] Training Random Forest Classifier...")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        print(f"[INFO] Training complete.")
        print(f"[INFO] OOB Score (out-of-bag accuracy): "
              f"{self.model.oob_score_:.4f}")
        print(f"       OOB score is an unbiased estimate of generalisation")
        print(f"       accuracy using samples not in each tree's bootstrap.")

    def cross_validate(self, X_train, y_train, cv_folds=5):

        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")

        print(f"\n[INFO] Performing {cv_folds}-Fold Stratified Cross-Validation...")

        # Stratified K-Fold ensures class proportions are maintained
        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42
        )

        # Evaluate multiple metrics across folds
        scoring_metrics = {
            "accuracy": "accuracy",
            "precision_macro": "precision_macro",
            "recall_macro": "recall_macro",
            "f1_macro": "f1_macro"
        }

        self.cv_scores = {}
        for metric_name, scoring in scoring_metrics.items():
            scores = cross_val_score(
                self.model,
                X_train,
                y_train,
                cv=skf,
                scoring=scoring,
                n_jobs=-1
            )
            self.cv_scores[metric_name] = scores

            print(f"       {metric_name:20s}: "
                  f"Mean={scores.mean():.4f} ± Std={scores.std():.4f} "
                  f"| Folds: {np.round(scores, 4)}")

        print(f"\n[INFO] Cross-validation complete.")
        print(f"       Interpretation:")
        print(f"       - Low std across folds → model is stable (low variance)")
        print(f"       - High mean scores → model fits the data well (low bias)")

        return self.cv_scores

    def get_feature_importance(self, feature_names):

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        importances = self.model.feature_importances_

        # Pair feature names with their importance scores and sort
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

        print("\n[INFO] Feature Importances (Gini Importance / MDI):")
        for name, importance in feature_importance_pairs:
            bar = "█" * int(importance * 50)
            print(f"       {name:25s}: {importance:.4f} {bar}")

        return feature_importance_pairs

    def predict(self, X):

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X):

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def save_model(self, filepath="modules/random_forest_model.pkl"):

        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"[INFO] Model saved to: {filepath}")

    def load_model(self, filepath="modules/random_forest_model.pkl"):

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at '{filepath}'.")

        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"[INFO] Model loaded from: {filepath}")
        return self.model
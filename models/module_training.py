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

Author: Student
Course: Undergraduate AI Coursework
===============================================================================
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os


class BurnoutModelTrainer:
    """
    Handles training of the Random Forest Classifier for burnout prediction.

    This class encapsulates model creation, training, cross-validation,
    and persistence (saving/loading).

    Attributes
    ----------
    model : RandomForestClassifier
        The Random Forest model instance.
    cv_scores : dict
        Dictionary storing cross-validation scores for each metric.
    is_trained : bool
        Flag indicating whether the model has been trained.

    Methods
    -------
    create_model(**kwargs):
        Instantiates RandomForestClassifier with specified hyperparameters.
    train(X_train, y_train):
        Fits the model on training data.
    cross_validate(X_train, y_train, cv_folds):
        Performs stratified K-fold cross-validation.
    get_feature_importance(feature_names):
        Returns feature importance scores from the trained model.
    predict(X):
        Returns class predictions.
    predict_proba(X):
        Returns class probability estimates.
    save_model(filepath):
        Saves the trained model to disk.
    load_model(filepath):
        Loads a previously trained model from disk.
    """

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
        """
        Create a Random Forest Classifier with specified hyperparameters.

        Hyperparameter Choices and Justification:
        ------------------------------------------
        n_estimators=200:
            Number of Decision Trees in the forest. 200 provides a good
            balance between performance and computation. Increasing beyond
            200 yields diminishing returns in accuracy improvement.

        max_depth=15:
            Maximum depth of each tree. Limits tree complexity to prevent
            overfitting on the training data. A depth of 15 allows
            sufficiently complex decision boundaries for 9 features.

        min_samples_split=5:
            Minimum samples required to split an internal node. Setting
            this above 2 (default) acts as regularisation, preventing
            splits on very small groups that may represent noise.

        min_samples_leaf=2:
            Minimum samples required in a leaf node. Ensures that each
            leaf represents at least 2 training examples, reducing
            overfitting to individual data points.

        max_features="sqrt":
            Number of features considered for each split = √(total features).
            For 9 features, this means √9 = 3 features per split.
            This decorrelates the trees, which is the key mechanism by
            which Random Forest reduces variance.

        random_state=42:
            Seed for reproducibility. Ensures identical results across runs.

        class_weight="balanced":
            Automatically adjusts weights inversely proportional to class
            frequencies. This helps the model pay equal attention to all
            burnout risk levels, even if class sizes differ slightly.

        Parameters
        ----------
        All parameters are passed to sklearn.ensemble.RandomForestClassifier.

        Returns
        -------
        RandomForestClassifier
            Configured (but not yet trained) model instance.
        """
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
        """
        Train the Random Forest model on the training data.

        The .fit() method:
            1. Creates n_estimators bootstrap samples from (X_train, y_train)
            2. Grows one Decision Tree per bootstrap sample
            3. At each split, considers max_features random features
            4. Stores all trees internally for ensemble prediction

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
        y_train : np.ndarray or pd.Series
            Training target vector of shape (n_samples,).
        """
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
        """
        Perform Stratified K-Fold Cross-Validation.

        ==========================================================================
        CROSS-VALIDATION — THEORETICAL BACKGROUND
        ==========================================================================

        What is K-Fold Cross-Validation?
            K-Fold CV divides the training data into K equal-sized "folds".
            The model is trained K times, each time using K-1 folds for
            training and the remaining 1 fold for validation.

            This produces K performance estimates, which are averaged to
            give a more reliable estimate of model performance than a
            single train-test split.

        Why Stratified K-Fold?
            In stratified K-Fold, each fold preserves the percentage of
            samples for each class. This is crucial for multi-class problems
            like ours (Low/Medium/High) to ensure each fold is representative.

        Why is Cross-Validation Important?
            1. REDUCES EVALUATION VARIANCE: A single split might be "lucky"
               or "unlucky". CV gives K estimates, reducing variability.
            2. USES ALL DATA: Every sample is used for both training and
               validation exactly once (across the K iterations).
            3. DETECTS OVERFITTING: If training accuracy is much higher than
               CV accuracy, the model is overfitting.

        Relationship to Bias-Variance:
            - CV helps ESTIMATE the model's true generalisation error.
            - High variance in CV scores (large std) suggests the model
              is sensitive to the training data composition (high variance).
            - Consistently low CV scores suggest high bias (underfitting).
        ==========================================================================

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray or pd.Series
            Training target vector.
        cv_folds : int, default=5
            Number of cross-validation folds. Standard is 5 or 10.

        Returns
        -------
        dict
            Dictionary with metric names as keys and arrays of K scores
            as values.
        """
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
        """
        Extract and sort feature importances from the trained Random Forest.

        How Feature Importance is Calculated in Random Forest:
            Random Forest uses "Mean Decrease in Impurity" (MDI), also known
            as Gini Importance:
                - For each feature, compute the total reduction in Gini
                  impurity across all splits that use that feature, across
                  ALL trees in the forest.
                - Normalise so all importances sum to 1.0.
                - Higher importance → the feature contributes more to
                  distinguishing between classes.

        Parameters
        ----------
        feature_names : list
            List of feature column names matching the training data.

        Returns
        -------
        list of tuples
            Sorted list of (feature_name, importance_score) tuples,
            from most important to least important.
        """
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
        """
        Make class predictions using the trained Random Forest.

        The prediction for each sample is the MAJORITY VOTE across all
        200 trees in the forest.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get class probability estimates from the trained Random Forest.

        How probabilities are calculated:
            For each sample, each tree in the forest "votes" for a class.
            The probability of class k is the PROPORTION of trees that
            voted for class k.

            Example: If 150 out of 200 trees predict "High", the
            probability for "High" is 150/200 = 0.75.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Probability matrix of shape (n_samples, n_classes).
            Each row sums to 1.0.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def save_model(self, filepath="models/random_forest_model.pkl"):
        """
        Save the trained model to disk using joblib serialisation.

        Parameters
        ----------
        filepath : str
            File path for saving the model.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"[INFO] Model saved to: {filepath}")

    def load_model(self, filepath="models/random_forest_model.pkl"):
        """
        Load a previously trained model from disk.

        Parameters
        ----------
        filepath : str
            File path of the saved model.

        Returns
        -------
        RandomForestClassifier
            The loaded model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model found at '{filepath}'.")

        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"[INFO] Model loaded from: {filepath}")
        return self.model
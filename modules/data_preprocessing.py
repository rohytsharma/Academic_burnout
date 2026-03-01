"""
===============================================================================
DATA PREPROCESSING MODULE
===============================================================================
Purpose:
    Handles all data cleaning, transformation, and preparation steps required
    before training the Random Forest model.

Preprocessing Steps Implemented:
    1. Loading the CSV dataset
    2. Handling missing values using median imputation (robust to outliers)
    3. Encoding the categorical target variable (burnout_risk) using
       LabelEncoder: Low=1, Medium=2, High=0 (alphabetical by default,
       but we control this explicitly)
    4. Feature scaling using StandardScaler (optional for Random Forest,
       but good practice and demonstrates understanding)
    5. Train-test split (80/20 stratified)

AI Syllabus Relevance:
    - Data preprocessing is a fundamental step in any ML pipeline.
    - Garbage In, Garbage Out: model quality depends on data quality.
    - Handling missing data prevents errors and information loss.
    - Encoding converts categorical labels into numeric form for the model.
    - Stratified splitting preserves class distribution in both sets.

Note on Scaling for Random Forest:
    Random Forest is a tree-based ensemble method that makes decisions based
    on feature value thresholds (splits). It is invariant to monotonic
    transformations of features, meaning scaling does NOT affect its
    predictions. However, we include StandardScaler here because:
        (a) It demonstrates understanding of preprocessing pipelines.
        (b) It is good practice if the pipeline were ever extended.
        (c) It does not harm Random Forest performance.

Author: Student
Course: Undergraduate AI Coursework
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class DataPreprocessor:
    """
    A class to encapsulate all data preprocessing operations.

    Attributes
    ----------
    label_encoder : LabelEncoder
        Encoder for converting burnout_risk labels to integers and back.
    scaler : StandardScaler
        Scaler for standardising feature values (zero mean, unit variance).
    feature_names : list
        List of feature column names used for training.
    target_column : str
        Name of the target column in the dataset.

    Methods
    -------
    load_data(filepath):
        Loads CSV data into a pandas DataFrame.
    handle_missing_values(df):
        Fills missing numeric values with column medians.
    encode_target(df):
        Encodes the categorical target column into numeric labels.
    get_features_and_target(df):
        Separates features (X) and target (y) from the DataFrame.
    scale_features(X_train, X_test):
        Fits scaler on training data and transforms both sets.
    split_data(X, y, test_size, random_state):
        Performs stratified train-test split.
    preprocess_pipeline(filepath):
        Executes the full preprocessing pipeline end-to-end.
    save_artifacts(directory):
        Saves encoder and scaler for later use in prediction.
    load_artifacts(directory):
        Loads previously saved encoder and scaler.
    preprocess_single_input(input_dict):
        Preprocesses a single user input for prediction.
    """

    def __init__(self):
        """Initialise the preprocessor with empty encoder and scaler."""
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = [
            "attendance", "study_hours", "sleep_hours",
            "assignment_completion", "gpa", "stress_level",
            "screen_time", "part_time_hours", "social_hours"
        ]
        self.target_column = "burnout_risk"

    def load_data(self, filepath):
        """
        Load dataset from a CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. "
                f"Run 'python data/generate_dataset.py' first."
            )
        df = pd.read_csv(filepath)
        print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def handle_missing_values(self, df):
        """
        Handle missing values using median imputation for numeric columns.

        Why median instead of mean?
            - Median is robust to outliers. If a few students have extreme
              values (e.g., 0% attendance), the median won't be skewed,
              whereas the mean would be pulled toward the extreme.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame potentially containing missing values.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing values filled.
        """
        df_clean = df.copy()

        # Count missing values before imputation
        missing_before = df_clean.isnull().sum()
        total_missing = missing_before.sum()

        if total_missing > 0:
            print(f"[INFO] Found {total_missing} missing values across columns:")
            for col in missing_before[missing_before > 0].index:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"       {col}: {missing_before[col]} missing → "
                      f"filled with median ({median_val:.2f})")
        else:
            print("[INFO] No missing values found.")

        return df_clean

    def encode_target(self, df):
        """
        Encode the categorical target variable into numeric labels.

        The LabelEncoder maps:
            "High"   → 0
            "Low"    → 1
            "Medium" → 2
        (Alphabetical order by default in scikit-learn)

        We fit the encoder here so it can be reused for decoding predictions
        back into human-readable labels.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the categorical target column.

        Returns
        -------
        pd.DataFrame
            DataFrame with encoded target column.
        """
        df_encoded = df.copy()
        df_encoded[self.target_column] = self.label_encoder.fit_transform(
            df_encoded[self.target_column]
        )
        print(f"[INFO] Target classes: "
              f"{dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        return df_encoded

    def get_features_and_target(self, df):
        """
        Separate the DataFrame into feature matrix X and target vector y.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame.

        Returns
        -------
        tuple (pd.DataFrame, pd.Series)
            X: Feature matrix with shape (n_samples, n_features)
            y: Target vector with shape (n_samples,)
        """
        X = df[self.feature_names].copy()
        y = df[self.target_column].copy()
        print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def scale_features(self, X_train, X_test):
        """
        Standardise features using StandardScaler.

        The scaler is fitted ONLY on training data to prevent data leakage.
        The same transformation is then applied to the test data.

        Data Leakage Explanation:
            If we fitted the scaler on the entire dataset (train + test),
            information from the test set would "leak" into the training
            process through the mean and standard deviation calculations.
            This would give an overly optimistic estimate of model performance.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        X_test : pd.DataFrame
            Testing feature matrix.

        Returns
        -------
        tuple (np.ndarray, np.ndarray)
            Scaled training and testing feature matrices.
        """
        # Fit on training data only, then transform both
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("[INFO] Features scaled using StandardScaler (fit on train only).")
        return X_train_scaled, X_test_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets with stratification.

        Stratification ensures that the proportion of each burnout risk
        class (Low, Medium, High) is preserved in both the training and
        testing sets. This is important for imbalanced or multi-class
        datasets to ensure representative evaluation.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
        test_size : float, default=0.2
            Proportion of data reserved for testing (20%).
        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        tuple
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Preserve class distribution
        )
        print(f"[INFO] Train set: {X_train.shape[0]} samples")
        print(f"[INFO] Test set:  {X_test.shape[0]} samples")
        return X_train, X_test, y_train, y_test

    def preprocess_pipeline(self, filepath):
        """
        Execute the complete preprocessing pipeline.

        Pipeline Steps:
            1. Load data from CSV
            2. Handle missing values (median imputation)
            3. Encode target variable
            4. Separate features and target
            5. Split into train/test sets (stratified)
            6. Scale features

        Parameters
        ----------
        filepath : str
            Path to the CSV dataset.

        Returns
        -------
        dict
            Dictionary containing all preprocessed data components:
            - X_train_scaled, X_test_scaled: Scaled feature matrices
            - X_train, X_test: Original (unscaled) feature matrices
            - y_train, y_test: Target vectors
            - feature_names: List of feature column names
            - label_encoder: Fitted LabelEncoder instance
        """
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)

        # Step 1: Load
        df = self.load_data(filepath)

        # Step 2: Handle missing values
        df = self.handle_missing_values(df)

        # Step 3: Encode target
        df = self.encode_target(df)

        # Step 4: Separate features and target
        X, y = self.get_features_and_target(df)

        # Step 5: Train-test split
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Step 6: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        print("=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)

        return {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder
        }

    def save_artifacts(self, directory="modules"):
        """
        Save the fitted LabelEncoder and StandardScaler for reuse
        during prediction (e.g., in the Streamlit app).

        Parameters
        ----------
        directory : str
            Directory to save artifacts.
        """
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.label_encoder, os.path.join(directory, "label_encoder.pkl"))
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))
        joblib.dump(self.feature_names, os.path.join(directory, "feature_names.pkl"))
        print(f"[INFO] Preprocessing artifacts saved to '{directory}/'")

    def load_artifacts(self, directory="modules"):
        """
        Load previously saved LabelEncoder and StandardScaler.

        Parameters
        ----------
        directory : str
            Directory containing saved artifacts.
        """
        self.label_encoder = joblib.load(os.path.join(directory, "label_encoder.pkl"))
        self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(directory, "feature_names.pkl"))

    def preprocess_single_input(self, input_dict):
        """
        Preprocess a single user input (from Streamlit UI) for prediction.

        This method applies the same scaling transformation that was fitted
        on the training data, ensuring consistency between training and
        inference.

        Parameters
        ----------
        input_dict : dict
            Dictionary with feature names as keys and user-provided values.
            Example: {"attendance": 85.0, "study_hours": 6.0, ...}

        Returns
        -------
        np.ndarray
            Scaled feature array with shape (1, n_features), ready for
            model.predict() or model.predict_proba().
        """
        # Create DataFrame with single row, ensuring column order matches training
        input_df = pd.DataFrame([input_dict], columns=self.feature_names)

        # Apply the same scaler fitted on training data
        input_scaled = self.scaler.transform(input_df)

        return input_scaled

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class DataPreprocessor:

    def __init__(self):

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = [
            "attendance", "study_hours", "sleep_hours",
            "assignment_completion", "gpa", "stress_level",
            "screen_time", "part_time_hours", "social_hours"
        ]
        self.target_column = "burnout_risk"

    def load_data(self, filepath):

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset not found at '{filepath}'. "
                f"Run 'python data/generate_dataset.py' first."
            )
        df = pd.read_csv(filepath)
        print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def handle_missing_values(self, df):

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

        df_encoded = df.copy()
        df_encoded[self.target_column] = self.label_encoder.fit_transform(
            df_encoded[self.target_column]
        )
        print(f"[INFO] Target classes: "
              f"{dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        return df_encoded

    def get_features_and_target(self, df):

        X = df[self.feature_names].copy()
        y = df[self.target_column].copy()
        print(f"[INFO] Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    def scale_features(self, X_train, X_test):

        # Fit on training data only, then transform both
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("[INFO] Features scaled using StandardScaler (fit on train only).")
        return X_train_scaled, X_test_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):

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

        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.label_encoder, os.path.join(directory, "label_encoder.pkl"))
        joblib.dump(self.scaler, os.path.join(directory, "scaler.pkl"))
        joblib.dump(self.feature_names, os.path.join(directory, "feature_names.pkl"))
        print(f"[INFO] Preprocessing artifacts saved to '{directory}/'")

    def load_artifacts(self, directory="modules"):

        self.label_encoder = joblib.load(os.path.join(directory, "label_encoder.pkl"))
        self.scaler = joblib.load(os.path.join(directory, "scaler.pkl"))
        self.feature_names = joblib.load(os.path.join(directory, "feature_names.pkl"))

    def preprocess_single_input(self, input_dict):

        # Create DataFrame with single row, ensuring column order matches training
        input_df = pd.DataFrame([input_dict], columns=self.feature_names)

        # Apply the same scaler fitted on training data
        input_scaled = self.scaler.transform(input_df)

        return input_scaled
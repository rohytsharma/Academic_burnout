

import numpy as np
import pandas as pd
import os


def generate_synthetic_dataset(n_samples=600, random_state=42):

    np.random.seed(random_state)


    attendance = np.clip(np.random.normal(75, 15, n_samples), 10, 100)

    # Study hours: Daily study hours, typically 2-6 hours
    study_hours = np.clip(np.random.normal(5, 2.5, n_samples), 0, 14)

    # Sleep hours: Healthy range 6-8, but students vary widely
    sleep_hours = np.clip(np.random.normal(6.5, 1.5, n_samples), 3, 10)

    # Assignment completion rate: Most complete 50-90%
    assignment_completion = np.clip(np.random.normal(70, 20, n_samples), 0, 100)

    # GPA: Normally distributed around 2.8
    gpa = np.clip(np.random.normal(2.8, 0.7, n_samples), 0.0, 4.0)

    # Stress level: 1-10 scale, slightly right-skewed (more moderate stress)
    stress_level = np.clip(np.random.normal(5.5, 2.0, n_samples), 1, 10).astype(int)

    # Screen time: Recreational screen hours per day
    screen_time = np.clip(np.random.normal(5, 2.5, n_samples), 1, 16)

    # Part-time work hours per week
    # Many students don't work (0), some work 10-25 hours
    part_time_hours = np.clip(
        np.random.exponential(8, n_samples), 0, 30
    )

    # Social/leisure hours per day
    social_hours = np.clip(np.random.normal(3, 1.5, n_samples), 0, 8)

    # -------------------------------------------------------------------------
    # Step 2: Compute burnout risk score using weighted formula
    # -------------------------------------------------------------------------
    # This score simulates how real burnout accumulates:
    #   HIGH stress, screen_time, part_time_hours → INCREASE burnout
    #   HIGH attendance, study, sleep, assignments, gpa → DECREASE burnout

    burnout_score = (
        -0.02 * attendance            # Higher attendance → less burnout
        - 0.05 * study_hours          # More study → less burnout (engaged)
        - 0.15 * sleep_hours          # More sleep → less burnout
        - 0.02 * assignment_completion  # Completing work → less burnout
        - 0.3  * gpa                  # Higher GPA → less burnout
        + 0.25 * stress_level         # Higher stress → more burnout
        + 0.08 * screen_time          # More screen time → more burnout
        + 0.06 * part_time_hours      # More work → more burnout
        + 0.05 * social_hours         # Excessive socialising → more burnout
    )

    # Add Gaussian noise to simulate real-world unpredictability
    noise = np.random.normal(0, 0.5, n_samples)
    burnout_score += noise

    # -------------------------------------------------------------------------
    # Step 3: Discretise burnout score into risk categories
    # -------------------------------------------------------------------------
    # Use percentile-based thresholds to ensure balanced-ish class distribution
    low_threshold = np.percentile(burnout_score, 33)
    high_threshold = np.percentile(burnout_score, 66)

    burnout_risk = []
    for score in burnout_score:
        if score <= low_threshold:
            burnout_risk.append("Low")
        elif score <= high_threshold:
            burnout_risk.append("Medium")
        else:
            burnout_risk.append("High")

    # -------------------------------------------------------------------------
    # Step 4: Assemble into DataFrame
    # -------------------------------------------------------------------------
    df = pd.DataFrame({
        "attendance": np.round(attendance, 1),
        "study_hours": np.round(study_hours, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "assignment_completion": np.round(assignment_completion, 1),
        "gpa": np.round(gpa, 2),
        "stress_level": stress_level,
        "screen_time": np.round(screen_time, 1),
        "part_time_hours": np.round(part_time_hours, 1),
        "social_hours": np.round(social_hours, 1),
        "burnout_risk": burnout_risk
    })

    # -------------------------------------------------------------------------
    # Step 5: Inject realistic missing values (~3% of data)
    # -------------------------------------------------------------------------
    # Real datasets always have missing values. We simulate this so the
    # preprocessing pipeline can demonstrate imputation techniques.
    n_missing = int(0.03 * n_samples)
    missing_cols = ["attendance", "sleep_hours", "gpa", "screen_time"]

    for col in missing_cols:
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan

    return df


def save_dataset(output_path="data/student_burnout_dataset.csv"):
    """
    Generate and save the synthetic dataset to a CSV file.

    Parameters
    ----------
    output_path : str
        File path where the CSV will be saved.
    """
    df = generate_synthetic_dataset(n_samples=600, random_state=42)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset generated and saved to: {output_path}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Class distribution:\n{df['burnout_risk'].value_counts()}")
    print(f"[INFO] Missing values:\n{df.isnull().sum()}")

    return df


# Allow running this file directly to generate the dataset
if __name__ == "__main__":
    save_dataset()
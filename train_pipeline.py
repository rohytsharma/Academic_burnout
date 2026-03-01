
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_dataset import save_dataset
from modules.data_preprocessing import DataPreprocessor
from modules.model_training import BurnoutModelTrainer
from modules.evaluation import ModelEvaluator


def run_training_pipeline():
    """
    Execute the complete training pipeline from data generation to model saving.
    """
    print("\n" + "=" * 70)
    print("   AI-BASED ACADEMIC BURNOUT RISK PREDICTION SYSTEM")
    print("   Training Pipeline — Random Forest Classifier")
    print("=" * 70 + "\n")

    dataset_path = "data/student_burnout_dataset.csv"

    # =========================================================================
    # STEP 1: Generate Dataset (if not exists)
    # =========================================================================
    if not os.path.exists(dataset_path):
        print("STEP 1: Generating synthetic dataset...\n")
        save_dataset(dataset_path)
    else:
        print(f"STEP 1: Dataset already exists at '{dataset_path}'. Skipping generation.\n")

    # =========================================================================
    # STEP 2: Preprocess Data
    # =========================================================================
    print("\nSTEP 2: Preprocessing data...\n")
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_pipeline(dataset_path)

    # Save preprocessing artifacts (encoder, scaler) for Streamlit
    preprocessor.save_artifacts("modules")

    # =========================================================================
    # STEP 3: Create Random Forest Model
    # =========================================================================
    print("\nSTEP 3: Creating Random Forest model...\n")
    trainer = BurnoutModelTrainer()
    trainer.create_model(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    )

    # =========================================================================
    # STEP 4: Cross-Validation (BEFORE final training for unbiased estimate)
    # =========================================================================
    print("\nSTEP 4: Cross-validation on training data...\n")
    cv_scores = trainer.cross_validate(
        data["X_train_scaled"],
        data["y_train"],
        cv_folds=5
    )

    # =========================================================================
    # STEP 5: Train on Full Training Set
    # =========================================================================
    print("\nSTEP 5: Training on full training set...\n")
    trainer.train(data["X_train_scaled"], data["y_train"])

    # =========================================================================
    # STEP 6: Evaluate on Test Set
    # =========================================================================
    print("\nSTEP 6: Evaluating on test set...\n")
    y_pred = trainer.predict(data["X_test_scaled"])

    evaluator = ModelEvaluator(
        y_test=data["y_test"],
        y_pred=y_pred,
        class_names=list(data["label_encoder"].classes_)
    )

    # Compute all metrics
    metrics = evaluator.compute_metrics()

    # Generate confusion matrix
    evaluator.generate_confusion_matrix()
    evaluator.plot_confusion_matrix("modules/confusion_matrix.png")

    # Print detailed report
    evaluator.print_classification_report()

    # Feature importance
    feature_importances = trainer.get_feature_importance(data["feature_names"])
    evaluator.plot_feature_importance(feature_importances, "modules/feature_importance.png")

    # =========================================================================
    # STEP 7: Save Model
    # =========================================================================
    print("\nSTEP 7: Saving trained model...\n")
    trainer.save_model("modules/random_forest_model.pkl")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("   TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"   Model saved to:           modules/random_forest_model.pkl")
    print(f"   Scaler saved to:          modules/scaler.pkl")
    print(f"   Encoder saved to:         modules/label_encoder.pkl")
    print(f"   Confusion matrix plot:    modules/confusion_matrix.png")
    print(f"   Feature importance plot:  modules/feature_importance.png")
    print(f"\n   To launch the web app:")
    print(f"   streamlit run streamlit_app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_training_pipeline()
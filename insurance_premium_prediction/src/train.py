# insurance_premium_prediction/src/train.py

import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Common utils
from common_utils.feature_engineering import add_engineered_features
from common_utils.preprocessing import build_preprocessing_pipeline
from common_utils.model_factory import get_regression_models
from common_utils.model_params import PARAM_GRIDS
from common_utils.tuning import tune_model
from common_utils.metrics import regression_metrics, create_leaderboard


# =========================
# CONFIGURATION
# =========================

TARGET = "charges"

NUM_FEATURES = [
    "age",
    "bmi",
    "children",
    "family_size",
    "is_high_risk"
]

CAT_FEATURES = [
    "gender",
    "smoker",
    "region",
    "occupation",
    "coverage_level",
    "medical_history",
    "family_medical_history",
    "exercise_frequency",
    "bmi_category",
    "age_bucket"
]

DATA_DIR = (
    PROJECT_ROOT /
    "insurance_premium_prediction" /
    "data" /
    "processed"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "insurance_premium_prediction" /
    "models"
)

REPORT_DIR = (
    PROJECT_ROOT /
    "insurance_premium_prediction" /
    "reports"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD DATA
# =========================

def load_data():
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    return train_df, val_df


# =========================
# TRAINING PIPELINE
# =========================

def train():
    print("Loading data...")
    train_df, val_df = load_data()

    print("Applying feature engineering...")
    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET]

    print("Building preprocessing pipeline...")
    preprocessor = build_preprocessing_pipeline(
        numerical_features=NUM_FEATURES,
        categorical_features=CAT_FEATURES
    )

    print("Fitting preprocessing pipeline...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print("Saving preprocessing pipeline...")
    joblib.dump(
        preprocessor,
        MODEL_DIR / "preprocessing_pipeline.pkl"
    )

    models = get_regression_models()
    results = []

    best_model = None
    best_rmse = float("inf")
    best_model_name = None

    print("Starting model training...")

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        # Hyperparameter tuning if applicable
        if model_name in PARAM_GRIDS:
            model, best_params, best_score = tune_model(
                model,
                PARAM_GRIDS[model_name],
                X_train_processed,
                y_train
            )
            rmse = -best_score
        else:
            model.fit(X_train_processed, y_train)
            preds = model.predict(X_val_processed)
            rmse, _, _ = regression_metrics(y_val, preds)

        preds = model.predict(X_val_processed)
        rmse, mae, r2 = regression_metrics(y_val, preds)

        results.append({
            "model": model_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })

        print(f"{model_name} | RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = model_name

    print(f"\nBest model: {best_model_name} (RMSE: {best_rmse:.2f})")

    # Save best model
    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.pkl"
    )

    # Save evaluation report
    leaderboard = create_leaderboard(results)
    leaderboard.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    metrics = {
        "best_model": best_model_name,
        "best_rmse": best_rmse,
        "models_evaluated": results
    }

    with open(REPORT_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Training completed successfully.")


if __name__ == "__main__":
    train()

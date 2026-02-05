import sqlite3
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

from preprocessing import build_preprocessor


def main():
    # 1. Load data
    conn = sqlite3.connect("../data/regression.db")
    df = pd.read_sql_query("SELECT * FROM Insurance_Prediction", conn)
    conn.close()

    # 2. Split data (as per problem statement)
    train_df = df.iloc[:700_000]
    eval_df  = df.iloc[700_000:900_000]

    target = "charges"

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_eval = eval_df.drop(columns=[target])
    y_eval = eval_df[target]

    # 3. Missing value handling
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            X_train[col] = X_train[col].fillna("Unknown")
            X_eval[col] = X_eval[col].fillna("Unknown")
        else:
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median)
            X_eval[col] = X_eval[col].fillna(median)

    # 4. Preprocessing
    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_eval_processed  = preprocessor.transform(X_eval)

    # 5. Model training
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_processed, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_eval_processed)

    print("R2:", r2_score(y_eval, y_pred))
    print("MAE:", mean_absolute_error(y_eval, y_pred))
    print("RMSE:", np.sqrt(((y_eval - y_pred) ** 2).mean()))

    # 7. Save artifacts
    joblib.dump(model, "../models/insurance_model.pkl")
    joblib.dump(preprocessor, "../models/preprocessor.pkl")

    print("Model and preprocessor saved successfully.")


if __name__ == "__main__":
    main()

import sqlite3
import pandas as pd
import joblib


def main():
    # Load model & preprocessor
    model = joblib.load("../models/insurance_model.pkl")
    preprocessor = joblib.load("../models/preprocessor.pkl")

    # Load production data (remaining records)
    conn = sqlite3.connect("../data/regression.db")
    df = pd.read_sql_query("SELECT * FROM Insurance_Prediction", conn)
    conn.close()

    prod_df = df.iloc[900_000:]
    X_prod = prod_df.drop(columns=["charges"])

    # Handle missing values
    for col in X_prod.columns:
        if X_prod[col].dtype == "object":
            X_prod[col] = X_prod[col].fillna("Unknown")
        else:
            X_prod[col] = X_prod[col].fillna(X_prod[col].median())

    # Preprocess & predict
    X_prod_processed = preprocessor.transform(X_prod)
    predictions = model.predict(X_prod_processed)

    prod_df["predicted_charges"] = predictions

    # Save predictions
    prod_df.to_csv("../insurance_predictions.csv", index=False)

    print("Predictions generated successfully.")


if __name__ == "__main__":
    main()

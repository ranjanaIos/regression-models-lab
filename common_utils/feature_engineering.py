# common_utils/feature_engineering.py

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds domain-specific engineered features for insurance premium prediction.
    This function MUST be used in both training and prediction.
    """

    df = df.copy()

    # BMI Category
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"]
    )

    # Age Bucket
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 25, 40, 55, 100],
        labels=["young", "adult", "middle_aged", "senior"]
    )

    # Family size
    df["family_size"] = df["children"] + 1

    # High risk indicator
    df["is_high_risk"] = (
        (df["smoker"] == "yes") & (df["bmi"] >= 30)
    ).astype(int)

    return df

# common_utils/time_series_utils.py

def create_lag_features(df, target_col, lags=[1, 7, 14]):
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def extract_datetime_features(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df["day"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["weekday"] = df[date_col].dt.weekday
    return df

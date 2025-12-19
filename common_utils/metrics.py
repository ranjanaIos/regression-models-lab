# common_utils/metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_metrics(y_true, y_pred):
    """
    Computes standard regression metrics.
    Returns: RMSE, MAE, R2
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def create_leaderboard(results):
    """
    Creates a sorted leaderboard DataFrame from model results.
    """
    df = pd.DataFrame(results)
    return df.sort_values(by="rmse").reset_index(drop=True)

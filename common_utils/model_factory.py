# common_utils/model_factory.py

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Optional: handle xgboost safely
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False


def get_regression_models(random_state=42):
    """
    Returns a dictionary of regression models.
    All models learned during the course are included.
    """

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(random_state=random_state, max_iter=5000),

        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=random_state
        ),

        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),

        "AdaBoost": AdaBoostRegressor(random_state=random_state),

        "NeuralNetwork": MLPRegressor(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=random_state
        )
    }

    if xgb_available:
        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state
        )

    return models

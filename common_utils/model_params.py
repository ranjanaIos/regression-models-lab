# common_utils/model_params.py

PARAM_GRIDS = {

    "Ridge": {
        "alpha": [0.1, 1.0, 10.0]
    },

    "Lasso": {
        "alpha": [0.01, 0.1, 1.0]
    },

    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },

    "SVR": {
        "C": [1, 10],
        "kernel": ["rbf", "linear"]
    },

    "KNN": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"]
    },

    "AdaBoost": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1]
    },

    "XGBoost": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 6]
    },

    "NeuralNetwork": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "alpha": [0.0001, 0.001]
    }
}

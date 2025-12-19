# common_utils/tuning.py

from sklearn.model_selection import GridSearchCV


def tune_model(
    model,
    param_grid,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=5
):
    """
    Performs GridSearchCV and returns the best model and parameters.
    """

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid.best_score_

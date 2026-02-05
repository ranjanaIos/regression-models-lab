import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


NUM_FEATURES = ['age', 'bmi', 'children']

CAT_FEATURES = [
    'gender', 'smoker', 'region', 'medical_history',
    'family_medical_history', 'exercise_frequency',
    'occupation', 'coverage_level'
]


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUM_FEATURES),
            ('cat', categorical_transformer, CAT_FEATURES)
        ]
    )

    return preprocessor

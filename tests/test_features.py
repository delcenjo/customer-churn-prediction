import numpy as np
import pandas as pd

from churn.features import build_preprocessor


def sample_features():
    return pd.DataFrame(
        {
            "gender": ["Male", "Female", "Male", "Female"],
            "tenure": [1, 10, 20, 30],
            "MonthlyCharges": [20.0, 50.0, 80.0, 100.0],
            "TotalCharges": [20.0, 500.0, 1600.0, 3000.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        }
    )


def test_preprocessor_outputs_dense_array_without_nan():
    X = sample_features()
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == len(X)
    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unknown_categories():
    X = sample_features()
    pre = build_preprocessor(X).fit(X)
    unseen = X.copy()
    unseen.loc[0, "Contract"] = "Weekly"
    assert pre.transform(unseen).shape[0] == len(unseen)

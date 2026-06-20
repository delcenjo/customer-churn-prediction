import pandas as pd

from churn.data import clean


def sample_df():
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003"],
            "gender": ["Male", "Female", "Male"],
            "tenure": [0, 12, 24],
            "MonthlyCharges": [50.0, 70.0, 90.0],
            "TotalCharges": [" ", "840", "2160"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_clean_drops_customer_id():
    assert "customerID" not in clean(sample_df()).columns


def test_clean_casts_total_charges_to_numeric():
    out = clean(sample_df())
    assert out["TotalCharges"].dtype.kind in "fi"
    assert out["TotalCharges"].iloc[0] == 0.0


def test_clean_encodes_target():
    out = clean(sample_df())
    assert out["Churn"].tolist() == [0, 1, 0]

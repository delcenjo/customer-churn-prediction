import pandas as pd

from .config import DATA_PATH, DROP_COLUMNS, TARGET


def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    # TotalCharges arrives as text and is blank for brand-new customers (tenure 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load_dataset(path=DATA_PATH):
    return clean(load_raw(path))

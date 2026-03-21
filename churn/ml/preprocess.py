import pandas as pd


NUMERICAL_COLS = [
    "subscription_age",
    "bill_avg",
    "reamining_contract",
    "service_failure_count",
    "download_avg",
    "upload_avg",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missing values handling
    df["reamining_contract"] = df["reamining_contract"].fillna(0)
    df["download_avg"] = df["download_avg"].fillna(df["download_avg"].median())
    df["upload_avg"] = df["upload_avg"].fillna(df["upload_avg"].median())

    return df


def get_feature_columns(df: pd.DataFrame):
    return df.columns.tolist()

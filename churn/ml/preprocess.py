import pandas as pd
import logging

logger = logging.getLogger(__name__)

NUMERICAL_COLS = [
    "subscription_age",
    "bill_avg",
    "remaining_contract",
    "service_failure_count",
    "download_avg",
    "upload_avg",
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # fill missing
    df["remaining_contract"] = df["remaining_contract"].fillna(0)
    df["download_avg"] = df["download_avg"].fillna(df["download_avg"].median())
    df["upload_avg"] = df["upload_avg"].fillna(df["upload_avg"].median())

    # convert categorical/binary to numeric
    binary_cols = [
        "is_tv_subscriber",
        "is_movie_package_subscriber",
        "download_over_limit",
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    logger.debug(f"DTYPES:\n{df.dtypes}")

    return df


def get_feature_columns(df: pd.DataFrame):
    return df.columns.tolist()

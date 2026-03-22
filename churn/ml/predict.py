import os
import joblib
import pandas as pd
import logging

from churn.ml.preprocess import preprocess

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "training_data", "feature_names.pkl")


# 🔒 кешуємо
_model = None
_feature_names = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def get_feature_names():
    global _feature_names
    if _feature_names is None:
        if not os.path.exists(FEATURES_PATH):
            raise FileNotFoundError(f"Features not found: {FEATURES_PATH}")
        _feature_names = joblib.load(FEATURES_PATH)
    return _feature_names


def predict_churn(data: dict):
    try:
        logger.error(f"INPUT DATA: {data}")

        model = get_model()
        feature_names = get_feature_names()

        # 1. DataFrame
        df = pd.DataFrame([data])
        logger.error(f"RAW DF:\n{df}")

        # 2. preprocess
        df = preprocess(df)
        logger.error(f"AFTER PREPROCESS:\n{df}")

        # 3. align columns
        df = df.reindex(columns=feature_names)
        logger.error(f"AFTER REINDEX:\n{df}")

        # 4. check NaN
        nulls = df.isnull().sum()
        logger.error(f"NULLS:\n{nulls}")

        if nulls.sum() > 0:
            raise ValueError(f"NaN detected in input data:\n{nulls}")

        # 5. predict
        proba = model.predict_proba(df)[0][1]
        pred = model.predict(df)[0]

        risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"

        result = {
            "churn_probability": float(proba),
            "churn_prediction": int(pred),
            "risk_level": risk_level,
        }

        logger.error(f"RESULT: {result}")

        return result

    except Exception as e:
        logger.exception("🔥 PREDICT ERROR")
        raise e

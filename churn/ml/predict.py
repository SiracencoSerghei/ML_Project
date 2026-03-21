import os
import joblib
import pandas as pd

from churn.ml.preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "training_data", "feature_names.pkl")

_feature_names = joblib.load(FEATURES_PATH)

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_churn(data: dict):
    model = get_model()

    df = pd.DataFrame([data])

    # 🔥 ВАЖЛИВО: той самий preprocess що і при training
    df = preprocess(df)

    # 🔥 вирівнюємо колонки
    df = df.reindex(columns=_feature_names)

    proba = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"

    return {
        "churn_probability": float(proba),
        "churn_prediction": int(pred),
        "risk_level": risk_level,
    }

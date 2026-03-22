import os
import joblib
import pandas as pd

from churn.ml.preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "training_data", "feature_names.pkl")

print("FEATURES PATH EXISTS:", os.path.exists(FEATURES_PATH))
print("MODEL PATH EXISTS:", os.path.exists(MODEL_PATH))


_feature_names = None


def get_feature_names():
    global _feature_names
    if _feature_names is None:
        _feature_names = joblib.load(FEATURES_PATH)
    return _feature_names


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
    df = df.reindex(columns=get_feature_names())

    proba = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    risk_level = "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"

    print("INPUT DATA:", data)
    print("DF BEFORE:", df)

    df = preprocess(df)

    print("DF AFTER PREPROCESS:", df)

    df = df.reindex(columns=get_feature_names())

    print("DF FINAL:", df)
    print("SHAPE:", df.shape)
    print("NULLS:", df.isnull().sum())

    return {
        "churn_probability": float(proba),
        "churn_prediction": int(pred),
        "risk_level": risk_level,
    }

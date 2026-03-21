from django.shortcuts import render
from django.core.cache import cache

import os
import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from churn.ml.preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "ml")

MODEL_PATH = os.path.join(ML_DIR, "model", "churn_model.joblib")
DATA_PATH = os.path.join(ML_DIR, "training_data", "internet_service_churn.csv")


def compute_metrics():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)
    df = df.drop("id", axis=1)

    X = df.drop("churn", axis=1)
    y = df["churn"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_prob)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots()
    labels = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
    sns.heatmap(cm, annot=labels, fmt="", ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    cm_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y, y_prob)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax2.plot([0, 1], [0, 1], "k--")
    ax2.legend()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    roc_base64 = base64.b64encode(buf2.getvalue()).decode()
    plt.close(fig2)

    return {
        "accuracy": f"{accuracy:.2%}",
        "precision": f"{precision:.2%}",
        "recall": f"{recall:.2%}",
        "f1_score": f"{f1:.2%}",
        "roc_auc": f"{roc_auc:.2%}",
        "cm_base64": cm_base64,
        "roc_base64": roc_base64,
    }


def model_metrics_view(request):
    cache_key = "model_metrics"

    data = cache.get(cache_key)

    if data is None:
        data = compute_metrics()
        cache.set(cache_key, data, timeout=3600)  # 1 година

    return render(request, "churn/model_metrics.html", data)

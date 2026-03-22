import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from churn.ml.preprocess import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "training_data", "internet_service_churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "training_data", "feature_names.pkl")

# ------------------- Load Data -------------------
df = pd.read_csv(DATA_PATH)

# ------------------- Preprocess -------------------
df = preprocess(df)

df = df.drop("id", axis=1)

X = df.drop("churn", axis=1)
y = df["churn"]

# Save feature names (CRITICAL for inference)
feature_names = [
    "subscription_age",
    "bill_avg",
    "reamining_contract",
    "service_failure_count",
    "download_avg",
    "upload_avg",
    "is_tv_subscriber",
    "is_movie_package_subscriber",
    "download_over_limit",
]

# ------------------- Train/Test Split -------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------- Models -------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
}

# ------------------- Pipelines -------------------
pipelines = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    ),
    "Random Forest": Pipeline(
        [
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight="balanced"
                ),
            )
        ]
    ),
    "XGBoost": Pipeline(
        [
            (
                "model",
                XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
            )
        ]
    ),
}

# ------------------- Training -------------------
best_model = None
best_model_name = None
best_f1 = 0
model_infos = {}

for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...")

    print(X.dtypes)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)

    f1 = report["1"]["f1-score"]

    print(classification_report(y_val, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")

    model_infos[name] = {
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

# ------------------- Save Best Model -------------------
joblib.dump(best_model, MODEL_PATH)

# Save metadata
joblib.dump(
    {"best_model_name": best_model_name, "metrics": model_infos},
    os.path.join(MODEL_DIR, "model_info.pkl"),
)

print("\n✅ Best model:", best_model_name)
print("Validation F1:", best_f1)
print("Model saved to:", MODEL_PATH)

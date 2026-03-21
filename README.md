<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Churn Prediction Project</title>
</head>

<body>

<h1>📊 Telecom Customer Churn Prediction System</h1>

<hr>

<h2>Overview</h2>

<p>
End-to-end machine learning web application for predicting customer churn in a telecommunications environment.
The system allows input of customer attributes and returns churn probability, classification result, and risk level.
</p>

<p>
The project demonstrates integration of data science workflows with a production-style web backend.
</p>

<hr>

<h2>Business Objective</h2>

<ul>
  <li>Identify customers at risk of leaving</li>
  <li>Enable proactive retention strategies</li>
  <li>Support data-driven decision making</li>
</ul>

<hr>

<h2>System Architecture</h2>

<p>
The system follows a modular ML deployment architecture:
</p>

<pre><code>
User Input (Web Form)
        ↓
Django Backend
        ↓
Feature Alignment & Validation
        ↓
ML Pipeline (Preprocessing + Model)
        ↓
Prediction Output
        ↓
Response Rendering (UI)
</code></pre>

<hr>

<h2>Machine Learning Approach</h2>

<ul>
  <li>Multiple models evaluated: Logistic Regression, Random Forest, XGBoost</li>
  <li>Model selection based on validation performance</li>
  <li>Evaluation metrics: ROC-AUC, F1-score, Precision, Recall</li>
</ul>

<p>
Final model is selected automatically based on F1-score and deployed as a unified Pipeline.
</p>

<hr>

<h3>Final Model</h3>

<p>
A scikit-learn Pipeline combining preprocessing and the estimator into a single serialized object.
</p>

<hr>

<h2>Model Artifacts</h2>

<pre><code>
churn/ml/
├── model/
│   ├── churn_model.joblib
│   └── model_info.pkl
├── training_data/
│   ├── feature_names.pkl
│   └── internet_service_churn.csv
</code></pre>

<ul>
  <li><strong>churn_model.joblib</strong> — serialized Pipeline (preprocessing + model)</li>
  <li><strong>feature_names.pkl</strong> — feature schema used for inference</li>
  <li><strong>model_info.pkl</strong> — evaluation metrics and metadata</li>
</ul>

<hr>

<h2>Backend (Django)</h2>

<p>
The backend is responsible for request handling, data validation, model inference, and rendering responses.
</p>

<h3>Core endpoints</h3>

<ul>
  <li><code>/</code> — Project overview</li>
  <li><code>/features/</code> — Feature descriptions</li>
  <li><code>/metrics/</code> — Model evaluation results</li>
  <li><code>/predict/</code> — Churn prediction interface</li>
</ul>

<hr>

<h2>Key Implementation Details</h2>

<ul>
  <li>Lazy-loaded ML model using singleton pattern</li>
  <li>Model preloading via Django AppConfig.ready()</li>
  <li>Consistent feature ordering enforced via feature_names.pkl</li>
  <li>Pipeline ensures identical preprocessing during training and inference</li>
</ul>

<hr>

<h2>Technologies</h2>

<ul>
  <li>Python 3.11</li>
  <li>Django</li>
  <li>scikit-learn</li>
  <li>pandas, numpy</li>
  <li>XGBoost</li>
  <li>matplotlib, seaborn</li>
  <li>Git & Git LFS</li>
</ul>

<hr>

<h2>Highlights</h2>

<ul>
  <li>End-to-end ML lifecycle implementation</li>
  <li>Production-style model packaging using Pipeline</li>
  <li>Separation of concerns between ML and web layers</li>
  <li>Reproducible inference pipeline</li>
</ul>

<hr>

<h2>Outcome</h2>

<p>
The system provides a practical example of deploying a machine learning model into a web application with structured preprocessing, consistent inference, and modular backend design.
</p>

</body>
</html>

## Запуск проєкту локально

1️⃣ Клонування репозиторію
```
git clone https://github.com/SiracencoSerghei/ML_Project.git
cd project-11_GoIT
```

2️⃣ Створення virtual environment
```
python -m venv .venv
source .venv/bin/activate
```

3️⃣ Встановлення залежностей
```
pip install -r requirements.txt
```
### або однією командою
```
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
4️⃣ Запуск сервера
```
python manage.py runserver 8000
```

Відкрий у браузері:
```
http://127.0.0.1:8000/
```

# Встановлення Docker

### Будуємо образ

```
docker-compose build
```

### Запускаємо контейнер

```
docker-compose up
```

### Відкриваємо у браузері

```
http://127.0.0.1:8002/
```

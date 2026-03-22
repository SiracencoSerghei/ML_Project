FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# 🔥 TRAIN HERE
RUN python churn/ml/train.py

CMD ["sh", "-c", "gunicorn churn_project.wsgi:application --bind 0.0.0.0:$PORT --workers 2"]

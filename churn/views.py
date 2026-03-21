from django.shortcuts import render
from .forms import ChurnPredictionForm
from churn.ml.predict import predict_churn


def home_view(request):
    return render(request, "churn/home.html")


def feature_names_view(request):
    feature_names = {
        "is_tv_subscriber": "TV subscription",
        "is_movie_package_subscriber": "Movie package",
        "subscription_age": "Subscription age",
        "bill_avg": "Average bill",
        "reamining_contract": "Remaining contract",
        "service_failure_count": "Service failures",
        "download_avg": "Download avg",
        "upload_avg": "Upload avg",
        "download_over_limit": "Download over limit",
    }
    return render(request, "churn/feature_names.html", {"feature_names": feature_names})


def predict_view(request):
    form = ChurnPredictionForm()
    message = None
    probability = None

    if request.method == "POST":
        form = ChurnPredictionForm(request.POST)

        if form.is_valid():
            result = predict_churn(form.cleaned_data)

            probability = result["churn_probability"]
            prediction = result["churn_prediction"]
            risk = result["risk_level"]

            message = (
                f"⚠️ WILL churn ({risk})"
                if prediction == 1
                else f"✅ Will NOT churn ({risk})"
            )

    return render(
        request,
        "churn/predict.html",
        {"form": form, "message": message, "probability": probability},
    )

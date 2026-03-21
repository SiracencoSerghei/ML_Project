from django.urls import path
from . import views
from .metrics_views import model_metrics_view

urlpatterns = [
    path("", views.home_view, name="home"),
    path("features/", views.feature_names_view, name="feature_names"),
    path("predict/", views.predict_view, name="predict"),
    path("metrics/", model_metrics_view, name="model_metrics"),
]

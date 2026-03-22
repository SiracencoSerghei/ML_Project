import logging
import sklearn
import xgboost
import joblib
import numpy as np
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ChurnConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "churn"

    def ready(self):
        logger.info(f"sklearn: {sklearn.__version__}")
        logger.info(f"xgboost: {xgboost.__version__}")
        logger.info(f"joblib: {joblib.__version__}")
        logger.info(f"numpy: {np.__version__}")

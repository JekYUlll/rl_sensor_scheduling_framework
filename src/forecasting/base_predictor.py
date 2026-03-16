from __future__ import annotations

from core.interfaces import BasePredictor as InterfaceBasePredictor
from evaluation.forecast_metrics import compute_forecast_metrics


class BasePredictor(InterfaceBasePredictor):
    def evaluate(self, test_data) -> dict:
        y_pred = self.predict(test_data)
        return compute_forecast_metrics(test_data.Y, y_pred)

from __future__ import annotations

from typing import Any

import numpy as np

from core.interfaces import BasePredictor as InterfaceBasePredictor
from evaluation.forecast_metrics import compute_forecast_metrics


class BasePredictor(InterfaceBasePredictor):
    def evaluate(self, test_data: Any) -> dict[str, float]:
        y_pred = self.predict(test_data)
        return compute_forecast_metrics(test_data.Y, y_pred)

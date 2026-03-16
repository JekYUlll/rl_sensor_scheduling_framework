from __future__ import annotations

import numpy as np

from forecasting.base_predictor import BasePredictor


class NaivePredictor(BasePredictor):
    def fit(self, train_data, val_data=None) -> None:
        return None

    def predict(self, test_data):
        # Repeat last observed step for all horizons.
        last = test_data.X[:, -1:, :]
        return np.repeat(last, test_data.Y.shape[1], axis=1)

from __future__ import annotations

import numpy as np

from forecasting.base_predictor import BasePredictor


class NaivePredictor(BasePredictor):
    def fit(self, train_data, val_data=None) -> None:
        return None

    def predict(self, test_data) -> np.ndarray:
        # Repeat last observed step for all horizons.
        last = test_data.X[:, -1:, :]
        target_indices = getattr(test_data, "target_indices", None)
        if target_indices is not None:
            last = last[:, :, target_indices]
        return np.repeat(last, test_data.Y.shape[1], axis=1)

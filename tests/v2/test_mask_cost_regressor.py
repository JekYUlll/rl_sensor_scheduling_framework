from __future__ import annotations

import numpy as np
import torch

from v2.mask_cost_regressor import MaskCostRegressor, MaskCostRegressorConfig


def test_mask_cost_regressor_scores_arbitrary_candidate_sets() -> None:
    model = MaskCostRegressor(MaskCostRegressorConfig(context_dim=5, sensor_count=3))
    context = torch.zeros((4, 5), dtype=torch.float32)
    masks = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 1, 0]],
        dtype=torch.float32,
    )
    cost_features = torch.tensor(
        [[0.0, 0.0], [0.4, 0.5], [0.8, 0.9], [0.7, 0.8]],
        dtype=torch.float32,
    )
    scores = model(context, masks, cost_features)
    assert scores.shape == (4, 4)
    assert torch.isfinite(scores).all()


def test_mask_cost_regressor_shares_parameters_across_action_order() -> None:
    torch.manual_seed(4)
    model = MaskCostRegressor(MaskCostRegressorConfig(context_dim=4, sensor_count=2))
    context = torch.randn((3, 4))
    masks = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.float32)
    costs = torch.tensor([[0.0, 0.0], [0.5, 0.6], [0.4, 0.5]], dtype=torch.float32)
    original = model(context, masks, costs)
    order = np.asarray([2, 0, 1])
    reordered = model(context, masks[order], costs[order])
    assert torch.allclose(original[:, order], reordered, atol=1e-6)


def test_monotonic_quality_adjustment_lowers_selected_sensor_cost() -> None:
    torch.manual_seed(7)
    model = MaskCostRegressor(
        MaskCostRegressorConfig(context_dim=4, sensor_count=2, quality_feature_count=2)
    )
    low_quality = torch.tensor([[0.1, 0.8, 0.0, 0.0]], dtype=torch.float32)
    high_quality = torch.tensor([[0.9, 0.8, 0.0, 0.0]], dtype=torch.float32)
    masks = torch.tensor([[1, 0]], dtype=torch.float32)
    costs = torch.tensor([[0.5, 0.6]], dtype=torch.float32)
    assert model(high_quality, masks, costs).item() < model(low_quality, masks, costs).item()
    assert model.quality_scale_raw is not None
    assert model.quality_scale_raw.shape == (2,)

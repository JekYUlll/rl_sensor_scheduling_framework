from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class PhysicsContext:
    input_feature_names: list[str]
    target_feature_names: list[str]
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_mean: torch.Tensor
    y_std: torch.Tensor

    def denormalize_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.x_std + self.x_mean

    def denormalize_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.y_std + self.y_mean


class PhysicsConstraint:
    def __init__(self, cfg: dict, context: PhysicsContext) -> None:
        self.cfg = dict(cfg)
        self.context = context
        self.weight = float(cfg.get("weight", 1.0))

    def compute(self, x_norm: torch.Tensor, y_pred_norm: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NonNegativeTargetConstraint(PhysicsConstraint):
    def __init__(self, cfg: dict, context: PhysicsContext) -> None:
        super().__init__(cfg, context)
        target_name = str(cfg["target"])
        self.target_idx = context.target_feature_names.index(target_name)

    def compute(self, x_norm: torch.Tensor, y_pred_norm: torch.Tensor) -> torch.Tensor:
        y_pred = self.context.denormalize_y(y_pred_norm)
        target = y_pred[:, :, self.target_idx]
        return torch.relu(-target).mean()


class CalmWindFluxConstraint(PhysicsConstraint):
    def __init__(self, cfg: dict, context: PhysicsContext) -> None:
        super().__init__(cfg, context)
        target_name = str(cfg["target"])
        wind_feature = str(cfg.get("wind_feature", "wind_speed_ms"))
        self.target_idx = context.target_feature_names.index(target_name)
        self.wind_idx = context.input_feature_names.index(wind_feature)
        self.wind_threshold = float(cfg.get("wind_threshold", 6.0))
        self.max_flux_when_calm = float(cfg.get("max_flux_when_calm", 1e-5))

    def compute(self, x_norm: torch.Tensor, y_pred_norm: torch.Tensor) -> torch.Tensor:
        x = self.context.denormalize_x(x_norm)
        y_pred = self.context.denormalize_y(y_pred_norm)
        last_wind = x[:, -1, self.wind_idx]
        calm_mask = (last_wind < self.wind_threshold).to(y_pred.dtype).unsqueeze(1)
        target = y_pred[:, :, self.target_idx]
        excess = torch.relu(target - self.max_flux_when_calm)
        return (excess * calm_mask).mean()


class ThresholdTransportConstraint(PhysicsConstraint):
    def __init__(self, cfg: dict, context: PhysicsContext) -> None:
        super().__init__(cfg, context)
        target_name = str(cfg["target"])
        wind_feature = str(cfg.get("wind_feature", "wind_speed_ms"))
        self.target_idx = context.target_feature_names.index(target_name)
        self.wind_idx = context.input_feature_names.index(wind_feature)
        self.wind_threshold = float(cfg.get("wind_threshold", 6.0))
        self.max_flux_when_calm = float(cfg.get("max_flux_when_calm", 1e-5))
        self.min_flux_when_active = float(cfg.get("min_flux_when_active", 2e-5))

    def compute(self, x_norm: torch.Tensor, y_pred_norm: torch.Tensor) -> torch.Tensor:
        x = self.context.denormalize_x(x_norm)
        y_pred = self.context.denormalize_y(y_pred_norm)
        last_wind = x[:, -1, self.wind_idx]
        target = y_pred[:, :, self.target_idx]

        calm_mask = (last_wind < self.wind_threshold).to(y_pred.dtype).unsqueeze(1)
        active_mask = (last_wind >= self.wind_threshold).to(y_pred.dtype).unsqueeze(1)

        calm_excess = torch.relu(target - self.max_flux_when_calm)
        active_deficit = torch.relu(self.min_flux_when_active - target)
        return (calm_excess * calm_mask + active_deficit * active_mask).mean()


@dataclass
class EventWeighting:
    target_idx: int
    threshold: float
    weight: float

    def weights(self, y_true_denorm: torch.Tensor) -> torch.Tensor:
        target = y_true_denorm[:, :, self.target_idx]
        mask = (target >= self.threshold).to(y_true_denorm.dtype)
        base = torch.ones_like(target)
        weighted = base + (float(self.weight) - 1.0) * mask
        return weighted.unsqueeze(-1)


def resolve_event_weighting(
    cfg: dict | None,
    context: PhysicsContext,
    y_train_norm: np.ndarray,
) -> EventWeighting | None:
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    target_name = str(cfg.get("target", context.target_feature_names[0]))
    target_idx = context.target_feature_names.index(target_name)
    mode = str(cfg.get("mode", "quantile"))
    if mode == "absolute":
        threshold = float(cfg["threshold"])
    else:
        quantile = float(cfg.get("quantile", 0.9))
        y_train = context.denormalize_y(torch.as_tensor(y_train_norm, dtype=torch.float32, device=context.y_mean.device))
        target = y_train[:, :, target_idx].detach().cpu().numpy().reshape(-1)
        threshold = float(np.quantile(target, quantile))
    return EventWeighting(
        target_idx=target_idx,
        threshold=threshold,
        weight=float(cfg.get("weight", 1.0)),
    )


_REGISTRY = {
    "nonnegative_target": NonNegativeTargetConstraint,
    "calm_wind_flux": CalmWindFluxConstraint,
    "threshold_transport": ThresholdTransportConstraint,
}


def build_physics_constraints(constraints_cfg: list[dict], context: PhysicsContext) -> list[PhysicsConstraint]:
    constraints: list[PhysicsConstraint] = []
    for item in constraints_cfg:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unsupported physics constraint: {name}")
        constraints.append(cls(item, context))
    return constraints

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from v2.sensor_spec import SensorSpecV2


class SensorMode(IntEnum):
    OFF = 0
    WARMING = 1
    ACTIVE = 2


@dataclass
class SensorRuntime:
    spec: SensorSpecV2
    mode: SensorMode = SensorMode.OFF
    warm_remaining: int = 0
    last_observed_step: int | None = None
    warmup_abort_count: int = 0

    def reset(self) -> None:
        self.mode = SensorMode.OFF
        self.warm_remaining = 0
        self.last_observed_step = None
        self.warmup_abort_count = 0

    def begin_step(self, selected: bool, step_idx: int) -> dict[str, float | int | bool | str]:
        del step_idx
        previous_mode = self.mode
        if not selected:
            if self.mode == SensorMode.WARMING and self.warm_remaining > 0:
                self.warmup_abort_count += 1
            self.mode = SensorMode.OFF
            self.warm_remaining = 0
            return self.status(powered=False, previous_mode=previous_mode)

        if self.mode == SensorMode.OFF:
            if self.spec.warmup_steps > 0:
                self.mode = SensorMode.WARMING
                self.warm_remaining = self.spec.warmup_steps
            else:
                self.mode = SensorMode.ACTIVE
                self.warm_remaining = 0
        elif self.mode == SensorMode.WARMING and self.warm_remaining <= 0:
            self.mode = SensorMode.ACTIVE

        return self.status(powered=True, previous_mode=previous_mode)

    def end_step(self, selected: bool) -> None:
        if selected and self.mode == SensorMode.WARMING and self.warm_remaining > 0:
            self.warm_remaining -= 1
            if self.warm_remaining <= 0:
                self.mode = SensorMode.ACTIVE

    def can_observe(self, step_idx: int) -> bool:
        if self.mode != SensorMode.ACTIVE:
            return False
        interval = max(1, int(self.spec.sampling_interval))
        return step_idx % interval == 0

    def mark_observed(self, step_idx: int) -> None:
        self.last_observed_step = int(step_idx)

    def freshness(self, step_idx: int) -> float:
        if self.last_observed_step is None:
            return 1.0
        return min(1.0, max(0.0, float(step_idx - self.last_observed_step) / 100.0))

    def status(self, *, powered: bool, previous_mode: SensorMode) -> dict[str, float | int | bool | str]:
        steady_power = self.spec.power_cost if powered else 0.0
        peak_power = steady_power
        if powered and previous_mode == SensorMode.OFF:
            peak_power = max(peak_power, self.spec.startup_peak_power)
        return {
            "sensor_id": self.spec.sensor_id,
            "mode": self.mode.name.lower(),
            "mode_id": int(self.mode),
            "powered": bool(powered),
            "ready": self.mode == SensorMode.ACTIVE,
            "warming": self.mode == SensorMode.WARMING,
            "warm_remaining": int(self.warm_remaining),
            "power_cost": float(steady_power),
            "peak_power": float(peak_power),
            "warmup_abort_count": int(self.warmup_abort_count),
        }


from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from v2.oracle import build_supervised_windows, build_supervised_windows_with_context  # noqa: E402
from v2.tcn_oracle import TCNFrozenForecastOracle, TCNOracleConfig  # noqa: E402


def test_tcn_oracle_trains_and_roundtrips(tmp_path: Path) -> None:
    t = np.linspace(0.0, 4.0 * np.pi, 72)
    target = np.stack([np.sin(t), np.cos(t)], axis=1)
    observed = target + 0.01 * np.stack([np.cos(t), np.sin(t)], axis=1)
    mask = np.ones_like(observed)
    x, y = build_supervised_windows(observed, mask, target, lookback=8, horizon=2)

    oracle = TCNFrozenForecastOracle(
        TCNOracleConfig(
            lookback=8,
            horizon=2,
            channels=8,
            levels=1,
            epochs=2,
            batch_size=16,
            learning_rate=1e-3,
            device="cpu",
            seed=3,
            target_weights=(1.0, 2.0),
            target_scales=(1.0, 3.0),
            use_mask_channels=False,
        )
    )
    oracle.fit(x, y)
    loss = oracle.loss(x[4], y[4])

    path = tmp_path / "tcn_oracle.pt"
    oracle.save(path)
    loaded = TCNFrozenForecastOracle.load(path, device="cpu")

    assert np.isfinite(loss)
    assert loaded.is_fitted
    assert loaded.cfg.target_weights == (1.0, 2.0)
    assert loaded.cfg.target_scales == (1.0, 3.0)
    assert loaded.cfg.use_mask_channels is False
    assert np.allclose(oracle.predict(x[0]), loaded.predict(x[0]), atol=1e-5)


def test_tcn_oracle_training_weights_follow_target_weights() -> None:
    oracle = TCNFrozenForecastOracle(
        TCNOracleConfig(
            lookback=3,
            horizon=2,
            target_weights=(1.0, 3.0),
        )
    )

    weights = oracle._flat_training_weights(4)

    assert np.allclose(weights, np.asarray([0.5, 1.5, 0.5, 1.5], dtype=np.float32))
    assert np.isclose(float(np.mean(weights)), 1.0)


def test_supervised_windows_with_context_uses_last_history_step() -> None:
    observed = np.arange(12, dtype=float).reshape(6, 2)
    mask = np.ones_like(observed)
    target = observed.copy()
    context = np.arange(100, 106, dtype=int)

    x, y, contexts = build_supervised_windows_with_context(
        observed,
        mask,
        target,
        context_series=context,
        lookback=2,
        horizon=1,
    )

    assert x.shape[0] == 3
    assert y.shape[0] == 3
    assert np.array_equal(contexts, np.asarray([101, 102, 103]))


def test_tcn_oracle_sample_training_weights_follow_subtype_contexts() -> None:
    oracle = TCNFrozenForecastOracle(
        TCNOracleConfig(
            lookback=3,
            horizon=2,
            target_weights=(1.0, 1.0),
            subtype_loss_weighting=True,
            subtype_particle_target_weights=(3.0, 1.0),
            subtype_flux_target_weights=(1.0, 5.0),
        )
    )

    weights = oracle._sample_training_weights(
        4,
        sample_contexts=np.asarray([0, 1, 2]),
        n_samples=3,
    )

    assert weights.shape == (3, 4)
    assert np.allclose(weights[0], np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
    assert np.allclose(weights[1], np.asarray([1.5, 0.5, 1.5, 0.5], dtype=np.float32))
    assert np.allclose(weights[2], np.asarray([0.5, 2.5, 0.5, 2.5], dtype=np.float32))
    assert np.isclose(float(np.mean(weights)), 1.0)

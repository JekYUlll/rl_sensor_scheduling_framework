from __future__ import annotations

import numpy as np
from pathlib import Path

from v2.oracle import LinearFrozenForecastOracle, OracleConfig, build_supervised_windows


def test_linear_oracle_fits_simple_shifted_series() -> None:
    t = np.arange(80, dtype=float)
    target = np.stack([t, 2.0 * t], axis=1)
    observed = target.copy()
    masks = np.ones_like(observed)
    x, y = build_supervised_windows(observed, masks, target, lookback=5, horizon=2)

    oracle = LinearFrozenForecastOracle(OracleConfig(lookback=5, horizon=2, ridge_alpha=0.01))
    oracle.fit(x, y)
    loss = oracle.loss(x[10], y[10])

    assert loss < 1.0


def test_linear_oracle_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 6))
    y = rng.normal(size=(32, 4))
    oracle = LinearFrozenForecastOracle(
        OracleConfig(
            lookback=2,
            horizon=2,
            ridge_alpha=0.1,
            target_weights=(1.0, 2.0),
            target_scales=(1.0, 3.0),
        )
    )
    oracle.fit(x, y)
    path = tmp_path / "oracle.npz"

    oracle.save(str(path))
    loaded = LinearFrozenForecastOracle.load(str(path))

    pred_a = oracle.predict(x[0])
    pred_b = loaded.predict(x[0])
    assert np.allclose(pred_a, pred_b)
    assert loaded.cfg.target_weights == (1.0, 2.0)
    assert loaded.cfg.target_scales == (1.0, 3.0)


def test_linear_oracle_loss_uses_configured_target_scales() -> None:
    x = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    y = np.asarray([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=float)
    oracle = LinearFrozenForecastOracle(
        OracleConfig(
            lookback=1,
            horizon=1,
            ridge_alpha=0.0,
            target_weights=(1.0, 1.0),
            target_scales=(1.0, 10.0),
        )
    ).fit(x, y)

    loss = oracle.loss(x[2], np.asarray([[3.0, 10.0]], dtype=float))

    assert np.isfinite(loss)
    assert loss < 2.0

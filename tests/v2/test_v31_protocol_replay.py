from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_non_overlapping_random_starts_are_deterministic_and_disjoint() -> None:
    module = _load_script("57_v31_independent_replay.py")
    starts = module.non_overlapping_random_starts(
        sequence_steps=30000,
        eval_steps=1024,
        horizon=8,
        n_rollouts=6,
        seed=11818,
    )
    repeated = module.non_overlapping_random_starts(
        sequence_steps=30000,
        eval_steps=1024,
        horizon=8,
        n_rollouts=6,
        seed=11818,
    )

    assert starts == repeated
    assert len(starts) == 6
    for idx, left in enumerate(starts):
        for right in starts[idx + 1 :]:
            assert left + 1024 <= right or right + 1024 <= left


def test_interval_overlap_pairs_detect_cross_window_overlap() -> None:
    module = _load_script("56_v31_protocol_audit.py")

    pairs = module.interval_overlap_pairs((100, 1000), 200, (250, 1200), 200)

    assert pairs == [(0, 0)]


def test_split_protocol_reserves_non_overlapping_final_test_windows() -> None:
    module = _load_script("58_v31_split_protocol_run.py")
    bounds = module.partition_bounds(90000, (0.35, 0.50, 0.075, 0.075))
    final_starts = module.non_overlapping_starts(
        bounds=bounds["final_test"],
        window_steps=1024,
        horizon=8,
        count=6,
        seed=1818,
    )

    assert bounds["final_test"] == (83250, 90000)
    assert len(final_starts) == 6
    assert min(final_starts) >= bounds["final_test"][0]
    assert max(final_starts) + 1024 + 8 < bounds["final_test"][1]


def test_control_source_files_follow_reward_normalization_mode() -> None:
    module = _load_script("25_v2_train_custom_ppo.py")

    unnormalized = module.control_source_required_files("none")
    staticnorm = module.control_source_required_files("staticnorm_subtype")

    assert "reward_staticnorm_candidates.csv" not in unnormalized
    assert "reward_staticnorm_normalizers.json" not in unnormalized
    assert "reward_staticnorm_candidates.csv" in staticnorm
    assert "reward_staticnorm_normalizers.json" in staticnorm

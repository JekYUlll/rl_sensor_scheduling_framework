from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pipelines.truth_pipeline import build_scheduler_dataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_csv', default='data/generated/windblown_truth.csv')
    parser.add_argument('--base_cfg', default='configs/base.yaml')
    parser.add_argument('--env_cfg', default='configs/env/windblown_case.yaml')
    parser.add_argument('--sensor_cfg', default='configs/sensors/windblown_sensors.yaml')
    parser.add_argument('--estimator_cfg', default='configs/estimator/kalman.yaml')
    parser.add_argument('--scheduler_cfg', required=True)
    parser.add_argument('--run_id', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--out_npz', default=None)
    parser.add_argument('--split', default='final_test')
    args = parser.parse_args()
    out_npz = args.out_npz or f'data/processed/{args.run_id}.npz'
    out = build_scheduler_dataset(truth_csv=args.truth_csv, env_cfg_path=args.env_cfg, sensor_cfg_path=args.sensor_cfg, estimator_cfg_path=args.estimator_cfg, scheduler_cfg_path=args.scheduler_cfg, run_id=args.run_id, out_npz=out_npz, checkpoint=args.checkpoint, split_name=args.split, base_cfg_path=args.base_cfg)
    print(out['out_npz'])
if __name__ == '__main__':
    main()

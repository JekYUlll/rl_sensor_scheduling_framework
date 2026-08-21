#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATE_TAG="${DATE_TAG:-20260621}"
SEEDS_TEXT="${SEEDS_TEXT:-117 118 119 120 121 122}"
SESSION="${SESSION:-mech_ablation_20260621}"
SEED_LABEL="$(printf '%s' "${SEEDS_TEXT}" | tr ' ' '_')"
STATUS_MD="reports/aggregate/mechanism_ablation_pilot_${SEED_LABEL}_status_${DATE_TAG}.md"

mkdir -p reports/aggregate

{
  printf '# Mechanism Ablation Pilot Status\n\n'
  printf -- '- Generated: `%s`\n' "$(date -Is)"
  printf -- '- tmux session: `%s`\n' "${SESSION}"
  printf -- '- Seeds: `%s`\n\n' "${SEEDS_TEXT}"

  printf '## tmux\n\n```text\n'
  tmux ls 2>/dev/null | grep -E "(^| )${SESSION}:" || true
  printf '```\n\n'

  printf '## Recent Processes\n\n```text\n'
  ps -fu "$(whoami)" | grep -E "mechanism_ablation|run_v31_scenebal2|58_v31_split|25_v2_train_custom_ppo|64_v31_eval|70_v31_split|71_v31_behavior|72_v31_collect|73_v31_collect|75_v31_decide|78_v31_collect" | grep -v grep || true
  printf '```\n\n'

  printf '## GPU Snapshot\n\n```text\n'
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
  printf '```\n\n'

  printf '## Aggregate Artifacts\n\n```text\n'
  find reports/aggregate -maxdepth 2 \( -name 'mechanism_ablation_*decision_audit_*.json' -o -path "reports/aggregate/mechanism_ablation_pilot_${SEED_LABEL}_${DATE_TAG}/*" \) -print 2>/dev/null | sort || true
  printf '```\n\n'

  printf '## Decision Summary\n\n```text\n'
  python - <<'PY' 2>/dev/null || true
import json
from pathlib import Path

for path in sorted(Path("reports/aggregate").glob("mechanism_ablation_*_decision_audit_20260621.json")):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"{path}: unreadable: {exc}")
        continue
    gates = data.get("gates") or {}
    margins = data.get("margins") or {}
    print(
        f"{path.name}: decision={data.get('decision')} "
        f"complete={gates.get('complete_seeds')} "
        f"op_step={gates.get('step_operational_gate_count')} "
        f"true_static_macro={gates.get('true_static_macro_gate_count')} "
        f"behavior={gates.get('behavior_gate_count')} "
        f"mean_step_margin={margins.get('operational_step_margin_mean')} "
        f"mean_true_static_step={margins.get('true_static_step_margin_mean')}"
    )
PY
  printf '```\n\n'

  printf '## Log Tail\n\n```text\n'
  for log in $(ls -1t logs/mechanism_ablation_*_${DATE_TAG}.log 2>/dev/null | head -12); do
    printf '\n### %s\n' "${log}"
    tail -20 "${log}" || true
  done
  printf '```\n'
} >"${STATUS_MD}"

cat "${STATUS_MD}"

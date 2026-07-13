#!/usr/bin/env bash
# F4: Train 6 deliberation controllers (Config A vanilla × 3 seeds + Config B weighted × 3 seeds)
# Then rank by composite score and write selection summary.
#
# Usage:
#   bash scripts/train_f4_controllers.sh
#   CUDA_VISIBLE_DEVICES=0 PYTHON=python3 bash scripts/train_f4_controllers.sh
#   SKIP_EXISTING=1 bash scripts/train_f4_controllers.sh   # skip dirs that already have best_controller.pt
#
# Outputs under: deliberation_controller/checkpoints_fixed/
#   A_vanilla_seed{42,43,44}/best_controller.pt
#   A_vanilla_seed{42,43,44}/training_log.json
#   B_weighted_seed{42,43,44}/...
#   f4_selection_summary.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_TRAJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${AGENT_TRAJ_ROOT}"

export PYTHONPATH="${AGENT_TRAJ_ROOT}:${PYTHONPATH:-}"

# ── Config (override via env) ─────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

DATA_PATH="${DATA_PATH:-${AGENT_TRAJ_ROOT}/deliberation_controller/data/hotpotqa_qwen3_full_dataset_v2_rules_fixed.json}"
CKPT_ROOT="${CKPT_ROOT:-${AGENT_TRAJ_ROOT}/deliberation_controller/checkpoints_fixed}"
SEEDS="${SEEDS:-42 43 44}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
PATIENCE="${PATIENCE:-10}"
GATE_THRESHOLD="${GATE_THRESHOLD:-0.5}"

# Set SKIP_EXISTING=1 to skip training when best_controller.pt already exists
SKIP_EXISTING="${SKIP_EXISTING:-1}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_one() {
  local config_name="$1"   # A_vanilla | B_weighted
  local seed="$2"
  local use_weights="$3"   # 0 | 1

  local save_dir="${CKPT_ROOT}/${config_name}_seed${seed}"
  local log_file="${CKPT_ROOT}/${config_name}_seed${seed}_train.log"
  mkdir -p "${save_dir}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${save_dir}/best_controller.pt" && -f "${save_dir}/training_log.json" ]]; then
    log "SKIP (exists): ${config_name} seed=${seed} -> ${save_dir}"
    return 0
  fi

  log "START: ${config_name} seed=${seed} GPU=${GPU} -> ${save_dir}"

  local -a extra_args=()
  if [[ "${use_weights}" == "1" ]]; then
    extra_args+=(--use_class_weights)
  fi

  "${PYTHON}" -m deliberation_controller.train.train_sl \
    --data_path "${DATA_PATH}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --patience "${PATIENCE}" \
    --gate_threshold "${GATE_THRESHOLD}" \
    --seed "${seed}" \
    --save_dir "${save_dir}/" \
    "${extra_args[@]}" \
    > "${log_file}" 2>&1

  if [[ ! -f "${save_dir}/best_controller.pt" ]]; then
    log "ERROR: training failed, no best_controller.pt in ${save_dir}"
    log "  tail log: ${log_file}"
    tail -n 20 "${log_file}" || true
    exit 1
  fi
  log "DONE:  ${config_name} seed=${seed} (log: ${log_file})"
}

# ── Preflight ───────────────────────────────────────────────────────────────
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: dataset not found: ${DATA_PATH}"
  exit 1
fi

mkdir -p "${CKPT_ROOT}"

log "F4 controller training"
log "  AGENT_TRAJ_ROOT=${AGENT_TRAJ_ROOT}"
log "  DATA_PATH=${DATA_PATH}"
log "  CKPT_ROOT=${CKPT_ROOT}"
log "  PYTHON=${PYTHON}"
log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "  SEEDS=${SEEDS}"
log "  SKIP_EXISTING=${SKIP_EXISTING}"

# ── Config A: vanilla (no class weights) ────────────────────────────────────
for seed in ${SEEDS}; do
  run_one "A_vanilla" "${seed}" 0
done

# ── Config B: weighted BCE+CE (paper §4 inverse frequency) ─────────────────
for seed in ${SEEDS}; do
  run_one "B_weighted" "${seed}" 1
done

log "All 6 trainings finished. Running selection..."

# ── F4b: composite score ranking ────────────────────────────────────────────
SUMMARY_JSON="${CKPT_ROOT}/f4_selection_summary.json"

"${PYTHON}" << PY
import json
import os
from pathlib import Path

ckpt_root = Path("${CKPT_ROOT}")
seeds = [int(s) for s in "${SEEDS}".split()]
configs = ["A_vanilla", "B_weighted"]

print("=" * 72)
print("Controller selection (composite score)")
print("=" * 72)
print("Formula: 0.45*val_gate_F1 + 0.35*val_overall_acc + 0.20*val_stop_precision")
print()

results = []
for config in configs:
    for seed in seeds:
        ckpt_dir = ckpt_root / f"{config}_seed{seed}"
        log_path = ckpt_dir / "training_log.json"
        ckpt_path = ckpt_dir / "best_controller.pt"
        if not log_path.is_file():
            print(f"MISSING log: {log_path}")
            continue
        if not ckpt_path.is_file():
            print(f"MISSING ckpt: {ckpt_path}")
            continue
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
        best = log.get("best_epoch_metrics", {})
        gate_f1 = float(best.get("val_gate_f1", 0))
        overall_acc = float(best.get("val_overall_accuracy", 0))
        stop_prec = float(best.get("val_stop_precision", 0))
        composite = 0.45 * gate_f1 + 0.35 * overall_acc + 0.20 * stop_prec
        results.append({
            "config": config,
            "seed": seed,
            "composite": composite,
            "gate_f1": gate_f1,
            "overall_acc": overall_acc,
            "stop_prec": stop_prec,
            "ckpt_path": str(ckpt_path),
            "log_path": str(log_path),
        })

results.sort(key=lambda x: -x["composite"])

print(f"{'Rank':<5}{'Config_seed':<22}{'Composite':<10}{'GateF1':<8}{'OvAcc':<8}{'StopP':<8}")
print("-" * 72)
for i, r in enumerate(results, 1):
    marker = " *" if i <= 2 else ""
    print(
        f"{i:<5}{r['config']+'_seed'+str(r['seed']):<22}"
        f"{r['composite']:.4f}    "
        f"{r['gate_f1']:.3f}    "
        f"{r['overall_acc']:.3f}    "
        f"{r['stop_prec']:.3f}{marker}"
    )

summary = {
    "formula": "0.45*val_gate_f1 + 0.35*val_overall_accuracy + 0.20*val_stop_precision",
    "rankings": results,
    "top2": results[:2] if len(results) >= 2 else results,
}
out_path = ckpt_root / "f4_selection_summary.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print()
print(f"Wrote summary: {out_path}")
if len(results) >= 2:
    print("Top-2 for smoke test:")
    print(f"  1. {results[0]['ckpt_path']}")
    print(f"  2. {results[1]['ckpt_path']}")
elif results:
    print(f"Only {len(results)} result(s); need at least 2 for smoke test pair.")
else:
    print("No valid training_log.json found.")
PY

log "F4 complete. Summary: ${SUMMARY_JSON}"

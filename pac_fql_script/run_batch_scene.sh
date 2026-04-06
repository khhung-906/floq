#!/bin/bash

set -euo pipefail

ENV_NAMES=(
  # "scene-play-singletask-task1-v0"
  # "scene-play-singletask-task2-v0"
  # "scene-play-singletask-task3-v0"
  # "scene-play-singletask-task4-v0"
  "scene-play-singletask-task5-v0"
)
SEEDS=(3 4)

# Defaults (override via CLI)
ALPHA="300"
OFFLINE_STEPS="1000000"
PROJECT_NAME="pac_fql_scene"
CRITIC_LR="1e-4"
ACTOR_LR="5e-4"
NUM_ATOMS="401"
V_MIN="-400.0"
V_MAX="0.0"
HIDDEN_DIMS_LIST=(
  # 80
  # 160
  # 320
  640
)
NUM_LAYERS="2"
ACTOR_LAYER_NORM="True"

# Optional: sbatch overrides (leave empty to use run_pac_fql_single.sh defaults)
SBATCH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./run_batch_scene.sh --alpha A [--offline_steps N] [--project_name PROJECT] [--critic_lr LR] [--actor_lr LR]
                            [--num_atoms N] [--v_min V] [--v_max V]

Example:
  ./run_batch_puzzle_4.sh --alpha 300 --project_name pac-fql-puzzle-4 --offline_steps 1000000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --alpha)
      ALPHA="${2:-}"; shift 2
      ;;
    --offline_steps)
      OFFLINE_STEPS="${2:-}"; shift 2
      ;;
    --project_name)
      PROJECT_NAME="${2:-}"; shift 2
      ;;
    --critic_lr)
      CRITIC_LR="${2:-}"; shift 2
      ;;
    --actor_lr)
      ACTOR_LR="${2:-}"; shift 2
      ;;
    --num_atoms)
      NUM_ATOMS="${2:-}"; shift 2
      ;;
    --v_min)
      V_MIN="${2:-}"; shift 2
      ;;
    --v_max)
      V_MAX="${2:-}"; shift 2
      ;;
    *)
      echo "[run_batch] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

job_idx=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for env in "${ENV_NAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for hidden_dim in "${HIDDEN_DIMS_LIST[@]}"; do
      name="pacfql_$(printf "%04d" "${job_idx}")_h${hidden_dim}"
      echo "[run_batch] sbatch --job-name ${name} run_pac_fql_single.sh --project_name ${PROJECT_NAME} --env_name ${env} --seed ${seed} --alpha ${ALPHA} --hidden_dim ${hidden_dim} --offline_steps ${OFFLINE_STEPS} --num_atoms ${NUM_ATOMS} --v_min ${V_MIN} --v_max ${V_MAX}"
      sbatch --job-name "${name}" "${SBATCH_ARGS[@]}" "${SCRIPT_DIR}/run_pac_fql_single.sh" \
        --project_name "${PROJECT_NAME}" \
        --env_name "${env}" \
        --seed "${seed}" \
        --alpha "${ALPHA}" \
        --hidden_dim "${hidden_dim}" \
        --num_layers "${NUM_LAYERS}" \
        --actor_layer_norm "${ACTOR_LAYER_NORM}" \
        --critic_lr "${CRITIC_LR}" \
        --actor_lr "${ACTOR_LR}" \
        --num_atoms "${NUM_ATOMS}" \
        --v_min "${V_MIN}" \
        --v_max "${V_MAX}" \
        --offline_steps "${OFFLINE_STEPS}"
      job_idx=$((job_idx + 1))
      sleep 0.5
    done
  done
done

echo "[run_batch] submitted ${job_idx} job(s)"


#!/bin/bash

set -euo pipefail

# Offline-to-online: 1M offline + 2M online (README settings). Per-env alpha / block_depth / noise_coverage.

ENV_NAMES=(
  # "cube-double-play-singletask-task2-v0"
  # "puzzle-3x3-play-singletask-task4-v0"
  # "puzzle-4x4-play-singletask-task4-v0"
  # "cube-triple-play-singletask-task1-v0"
  "scene-play-singletask-task2-v0"
  "antmaze-large-navigate-singletask-task1-v0"
)

declare -A ALPHA_BY_ENV=(
  ["cube-double-play-singletask-task2-v0"]=300
  ["puzzle-3x3-play-singletask-task4-v0"]=1000
  ["puzzle-4x4-play-singletask-task4-v0"]=1000
  ["cube-triple-play-singletask-task1-v0"]=300
  ["scene-play-singletask-task2-v0"]=300
  ["antmaze-large-navigate-singletask-task1-v0"]=10
)

declare -A NOISE_COVERAGE_BY_ENV=(
  ["puzzle-4x4-play-singletask-task4-v0"]=0.25
)

SEEDS=(1 2 3)
OFFLINE_STEPS=1000000
ONLINE_STEPS=1000000

BLOCK_WIDTH=2944
BLOCK_DEPTH=4
VALUE_HIDDEN_DIMS="(320, 320, 320, 320)"
PROJECT_NAME="o2o_floq"

# Optional: sbatch overrides (leave empty to use run_temp_single.sh defaults)
SBATCH_ARGS=()

for env in "${ENV_NAMES[@]}"; do
  if [[ -z "${ALPHA_BY_ENV[$env]:-}" ]]; then
    echo "[run_batch] error: no alpha configured for env=${env}" >&2
    exit 2
  fi
done

job_idx=0
for env in "${ENV_NAMES[@]}"; do
  alpha="${ALPHA_BY_ENV[$env]}"
  for seed in "${SEEDS[@]}"; do
    name="floq_o2o_$(printf "%04d" "${job_idx}")"

    EXTRA_ARGS=()
    if [[ -n "${NOISE_COVERAGE_BY_ENV[$env]:-}" ]]; then
      EXTRA_ARGS+=(--noise_coverage "${NOISE_COVERAGE_BY_ENV[$env]}")
    fi

    echo "[run_batch] sbatch --job-name ${name} --env_name ${env} --seed ${seed} --alpha ${alpha} --block_width ${BLOCK_WIDTH} --block_depth ${BLOCK_DEPTH} --project_name ${PROJECT_NAME} --offline_steps ${OFFLINE_STEPS} --online_steps ${ONLINE_STEPS} ${EXTRA_ARGS[*]:-}"
    sbatch --job-name "${name}" "${SBATCH_ARGS[@]}" ./floq_script/run_temp_single.sh \
      --env_name "${env}" \
      --seed "${seed}" \
      --alpha "${alpha}" \
      --block_width "${BLOCK_WIDTH}" \
      --block_depth "${BLOCK_DEPTH}" \
      --value_hidden_dims "${VALUE_HIDDEN_DIMS}" \
      --offline_steps "${OFFLINE_STEPS}" \
      --online_steps "${ONLINE_STEPS}" \
      --project_name "${PROJECT_NAME}" \
      "${EXTRA_ARGS[@]}"
    job_idx=$((job_idx + 1))
    sleep 0.5
  done
done

echo "[run_batch] submitted ${job_idx} job(s)"

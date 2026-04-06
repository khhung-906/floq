#!/bin/bash

set -euo pipefail

# Offline-to-online: 1M offline + 1M online, 5 seeds, smallest value_hidden_dims.
# One alpha per environment (README-style); edit ALPHA_BY_ENV to tune.

ENV_NAMES=(
  # "cube-double-play-singletask-task2-v0"
  # "puzzle-3x3-play-singletask-task4-v0"
  # "puzzle-4x4-play-singletask-task4-v0"
  # "cube-triple-play-singletask-task1-v0"
  # "scene-play-singletask-task2-v0"
  # "antmaze-large-navigate-singletask-task1-v0"
)

declare -A ALPHA_BY_ENV=(
  ["cube-double-play-singletask-task2-v0"]=300
  ["puzzle-3x3-play-singletask-task4-v0"]=1000
  ["puzzle-4x4-play-singletask-task4-v0"]=1000
  ["cube-triple-play-singletask-task1-v0"]=300
  ["scene-play-singletask-task2-v0"]=300
  ["antmaze-large-navigate-singletask-task1-v0"]=10
)

SEEDS=(1 2 3 4 5 6)
OFFLINE_STEPS=1000000
ONLINE_STEPS=1000000

VALUE_HIDDEN_DIMS="(2944, 2944, 2944, 2944)"
PROJECT_NAME="o2o_fql"

# Optional: sbatch overrides (leave empty to use run_fql_single.sh defaults)
SBATCH_ARGS=()

for env in "${ENV_NAMES[@]}"; do
  alpha="${ALPHA_BY_ENV[$env]:-}"
  if [[ -z "${alpha}" ]]; then
    echo "[run_batch] error: no alpha configured for env=${env}" >&2
    exit 2
  fi
done

job_idx=0
for env in "${ENV_NAMES[@]}"; do
  alpha="${ALPHA_BY_ENV[$env]}"
  for seed in "${SEEDS[@]}"; do
    name="fql_o2o_$(printf "%04d" "${job_idx}")"
    echo "[run_batch] sbatch --job-name ${name} --env_name ${env} --seed ${seed} --alpha ${alpha} --project_name ${PROJECT_NAME} --offline_steps ${OFFLINE_STEPS} --online_steps ${ONLINE_STEPS}"
    sbatch --job-name "${name}" "${SBATCH_ARGS[@]}" ./fql_script/run_fql_single.sh \
      --env_name "${env}" \
      --seed "${seed}" \
      --alpha "${alpha}" \
      --value_hidden_dims "${VALUE_HIDDEN_DIMS}" \
      --offline_steps "${OFFLINE_STEPS}" \
      --online_steps "${ONLINE_STEPS}" \
      --project_name "${PROJECT_NAME}"
    job_idx=$((job_idx + 1))
    sleep 0.5
  done
done

echo "[run_batch] submitted ${job_idx} job(s)"

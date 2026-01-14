#!/bin/bash

set -euo pipefail

# Only sweep env_name and seed (as lists), keep everything else fixed/default.
ENV_NAMES=(
  # "cube-triple-play-singletask-task1-v0"
  "cube-triple-play-singletask-task2-v0"
  "cube-triple-play-singletask-task3-v0"
  "cube-triple-play-singletask-task4-v0"
  "cube-triple-play-singletask-task5-v0"
)
SEEDS=(1 2 3 4 5)

# agent.alpha to pass into floq_script/run_temp_single.sh
ALPHA="300"

# Optional: sbatch overrides (leave empty to use run_temp_single.sh defaults)
# Example: SBATCH_ARGS=(--time=12:00:00 --mem=64G)
SBATCH_ARGS=()

job_idx=0
for env in "${ENV_NAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name="tempreg_$(printf "%04d" "${job_idx}")"
    echo "[run_batch] sbatch --job-name ${name} --env_name ${env} --seed ${seed} --alpha ${ALPHA}"
    sbatch --job-name "${name}" "${SBATCH_ARGS[@]}" ./floq_script/run_temp_single.sh \
      --env_name "${env}" \
      --seed "${seed}" \
      --alpha "${ALPHA}"
    job_idx=$((job_idx + 1))
    sleep 0.5
  done
done

echo "[run_batch] submitted ${job_idx} job(s)"

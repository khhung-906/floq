#!/bin/bash

set -euo pipefail

# Only sweep env_name and seed (as lists), keep everything else fixed/default.
ENV_NAMES=(
  "cube-double-play-singletask-task1-v0"
  "cube-double-play-singletask-task2-v0"
  "cube-double-play-singletask-task3-v0"
  # "cube-double-play-singletask-task4-v0"
  # "cube-double-play-singletask-task5-v0"
)
SEEDS=(1 2 3)

# agent.alpha to pass into floq_script/run_temp_single.sh
ALPHA="300"

# agent.value_hidden_dims to pass into floq_script/run_temp_single.sh
# Leave empty ("") to use floq_script/run_temp_single.sh / agent defaults.
VALUE_HIDDEN_DIMS="(2944, 2944, 2944, 2944)"

# Optional: sbatch overrides (leave empty to use run_temp_single.sh defaults)
# Example: SBATCH_ARGS=(--time=12:00:00 --mem=64G)
SBATCH_ARGS=()

job_idx=0
for env in "${ENV_NAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name="tempreg_$(printf "%04d" "${job_idx}")"
    echo "[run_batch] sbatch --job-name ${name} --env_name ${env} --seed ${seed} --alpha ${ALPHA} --value_hidden_dims ${VALUE_HIDDEN_DIMS}"
    sbatch --job-name "${name}" "${SBATCH_ARGS[@]}" ./floq_script/run_temp_single.sh \
      --env_name "${env}" \
      --seed "${seed}" \
      --alpha "${ALPHA}" \
      --value_hidden_dims "${VALUE_HIDDEN_DIMS}"
    job_idx=$((job_idx + 1))
    sleep 0.5
  done
done

echo "[run_batch] submitted ${job_idx} job(s)"

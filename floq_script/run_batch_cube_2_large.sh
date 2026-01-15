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

# agent.value_hidden_dims to pass into floq_script/run_temp_single.sh (not a list)
# Use "" to fall back to floq_script/run_temp_single.sh / agent defaults.
VALUE_HIDDEN_DIMS="(512, 512, 512, 512)"

# floq critic vector-field MLP size (not lists)
# Defaults in agents/floq.py are block_width=512, block_depth=4.
BLOCK_WIDTH="2944"
BLOCK_DEPTH="4"

# Optional: sbatch overrides (leave empty to use run_temp_single.sh defaults)
# Example: SBATCH_ARGS=(--time=12:00:00 --mem=64G)
SBATCH_ARGS=()

job_idx=0
for env in "${ENV_NAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    name="tempreg_$(printf "%04d" "${job_idx}")"
    echo "[run_batch] sbatch --job-name ${name} --env_name ${env} --seed ${seed} --alpha ${ALPHA} --value_hidden_dims ${VALUE_HIDDEN_DIMS} --block_width ${BLOCK_WIDTH} --block_depth ${BLOCK_DEPTH}"
    sbatch --job-name "${name}" "${SBATCH_ARGS[@]}" ./floq_script/run_temp_single.sh \
      --env_name "${env}" \
      --seed "${seed}" \
      --alpha "${ALPHA}" \
      --value_hidden_dims "${VALUE_HIDDEN_DIMS}" \
      --block_width "${BLOCK_WIDTH}" \
      --block_depth "${BLOCK_DEPTH}"
    job_idx=$((job_idx + 1))
    sleep 0.5
  done
done

echo "[run_batch] submitted ${job_idx} job(s)"

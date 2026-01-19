#!/bin/bash

set -euo pipefail

# Sweep env_name, seed, and value_hidden_dims (as lists); keep alpha fixed.
ENV_NAMES=(
  "cube-triple-play-singletask-task1-v0"
  # "cube-triple-play-singletask-task2-v0"
  # "cube-triple-play-singletask-task3-v0"
  # "cube-triple-play-singletask-task4-v0"
  # "cube-triple-play-singletask-task5-v0"
)
SEEDS=(1 2 3)

# agent.alpha to pass into fql_script/run_fql_single.sh
ALPHA="300"

# agent.value_hidden_dims to pass into fql_script/run_fql_single.sh
# Use "" to fall back to fql_script/run_fql_single.sh / agent defaults.
VALUE_HIDDEN_DIMS_LIST=(
  "(320, 320, 320, 320)"
  "(512, 512, 512, 512)"
  "(1536, 1536, 1536, 1536)"
  "(2944, 2944, 2944, 2944)"
)

# Optional: sbatch overrides (leave empty to use run_fql_single.sh defaults)
# Example: SBATCH_ARGS=(--time=12:00:00 --mem=64G)
SBATCH_ARGS=()

job_idx=0
for env in "${ENV_NAMES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for value_hidden_dims in "${VALUE_HIDDEN_DIMS_LIST[@]}"; do
      name="fql_$(printf "%04d" "${job_idx}")"
      echo "[run_batch] sbatch --job-name ${name} --env_name ${env} --seed ${seed} --alpha ${ALPHA} --value_hidden_dims ${value_hidden_dims}"
      sbatch --job-name "${name}" "${SBATCH_ARGS[@]}" ./fql_script/run_fql_single.sh \
        --env_name "${env}" \
        --seed "${seed}" \
        --alpha "${ALPHA}" \
        --value_hidden_dims "${value_hidden_dims}"
      job_idx=$((job_idx + 1))
      sleep 0.5
    done
  done
done

echo "[run_batch] submitted ${job_idx} job(s)"


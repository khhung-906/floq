#!/bin/bash

#SBATCH --partition=iris-hi
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=iris
#SBATCH --output=runs/floq_script/%A.out
#SBATCH --error=runs/floq_script/%A.err
#SBATCH --job-name="floq"
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris5,iris6,iris7,iris-hp-z8,iris-hgx-1,iris-hgx-2,iliad1,iliad2,iliad3,iliad4,iliad6,iliad-hgx-1,tiger8,tiger7,tiger6,tiger5,tiger4,tiger3,tiger2,tiger1,pasteur6,pasteur5,pasteur4,pasteur3,pasteur2,pasteur1,oriong9,oriong8,oriong7,oriong6,oriong5,oriong4,oriong3,oriong2,oriong1,next1,next2,next3,napoli113,napoli114,napoli115,napoli116,napoli117,napoli118,macondo3,macondo2,macondo1,jagupard20,jagupard21,jagupard27,jagupard26,jagupard28,deep17,deep18,deep19,deep20,deep21,deep22,deep23,deep24,deep25,deep26,britten1,britten2

set -euo pipefail

# This script intentionally keeps almost everything at the repo defaults.
# Only these are treated as variables:
# - env_name
# - seed
# - agent.alpha
# - agent.value_hidden_dims (optional)
# - agent.block_width (optional)
# - agent.block_depth (optional)
ENV_NAME="cube-double-play-singletask-v0"
SEED="0"
ALPHA="10"
# Default matches the agent config default in agents/floq.py
VALUE_HIDDEN_DIMS="(512, 512, 512, 512)"
# Defaults match the agent config defaults in agents/floq.py
BLOCK_WIDTH="512"
BLOCK_DEPTH="4"

usage() {
  cat <<'EOF'
Usage:
  sbatch floq_script/run_temp_single.sh --env_name ENV --seed SEED --alpha ALPHA [--value_hidden_dims DIMS] [--block_width W] [--block_depth D]

Examples:
  sbatch floq_script/run_temp_single.sh \
    --env_name cube-double-play-singletask-v0 \
    --seed 1 \
    --alpha 300
  sbatch floq_script/run_temp_single.sh \
    --env_name cube-double-play-singletask-v0 \
    --seed 1 \
    --alpha 300 \
    --value_hidden_dims "[512, 512, 512, 512]"

Notes:
  If --value_hidden_dims is omitted, defaults to (512, 512, 512, 512).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --env_name)
      ENV_NAME="${2:-}"; shift 2
      ;;
    --seed)
      SEED="${2:-}"; shift 2
      ;;
    --alpha)
      ALPHA="${2:-}"; shift 2
      ;;
    --value_hidden_dims)
      VALUE_HIDDEN_DIMS="${2:-}"; shift 2
      ;;
    --block_width)
      BLOCK_WIDTH="${2:-}"; shift 2
      ;;
    --block_depth)
      BLOCK_DEPTH="${2:-}"; shift 2
      ;;
    *)
      echo "[run_temp_single] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${ENV_NAME}" ]]; then
  echo "[run_temp_single] error: --env_name is required" >&2
  exit 2
fi
if [[ -z "${SEED}" ]]; then
  echo "[run_temp_single] error: --seed is required" >&2
  exit 2
fi
if [[ -z "${ALPHA}" ]]; then
  echo "[run_temp_single] error: --alpha is required" >&2
  exit 2
fi

echo "[run_temp_single] env_name=${ENV_NAME}"
echo "[run_temp_single] seed=${SEED}"
echo "[run_temp_single] agent=agents/floq.py"
echo "[run_temp_single] agent.alpha=${ALPHA}"
if [[ -n "${VALUE_HIDDEN_DIMS}" ]]; then
  echo "[run_temp_single] agent.value_hidden_dims=${VALUE_HIDDEN_DIMS}"
fi
if [[ -n "${BLOCK_WIDTH}" ]]; then
  echo "[run_temp_single] agent.block_width=${BLOCK_WIDTH}"
fi
if [[ -n "${BLOCK_DEPTH}" ]]; then
  echo "[run_temp_single] agent.block_depth=${BLOCK_DEPTH}"
fi

PY_ARGS=(
  python main.py
  --env_name="${ENV_NAME}"
  --seed="${SEED}"
  --agent="agents/floq.py"
  --agent.alpha="${ALPHA}"
  --save_dir=runs/floq_test
  --project_name=floq
)

if [[ -n "${VALUE_HIDDEN_DIMS}" ]]; then
  PY_ARGS+=(--agent.value_hidden_dims="${VALUE_HIDDEN_DIMS}")
fi
if [[ -n "${BLOCK_WIDTH}" ]]; then
  PY_ARGS+=(--agent.block_width="${BLOCK_WIDTH}")
fi
if [[ -n "${BLOCK_DEPTH}" ]]; then
  PY_ARGS+=(--agent.block_depth="${BLOCK_DEPTH}")
fi

"${PY_ARGS[@]}"

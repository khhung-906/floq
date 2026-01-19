#!/bin/bash

#SBATCH --partition=iliad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=iliad
#SBATCH --output=runs/fql_script/%A.out
#SBATCH --error=runs/fql_script/%A.err
#SBATCH --job-name="fql"
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris5,iris6,iris7,iris-hp-z8,iris-hgx-1,iris-hgx-2,iliad1,iliad2,iliad3,iliad4,iliad6,iliad-hgx-1,tiger8,tiger7,tiger6,tiger5,tiger4,tiger3,tiger2,tiger1,pasteur6,pasteur5,pasteur4,pasteur3,pasteur2,pasteur1,oriong9,oriong8,oriong7,oriong6,oriong5,oriong4,oriong3,oriong2,oriong1,next1,next2,next3,napoli113,napoli114,napoli115,napoli116,napoli117,napoli118,macondo3,macondo2,macondo1,jagupard20,jagupard21,jagupard27,jagupard26,jagupard28,deep17,deep18,deep19,deep20,deep21,deep22,deep23,deep24,deep25,deep26,britten1,britten2

set -euo pipefail

# This script keeps everything at defaults except these:
# - env_name
# - seed
# - agent.alpha
# - agent.value_hidden_dims
ENV_NAME="cube-triple-play-singletask-task1-v0"
SEED="0"
ALPHA="10"
# Default matches the agent config default in agents/fql_iqn.py
VALUE_HIDDEN_DIMS="(512, 512, 512, 512)"

usage() {
  cat <<'EOF'
Usage:
  sbatch fql_script/run_fql_single.sh --env_name ENV --seed SEED --alpha ALPHA [--value_hidden_dims DIMS]

Examples:
  sbatch fql_script/run_fql_single.sh \
    --env_name cube-triple-play-singletask-task1-v0 \
    --seed 1 \
    --alpha 300
  sbatch fql_script/run_fql_single.sh \
    --env_name cube-triple-play-singletask-task1-v0 \
    --seed 1 \
    --alpha 300 \
    --value_hidden_dims "(512, 512, 512, 512)"

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
    *)
      echo "[run_fql_single] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${ENV_NAME}" ]]; then
  echo "[run_fql_single] error: --env_name is required" >&2
  exit 2
fi
if [[ -z "${SEED}" ]]; then
  echo "[run_fql_single] error: --seed is required" >&2
  exit 2
fi
if [[ -z "${ALPHA}" ]]; then
  echo "[run_fql_single] error: --alpha is required" >&2
  exit 2
fi

echo "[run_fql_single] env_name=${ENV_NAME}"
echo "[run_fql_single] seed=${SEED}"
echo "[run_fql_single] agent=agents/fql_iqn.py"
echo "[run_fql_single] agent.alpha=${ALPHA}"
if [[ -n "${VALUE_HIDDEN_DIMS}" ]]; then
  echo "[run_fql_single] agent.value_hidden_dims=${VALUE_HIDDEN_DIMS}"
fi

PY_ARGS=(
  python main.py
  --env_name="${ENV_NAME}"
  --seed="${SEED}"
  --agent="agents/fql_iqn.py"
  --agent.alpha="${ALPHA}"
  --save_dir=runs/fql_test
  --project_name=fql
)

if [[ -n "${VALUE_HIDDEN_DIMS}" ]]; then
  PY_ARGS+=(--agent.value_hidden_dims="${VALUE_HIDDEN_DIMS}")
fi

"${PY_ARGS[@]}"


#!/bin/bash

#SBATCH --partition=iris-hi
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --account=iris
#SBATCH --output=runs/pac_fql_env/%A.out
#SBATCH --error=runs/pac_fql_env/%A.err
#SBATCH --job-name="pac_fql_env"
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris5,iris6,iris7,iris8,iris9,iris-hp-z8,iliad1,iliad2,iliad3,iliad4,tiger8,tiger7,tiger6,tiger5,tiger4,tiger3,tiger2,tiger1,pasteur6,pasteur5,pasteur4,pasteur3,pasteur2,pasteur1,oriong9,oriong8,oriong7,oriong6,oriong5,oriong4,oriong3,oriong2,oriong1,next1,next2,next3,napoli113,napoli114,napoli115,napoli116,napoli117,napoli118,macondo3,macondo2,macondo1,jagupard20,jagupard21,jagupard27,jagupard26,jagupard28,deep17,deep18,deep19,deep20,deep21,deep22,deep23,deep24,deep25,deep26,britten1,britten2

set -euo pipefail

ENV_NAME="cube-double-play-singletask-task2-v0"
SEED="0"
PROJECT_NAME="pac_fql_cube_double"
OFFLINE_STEPS="1000000"
CRITIC_LR="1e-4"
ACTOR_LR="1e-4"
ALPHA="300.0"
HIDDEN_DIM="128"
NUM_LAYERS="2"
ACTOR_LAYER_NORM="False"
NUM_ATOMS="201"
V_MIN="-200.0"
V_MAX="0.0"

usage() {
  cat <<'EOF'
Usage:
  sbatch run_pac_fql_single.sh --env_name ENV --seed SEED [--alpha A] [--hidden_dim H]
                               [--num_layers L] [--actor_layer_norm BOOL]
                               [--critic_lr LR] [--actor_lr LR] [--project_name PROJECT]
                               [--offline_steps N]
                               [--num_atoms N] [--v_min V] [--v_max V]

Example:
  sbatch run_pac_fql_single.sh \
    --env_name cube-double-play-singletask-task2-v0 \
    --seed 1 \
    --alpha 300 \
    --hidden_dim 128 \
    --num_layers 2 \
    --project_name pac-fql-cube
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
    --project_name)
      PROJECT_NAME="${2:-}"; shift 2
      ;;
    --offline_steps)
      OFFLINE_STEPS="${2:-}"; shift 2
      ;;
    --critic_lr)
      CRITIC_LR="${2:-}"; shift 2
      ;;
    --actor_lr)
      ACTOR_LR="${2:-}"; shift 2
      ;;
    --alpha)
      ALPHA="${2:-}"; shift 2
      ;;
    --hidden_dim)
      HIDDEN_DIM="${2:-}"; shift 2
      ;;
    --num_layers)
      NUM_LAYERS="${2:-}"; shift 2
      ;;
    --actor_layer_norm)
      ACTOR_LAYER_NORM="${2:-}"; shift 2
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
      echo "[run_pac_fql_single] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

echo "[run_pac_fql_single] env_name=${ENV_NAME}"
echo "[run_pac_fql_single] seed=${SEED}"
echo "[run_pac_fql_single] project_name=${PROJECT_NAME}"
echo "[run_pac_fql_single] offline_steps=${OFFLINE_STEPS}"
echo "[run_pac_fql_single] critic_lr=${CRITIC_LR}"
echo "[run_pac_fql_single] actor_lr=${ACTOR_LR}"
echo "[run_pac_fql_single] alpha=${ALPHA}"
echo "[run_pac_fql_single] hidden_dim=${HIDDEN_DIM}"
echo "[run_pac_fql_single] num_layers=${NUM_LAYERS}"
NUM_HEADS=$((HIDDEN_DIM / 32))
echo "[run_pac_fql_single] num_heads=${NUM_HEADS}"
echo "[run_pac_fql_single] num_atoms=${NUM_ATOMS}"
echo "[run_pac_fql_single] v_min=${V_MIN}"
echo "[run_pac_fql_single] v_max=${V_MAX}"
echo "[run_pac_fql_single] actor_hidden_dims=(512, 512, 512, 512)"
echo "[run_pac_fql_single] actor_layer_norm=${ACTOR_LAYER_NORM}"

python main.py \
  --env_name="${ENV_NAME}" \
  --seed="${SEED}" \
  --agent=agents/tql_pac_fql_actor.py \
  --save_dir="runs/pac_fql_env" \
  --project_name="${PROJECT_NAME}" \
  --offline_steps="${OFFLINE_STEPS}" \
  --agent.critic_lr="${CRITIC_LR}" \
  --agent.actor_lr="${ACTOR_LR}" \
  --agent.alpha="${ALPHA}" \
  --agent.hidden_dim="${HIDDEN_DIM}" \
  --agent.num_layers="${NUM_LAYERS}" \
  --agent.num_heads="${NUM_HEADS}" \
  --agent.train_steps="${OFFLINE_STEPS}" \
  --agent.num_atoms="${NUM_ATOMS}" \
  --agent.v_min="${V_MIN}" \
  --agent.v_max="${V_MAX}" \
  --agent.actor_hidden_dims="(512, 512, 512, 512)" \
  --agent.actor_layer_norm="${ACTOR_LAYER_NORM}"


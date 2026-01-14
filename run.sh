#!/bin/bash

#SBATCH --partition=iliad
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --account=iliad
#SBATCH --output=runs/tql_test/%A.out
#SBATCH --error=runs/tql_test/%A.err
#SBATCH --job-name="tql_test"
#SBATCH --exclude=iris1,iris2,iris3,iris4,iris5,iris6,iris8,iris-hp-z8,iris-hgx-1,iris-hgx-2,iliad1,iliad2,iliad3,iliad4,iliad-hgx-1,tiger8,tiger7,tiger6,tiger5,tiger4,tiger3,tiger2,tiger1,pasteur6,pasteur5,pasteur4,pasteur3,pasteur2,pasteur1,oriong9,oriong8,oriong7,oriong6,oriong5,oriong4,oriong3,oriong2,oriong1,next1,next2,next3,napoli113,napoli114,napoli115,napoli116,napoli117,napoli118,macondo3,macondo2,macondo1,jagupard20,jagupard21,jagupard27,jagupard26,jagupard28,deep17,deep18,deep19,deep20,deep21,deep22,deep23,deep24,deep25,deep26,britten1,britten2

# python -u main.py \
#     --save_dir=runs/floq_test \
#     --env_name=cube-triple-play-singletask-task1-v0 \
#     --seed=1 \
#     --agent=agents/floq.py \
#     --agent.alpha=300 \
#     --project_name=floq

# python -u main.py \
#     --save_dir=runs/floq_test \
#     --env_name=cube-triple-play-singletask-task1-v0 \
#     --seed=2 \
#     --agent=agents/floq.py \
#     --agent.alpha=300 \
#     --project_name=floq

# python -u main.py \
#     --save_dir=runs/floq_test \
#     --env_name=cube-triple-play-singletask-task1-v0 \
#     --seed=3 \
#     --agent=agents/floq.py \
#     --agent.alpha=300 \
#     --project_name=floq


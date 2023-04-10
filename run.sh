#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH -e /data/vision/polina/projects/wmh/inr-atlas/zeenchi/code/matrix/results/err.txt
#SBATCH -o /data/vision/polina/projects/wmh/inr-atlas/zeenchi/code/matrix/results/out.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=curcum

python3 main.py
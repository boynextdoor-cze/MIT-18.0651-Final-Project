#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=curcum
#SBATCH -e err.txt
#SBATCH -o out.txt

python3 train.py
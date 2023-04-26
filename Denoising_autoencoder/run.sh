#! /bin/bash
#SBATCH --partition=A6000
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=curcum,clove
#SBATCH -e err.txt
#SBATCH -o out.txt

python3 main.py
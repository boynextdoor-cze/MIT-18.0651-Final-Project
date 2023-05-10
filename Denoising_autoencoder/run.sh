#! /bin/bash
if [ ! -d $NFS/code/matrix/results/$1 ]
then
    mkdir $NFS/code/matrix/results/$1
fi
sbatch <<EOT
#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=curcum,clove
#SBATCH -e $NFS/code/matrix/results/$1/err.txt
#SBATCH -o $NFS/code/matrix/results/$1/out.txt

MKL_THREADING_LAYER=GNU python3 train.py $1
exit()
EOT
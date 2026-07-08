#!/bin/bash
#SBATCH -J torch_PEMS08
#SBATCH -o /bsuscratch/desmondboateng/blis/logs/%x-%j.out
#SBATCH -e /bsuscratch/desmondboateng/blis/logs/%x-%j.err
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 3-00:00:00

set -eo pipefail

cd /bsuscratch/desmondboateng/blis

mkdir -p logs
mkdir -p results/PEMS08

source ~/miniforge3/etc/profile.d/conda.sh
conda activate blis

echo "Starting GPU BLIS W2 Torch classifier run for PEMS08"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not_set}"

python torch_classify_scattering.py \
    --data_dir data \
    --dataset traffic \
    --sub_datasets PEMS08 \
    --task_types HOUR DAY WEEK \
    --wavelet_type W2 \
    --largest_scale 4 \
    --highest_moment 3 \
    --layer_list 1 2 3 \
    --moment_list 1 \
    --models LR MLP XGB \
    --seeds 42 43 44 45 56 \
    --pca_variance 0.99 \
    --epochs 100 \
    --batch_size 512 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --cv_folds 3 \
    --device cuda \
    --results_dir results/PEMS08

echo "Finished GPU BLIS W2 Torch classifier run for PEMS08"
echo "Date: $(date)"
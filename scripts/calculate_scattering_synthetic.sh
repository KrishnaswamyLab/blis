#!/bin/bash

#SBATCH --job-name=scattering_coefs_synthetic
#SBATCH --output=job_outputs/scattering_coefs_synthetic_%A_%a.txt
#SBATCH --array=0-19  # For 20 jobs in total
#SBATCH --mem=16G
#SBATCH --reservation=sumry2023
#SBATCH --time=3:00:00

module load miniconda
conda activate blis 

SCATTERING_TYPES=("blis" "modulus")
SUB_DATASETS=("camel_pm_0" "camel_pm_1" "camel_pm_2" "camel_pm_3" "camel_pm_4" "gaussian_pm_0" "gaussian_pm_1" "gaussian_pm_2" "gaussian_pm_3" "gaussian_pm_4")

SCATTERING_TYPE_INDEX=$(($SLURM_ARRAY_TASK_ID / 10))  # Because there are 10 sub_datasets
SUB_DATASET_INDEX=$(($SLURM_ARRAY_TASK_ID % 10))  # For 10 sub_datasets

python calculate_scattering.py --scattering_type ${SCATTERING_TYPES[$SCATTERING_TYPE_INDEX]} --largest_scale 4 --highest_moment 3 --dataset synthetic --sub_dataset ${SUB_DATASETS[$SUB_DATASET_INDEX]} --wavelet_type W1

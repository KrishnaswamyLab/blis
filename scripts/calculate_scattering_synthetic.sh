#!/bin/bash

#SBATCH --job-name=scattering_coefs_synthetic
#SBATCH --output=job_outputs/scattering_coefs_synthetic_%A_%a.txt
#SBATCH --array=0-3
#SBATCH --mem=16G
#SBATCH --reservation=sumry2023
#SBATCH --time=12:00:00

module load miniconda
conda activate blis 

SCATTERING_TYPES=("blis" "modulus")
SUB_DATASETS=("bimodal_normal" "bimodal_camel")

SCATTERING_TYPE_INDEX=$(($SLURM_ARRAY_TASK_ID / 2))  # Change 4 to 2 as there are 2 SUB_DATASETS
SUB_DATASET_INDEX=$(($SLURM_ARRAY_TASK_ID % 2))  # Change 4 to 2 for the same reason

python calculate_scattering.py --scattering_type ${SCATTERING_TYPES[$SCATTERING_TYPE_INDEX]} --largest_scale 4 --highest_moment 3 --dataset synthetic --sub_dataset ${SUB_DATASETS[$SUB_DATASET_INDEX]}

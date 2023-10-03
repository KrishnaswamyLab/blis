#!/bin/bash

#SBATCH --job-name=scattering_coefs_traffic
#SBATCH --output=job_outputs/scattering_coefs_%A_%a.txt
#SBATCH --array=0-3
#SBATCH --mem=16G
#SBATCH --reservation=sumry2023
#SBATCH --time=12:00:00

module load miniconda
conda activate blis 

SCATTERING_TYPES=("blis")
SUB_DATASETS=("PEMS03" "PEMS04" "PEMS07" "PEMS08")

SCATTERING_TYPE_INDEX=$(($SLURM_ARRAY_TASK_ID / 4))
SUB_DATASET_INDEX=$(($SLURM_ARRAY_TASK_ID % 4))

python calculate_scattering.py --scattering_type ${SCATTERING_TYPES[$SCATTERING_TYPE_INDEX]} --largest_scale 4 --highest_moment 3 --dataset traffic --sub_dataset ${SUB_DATASETS[$SUB_DATASET_INDEX]}

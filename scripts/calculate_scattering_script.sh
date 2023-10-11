#!/bin/bash

#SBATCH --job-name=traffic_scattering_coefs_traffic
#SBATCH --output=job_outputs/traffic_scattering_coefs_%A_%a.txt
#SBATCH --array=0-7 # Since there are now 2 scattering types and 4 datasets, the total combinations are 2x4 = 8, indexed 0-7.
#SBATCH --mem=16G
#SBATCH --reservation=sumry2023
#SBATCH --time=48:00:00

module load miniconda
conda activate blis 

SCATTERING_TYPES=("blis" "modulus") # Added "modulus" to the list
SUB_DATASETS=("PEMS03" "PEMS04" "PEMS07" "PEMS08")

# These calculations are adjusted to account for the additional scattering type
SCATTERING_TYPE_INDEX=$(($SLURM_ARRAY_TASK_ID / ${#SUB_DATASETS[@]}))
SUB_DATASET_INDEX=$(($SLURM_ARRAY_TASK_ID % ${#SUB_DATASETS[@]}))

python calculate_scattering.py --scattering_type ${SCATTERING_TYPES[$SCATTERING_TYPE_INDEX]} --largest_scale 4 --highest_moment 3 --dataset traffic --sub_dataset ${SUB_DATASETS[$SUB_DATASET_INDEX]}

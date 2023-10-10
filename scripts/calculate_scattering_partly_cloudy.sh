#!/bin/bash

#SBATCH --job-name=cloudy_scattering
#SBATCH --output=job_outputs/cloudy_scattering_%A_%a.txt
#SBATCH --array=0-309  # 2 scattering types * 155 datasets = 310 tasks, 0-indexed hence 0-309
#SBATCH --mem=8G
#SBATCH --reservation=sumry2023
#SBATCH --time=2:00:00

module load miniconda
conda activate blis 

SCATTERING_TYPES=("blis" "modulus")

# Generating the SUB_DATASETS array with values 0000 to 0154
SUB_DATASETS=($(seq -f "%04g" 0 154))

SCATTERING_TYPE_INDEX=$(($SLURM_ARRAY_TASK_ID / ${#SUB_DATASETS[@]}))
SUB_DATASET_INDEX=$(($SLURM_ARRAY_TASK_ID % ${#SUB_DATASETS[@]}))

python calculate_scattering.py --scattering_type ${SCATTERING_TYPES[$SCATTERING_TYPE_INDEX]} --largest_scale 4 --highest_moment 3 --dataset partly_cloudy --sub_dataset ${SUB_DATASETS[$SUB_DATASET_INDEX]}

#!/bin/bash

# === 2. List of SBATCH arguements ===
#SBATCH --job-name=pretrained_Resnet_normalized
#SBATCH --nodelist=bgpu-dhl1
#SBATCH --account=blanca-kann
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/pl/active/machinelearning/Solar_forecasting_project/job_outputs/pretrained_Resnet_normalized.%j.out

# === 3. Purge and load needed modules ===
module purge

module load python/3.6.5

# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"

# === 5. Running the program ===
python -u ../script_run_CNN.py


#!/bin/bash

# === 2. List of SBATCH arguements ===
#SBATCH --job-name=penn_state_transformers_qunatile
#SBATCH --nodelist=bgpu-dhl1
#SBATCH --account=blanca-kann
#SBATCH --gres=gpu
#SBATCH --qos=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --output=/pl/active/machinelearning/Solar_forecasting_project/job_outputs/penn_state_qunatile_transformers.%j.out

# === 3. Purge and load needed modules ===
module purge

module load python/3.6.5

# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"

# === 5. Running the program ===
python -u ../script_run_CNN.py


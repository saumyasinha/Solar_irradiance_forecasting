#!/bin/bash

# === 2. List of SBATCH arguements ===
#SBATCH --job-name=test-job
#SBATCH --partition= bgpu-kann1 #sgpu
#SBATCH --time=23:00:00
#SBATCH --output=/pl/active/machinelearning/SolarProject/job_outputs/test-job.%j.out

# === 3. Purge and load needed modules ===
module purge

module load python/3.6.5

# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"

# === 5. Running the program ===
python -u ../script_run_CNN.py


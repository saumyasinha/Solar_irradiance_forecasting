#!/bin/bash
# === 2. List of SBATCH arguements ===
#SBATCH --job-name=tcn_week_ahead_with_attention_no_quantile
#SBATCH --partition=blanca-kann
#SBATCH --account=blanca-kann
#SBATCH --qos=blanca-kann
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/pl/active/machinelearning/Solar_forecasting_project/job_outputs/tcn_week_ahead_with_attention_no_quantile.%j.out
# === 3. Purge and load needed modules ===
module purge
module load python/3.6.5
# === 4. Additional commands needed to run a program ===
echo "Set environment variables or create directories here!"
# === 5. Running the program ===
python -u ../script_run_CNN.py

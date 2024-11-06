#!/bin/bash
#SBATCH --job-name=sample_hyp_tune        # Job name
#SBATCH --output=/dev/null             # Suppress the default SLURM log file
#SBATCH --error=/dev/null              # Suppress the default SLURM error file
#SBATCH --time=2-00:00:00              # Max time for job 
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks=1                     # Number of tasks (CPU cores)
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=800GB                    # Total memory for the job
#SBATCH --partition=open               # Use the 'open' partition
#SBATCH --mail-type=END,FAIL           # Notifications for job done & fail
#SBATCH --mail-user=ybr5070@psu.edu    # Email address for notifications

# Load the Anaconda module
module load anaconda

# Activate the Anaconda environment
source activate /storage/icds/RISE/sw8/anaconda/anaconda3  # Path to the base environment

# Path to your Python script
PYTHON_SCRIPT='/storage/group/klk37/default/homebytes/code/scripts/sample_hyp_tune.py'

# Convert the script to Unix format (dos2unix)
dos2unix $PYTHON_SCRIPT

# Execute the Python script and direct all output to the log file
python $PYTHON_SCRIPT >> /storage/home/ybr5070/group/homebytes/code/scripts/logs/sample.log 2>&1

# Deactivate the environment
conda deactivate
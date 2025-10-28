#!/bin/bash
#SBATCH --job-name=clone_icon645
#SBATCH --output=clone_icon645.out
#SBATCH --error=clone_icon645.err
#SBATCH --partition=inf-train
#SBATCH --time=12:00:00
#SBATCH --account=dslab_jobs

echo "Starting clone_icon645 job on $(hostname)"
bash clone_icon645.sh
echo "✅ clone_icon645.sh completed"

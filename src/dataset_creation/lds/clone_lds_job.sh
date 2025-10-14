#!/bin/bash
#SBATCH --job-name=clone_lds
#SBATCH --output=clone_lds.out
#SBATCH --time=12:00:00
#SBATCH --error=clone_lds.err
#SBATCH --partition=inf-train
#SBATCH --account=dslab
#SBATCH --qos=ymerzouki-dslab

echo "Host: $(hostname)"
bash clone_lds.sh
echo "Job finished!"

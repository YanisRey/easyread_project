#!/bin/bash
#SBATCH --job-name=clone_arasaac
#SBATCH --output=clone_arasaac.out
#SBATCH --error=clone_arasaac.err
#SBATCH --partition=inf-train
#SBATCH --time=12:00:00
#SBATCH --account=dslab
#SBATCH --qos=ymerzouki-dslab

echo "Starting clone_arasaac job on $(hostname)"
bash clone_arasaac.sh
echo "✅ clone_arasaac.sh completed"

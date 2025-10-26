#!/bin/bash
#SBATCH --job-name=train_lora
#SBATCH --account=dslab
#SBATCH --gpus=5060ti:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --error=slurm_logs/slurm_%j.err
# Training configuration script for ARASAAC LoRA training

# Load bashrc
source ~/.bashrc

# Activate conda environment
conda activate diffusers
# Then run training
echo "Starting training..."
python train_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --data_dir="../../data/training_data" \
    --output_dir="../../lora_output" \
    --resolution=256 \
    --train_batch_size=128 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=100 \
    --learning_rate=1e-4 \
    --lora_rank=12 \
    --lora_alpha=16 \
    --mixed_precision="fp16" \
    --seed=42 \
    --logging_steps=50 \
    --save_steps=500 \
    --dataloader_num_workers=2

echo "Training complete!"

"""
LoRA training script for ARASAAC pictogram generation using Stable Diffusion 1.5
"""
import argparse
import os
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import set_seed


class FullDataset(Dataset):
    """Dataset for ARASAAC pictograms with captions."""

    def __init__(self, data_dir, tokenizer, size=512):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.size = size
        self.tokenizer = tokenizer

        # Load metadata: support JSONL and JSON
        jsonl_path = self.data_dir / "metadata.jsonl"
        json_path = self.data_dir / "metadata.json"

        raw = []
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw.append(json.loads(line))
        elif json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, list):
                    raise ValueError("metadata.json must be a list of objects")
                raw = loaded
        else:
            raise FileNotFoundError("No metadata.jsonl or metadata.json found in data_dir")

        # Normalize fields so downstream code always has: file_name, text
        def _as_list(x):
            if x is None:
                return []
            if isinstance(x, list):
                return x
            if isinstance(x, str):
                return [t.strip() for t in x.split("|") if t.strip()]
            return []

        self.data = []
        for row in raw:
            # file name: accept several variants
            fn = row.get("file_name") or row.get("image_file") or row.get("image") or row.get("path")
            if not fn:
                # skip entries without a usable filename
                continue
            # keep only the basename if a path was provided
            fn = os.path.basename(fn)

            # caption text
            text = row.get("text") or row.get("caption")
            if not text:
                # synthesize from title/keywords if missing
                title = row.get("title") or ""
                kws = _as_list(row.get("keywords"))
                kws = [k for k in kws if k and k != title][:7]
                if title and kws:
                    text = f"a pictogram of {title}, {', '.join(kws)}"
                elif title:
                    text = f"a pictogram of {title}"
                elif kws:
                    text = f"a pictogram: {', '.join(kws)}"
                else:
                    text = "a simple pictogram"

            # store normalized copy (preserve original fields too)
            nr = dict(row)
            nr["file_name"] = fn
            nr["text"] = text
            self.data.append(nr)

        if not self.data:
            raise ValueError("No valid samples found after normalizing metadata.")

        print(f"Loaded {len(self.data)} training samples")
		
    def __len__(self): 
        return len(self.data)
		
    def __getitem__(self, idx):
        import numpy as np  # safe local import
        item = self.data[idx]

        # Load and process image
        image_path = self.images_dir / item["file_name"]
        image = Image.open(image_path).convert("RGB")

        # Resize and normalize to [-1, 1]
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        image = image.resize((self.size, self.size), resample=resample)

        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0  # HWC in [-1,1]
        image = image.permute(2, 0, 1)  # HWC -> CHW

        # Tokenize caption
        caption = item["text"]
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {
            "pixel_values": image,
            "input_ids": input_ids,
        }





def collate_fn(examples):
    """Collate function for dataloader."""
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    input_ids = torch.stack([example['input_ids'] for example in examples])

    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids
    }


def train(args):
    """Main training function."""

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Load models
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder"
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet"
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add LoRA layers to UNet
    print("Adding LoRA layers...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none"
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Setup noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    # Create dataset and dataloader
    print("Loading dataset...")
    train_dataset = FullDataset(
        args.data_dir,
        tokenizer,
        size=args.resolution
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    # Calculate training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Setup learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_train_steps,
        eta_min=args.learning_rate * 0.1
    )

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Move models to device
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device)

    # Training loop
    print("Starting training...")
    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process
    )

    for epoch in range(args.num_train_epochs):
        unet.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch['pixel_values'].to(dtype=torch.float32)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample timesteps
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                )
                timesteps = timesteps.long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch['input_ids'])[0]

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                # Backprop
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Log metrics
                if global_step % args.logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    progress_bar.set_postfix(**logs)

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_lora_weights(accelerator, unet, save_path)

            if global_step >= args.max_train_steps:
                break

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final")
        save_lora_weights(accelerator, unet, save_path)

    print("Training complete!")


def save_lora_weights(accelerator, model, save_path):
    """Save LoRA weights."""
    os.makedirs(save_path, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(save_path)

    print(f"Saved LoRA weights to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA on ARASAAC pictograms")

    # Model arguments
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained model to use"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with prepared training data"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training resolution"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha"
    )

    # Training arguments
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides num_train_epochs)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm"
    )

    # Optimizer arguments
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Other arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_output",
        help="Output directory"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )

    return parser.parse_args()


if __name__ == "__main__":
    import numpy as np  # Import here for the dataset class

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

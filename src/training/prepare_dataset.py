"""
Prepare ARASAAC dataset for LoRA training by creating captions and saving images.
"""
import os
from datasets import load_dataset
from pathlib import Path
import json
from PIL import Image

def create_caption(example, style="descriptive"):
    """
    Create caption from dataset fields.

    Args:
        example: Dataset example with title, keywords, categories
        style: "simple", "descriptive", or "template"
    """
    title = example['title']
    keywords = example.get('keywords', '').split('|')

    if style == "simple":
        # Just use the title
        return title

    elif style == "descriptive":
        # Create natural language description
        # Include main concept and key descriptors
        main_keywords = [k for k in keywords[:7] if k != title]
        if main_keywords:
            return f"a pictogram of {title}, {', '.join(main_keywords)}"
        return f"a pictogram of {title}"

    elif style == "template":
        # Consistent template for style learning
        return f"ARASAAC pictogram showing {title}"

    return title

def prepare_training_data(
    data_dir="/work/courses/dslab/team4/easyread_project/data/arasaac",
    output_dir="./training_data",
    caption_style="descriptive",
    max_samples=None
):
    """
    Prepare dataset for training:
    - Load images
    - Create captions
    - Save in format for diffusers training

    Args:
        data_dir: Path to ARASAAC dataset
        output_dir: Where to save prepared data
        caption_style: How to format captions ("simple", "descriptive", "template")
        max_samples: Limit number of samples (for testing)
    """
    print(f"Loading dataset from {data_dir}...")
    dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Preparing {len(dataset)} samples...")

    # Create output directory
    output_path = Path(output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # Prepare metadata for each image
    metadata = []

    for idx, example in enumerate(dataset):
        # Generate caption
        caption = create_caption(example, style=caption_style)

        # Save image
        image_filename = f"{example['id']:06d}.png"
        image_path = images_path / image_filename

        # Convert to RGB if needed (some PNGs might be palette mode)
        image = example['image']
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image.save(image_path)

        # Store metadata
        metadata.append({
            "file_name": image_filename,
            "text": caption,
            "id": example['id'],
            "title": example['title'],
            "keywords": example.get('keywords', ''),
        })

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} images...")

    # Save metadata as JSONL (one JSON per line)
    metadata_path = output_path / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')

    print(f"\nDataset preparation complete!")
    print(f"Images saved to: {images_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Total samples: {len(metadata)}")

    # Print some example captions
    print("\nExample captions:")
    for i in range(min(5, len(metadata))):
        print(f"  {metadata[i]['file_name']}: {metadata[i]['text']}")

    return output_path

if __name__ == "__main__":
    # Test different caption styles
    print("=" * 60)
    print("Testing caption styles with first example...")
    print("=" * 60)

    dataset = load_dataset(
        "imagefolder",
        data_dir="/work/courses/dslab/team4/easyread_project/data/arasaac",
        split="train"
    )

    example = dataset[0]
    print(f"\nOriginal data:")
    print(f"  Title: {example['title']}")
    print(f"  Keywords: {example['keywords']}")

    print(f"\nCaption styles:")
    print(f"  Simple: {create_caption(example, 'simple')}")
    print(f"  Descriptive: {create_caption(example, 'descriptive')}")
    print(f"  Template: {create_caption(example, 'template')}")

    print("\n" + "=" * 60)
    print("Preparing full dataset...")
    print("=" * 60)

    # Prepare the full dataset (use descriptive captions)
    # Change caption_style to "simple" or "template" if preferred
    prepare_training_data(
        caption_style="descriptive",
        max_samples=None  # Set to small number for testing, None for all
    )

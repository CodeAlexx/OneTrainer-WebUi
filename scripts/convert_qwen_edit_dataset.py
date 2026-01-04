#!/usr/bin/env python3
"""
Convert SimpleTuner-style Qwen Edit dataset to OneTrainer format.

SimpleTuner format:
    dataset/
    ├── control/          # Source/control images
    │   ├── image1.jpg
    │   └── image2.jpg
    └── images/           # Target images + captions
        ├── image1.jpg
        ├── image1.txt
        ├── image2.jpg
        └── image2.txt

OneTrainer format:
    dataset/
    ├── image1.jpg              # Target image
    ├── image1-condlabel.jpg    # Control image (source)
    ├── image1.txt              # Caption
    ├── image2.jpg
    ├── image2-condlabel.jpg
    └── image2.txt

Usage:
    python scripts/convert_qwen_edit_dataset.py --input /path/to/simpletuner/dataset --output /path/to/onetrainer/dataset

    # Or download from HuggingFace:
    python scripts/convert_qwen_edit_dataset.py --hf-repo cyburn/qwen_edit_photo_restore_v1.0-dataset --output /path/to/output
"""

import argparse
import os
import shutil
from pathlib import Path


def convert_local_dataset(input_dir: str, output_dir: str):
    """Convert a local SimpleTuner-style dataset to OneTrainer format."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Find control and images directories
    control_dir = None
    images_dir = None

    # Check common patterns
    for subdir in input_path.iterdir():
        if subdir.is_dir():
            if subdir.name.lower() == 'control':
                control_dir = subdir
            elif subdir.name.lower() in ('images', 'target', 'output'):
                images_dir = subdir

    # Also check nested structure (datasets/dataset_0/...)
    if control_dir is None or images_dir is None:
        # Find all control directories, prefer ones with actual image files
        for dataset_dir in input_path.glob('**/control'):
            # Skip .cache directories
            if '.cache' in str(dataset_dir):
                continue
            potential_images_dir = dataset_dir.parent / 'images'
            if potential_images_dir.exists():
                # Check if this control dir actually has images
                has_images = any(dataset_dir.glob('*.jpg')) or any(dataset_dir.glob('*.png'))
                if has_images:
                    control_dir = dataset_dir
                    images_dir = potential_images_dir
                    break

    if control_dir is None or images_dir is None:
        raise ValueError(
            f"Could not find control/ and images/ directories in {input_dir}. "
            f"Expected SimpleTuner format with control/ and images/ subdirectories."
        )

    print(f"Found control dir: {control_dir}")
    print(f"Found images dir: {images_dir}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all control images
    control_images = {}
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        for img in control_dir.glob(ext):
            control_images[img.stem] = img

    print(f"Found {len(control_images)} control images")

    # Process each target image
    converted = 0
    skipped = 0

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        for target_img in images_dir.glob(ext):
            stem = target_img.stem

            # Check if we have a matching control image
            if stem not in control_images:
                print(f"  Skipping {stem}: no matching control image")
                skipped += 1
                continue

            control_img = control_images[stem]

            # Copy target image
            target_out = output_path / f"{stem}{target_img.suffix}"
            shutil.copy2(target_img, target_out)

            # Copy control image with -condlabel suffix
            control_out = output_path / f"{stem}-condlabel{control_img.suffix}"
            shutil.copy2(control_img, control_out)

            # Copy caption if exists
            caption_file = images_dir / f"{stem}.txt"
            if caption_file.exists():
                caption_out = output_path / f"{stem}.txt"
                shutil.copy2(caption_file, caption_out)
            else:
                # Create default caption for photo restoration
                caption_out = output_path / f"{stem}.txt"
                caption_out.write_text("restore this old damaged photo to a clean high quality image")

            converted += 1

    print(f"\nConverted {converted} image pairs")
    if skipped > 0:
        print(f"Skipped {skipped} images (no matching control)")
    print(f"Output directory: {output_path}")

    return converted


def download_and_convert_hf_dataset(repo_id: str, output_dir: str):
    """Download dataset from HuggingFace and convert to OneTrainer format."""
    from huggingface_hub import snapshot_download

    print(f"Downloading dataset from {repo_id}...")

    # Download to a temp location
    cache_dir = Path(output_dir).parent / ".hf_cache"
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=cache_dir / repo_id.replace("/", "_"),
        ignore_patterns=["*.md", "*.json", ".git*"],
    )

    print(f"Downloaded to: {local_dir}")

    # Convert
    return convert_local_dataset(local_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SimpleTuner-style Qwen Edit dataset to OneTrainer format"
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to local SimpleTuner-style dataset"
    )
    parser.add_argument(
        "--hf-repo",
        help="HuggingFace dataset repository ID (e.g., cyburn/qwen_edit_photo_restore_v1.0-dataset)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output OneTrainer-format dataset"
    )

    args = parser.parse_args()

    if args.input:
        convert_local_dataset(args.input, args.output)
    elif args.hf_repo:
        download_and_convert_hf_dataset(args.hf_repo, args.output)
    else:
        parser.error("Must specify either --input or --hf-repo")


if __name__ == "__main__":
    main()

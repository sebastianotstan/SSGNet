#!/usr/bin/env python3
"""
Enhance a class-foldered training set with StyleGAN3-generated images.

- Copies & resizes original images from:  <original_train_dir>/<class>/*
- Adds & resizes synthetic images from:   <synthetic_images_dir>/<class>/*
- Writes enhanced set to:                 <output_train_dir>/<class>/*
- Ensures a target number of images per class, plus an optional extra % synthetic.

Example:
    python enhance_with_gan.py \
        --original_train_dir chestxray/original/train \
        --synthetic_images_dir chestxray/GAN \
        --output_train_dir chestxray/enhanced_50pct/train \
        --target_num_images 3875 \
        --additional_synthetic_percentage 0.5 \
        --resize 256 256 \
        --seed 42
"""

import os
import random
import argparse
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import itertools


# ---------- I/O & Utility ----------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def resize_and_save(image_path: Path, output_path: Path, size: Tuple[int, int]) -> None:
    with Image.open(image_path) as img:
        # normalize to RGB to avoid palette/alpha surprises
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_resized = img.resize(size, Image.Resampling.LANCZOS)
        # unify saved format as PNG for consistency
        output_path = output_path.with_suffix(".png")
        img_resized.save(output_path, format="PNG")


def sample_with_replacement(population: List[Path], k: int) -> List[Path]:
    """If k <= len(population), sample without replacement. Else, repeat as needed."""
    if not population:
        return []
    if k <= len(population):
        return random.sample(population, k)
    # Need more than available -> cycle
    times = k // len(population)
    remainder = k % len(population)
    base = list(itertools.islice(itertools.cycle(population), len(population) * times))
    base += random.sample(population, remainder) if remainder else []
    return base


# ---------- Core Logic ----------

def enhance_class(
    class_label: str,
    original_class_dir: Path,
    synthetic_class_dir: Path,
    output_class_dir: Path,
    resize_dim: Tuple[int, int],
    target_num_images: int,
    additional_synthetic_percentage: float,
) -> None:
    ensure_dir(output_class_dir)

    # 1) Copy + resize originals
    original_images = list_images(original_class_dir)
    for img in original_images:
        out_name = img.stem + ".png"
        resize_and_save(img, output_class_dir / out_name, resize_dim)

    num_original = len(original_images)

    # 2) Decide how many synthetic images to add
    #    - Bring class up to 'target_num_images' (if short)
    #    - Plus an optional extra % of target (always added if available)
    need_to_reach_target = max(0, target_num_images - num_original)
    extra = max(0, int(additional_synthetic_percentage * target_num_images))
    total_needed = need_to_reach_target + extra

    synthetic_images = list_images(synthetic_class_dir)
    to_add = sample_with_replacement(synthetic_images, total_needed)

    # 3) Copy + resize synthetics (underscore prefix to distinguish)
    for idx, img in enumerate(to_add):
        out_name = f"_{idx:06d}_" + img.stem + ".png"
        resize_and_save(img, output_class_dir / out_name, resize_dim)

    print(
        f"[{class_label}] originals={num_original:5d} | "
        f"target={target_num_images:5d} | need={need_to_reach_target:5d} | "
        f"extra={extra:5d} | added_synth={len(to_add):5d} | "
        f"out_dir='{output_class_dir}'"
    )


def detect_classes(original_train_dir: Path) -> List[str]:
    if not original_train_dir.exists():
        raise FileNotFoundError(f"Original train dir not found: {original_train_dir}")
    classes = sorted([p.name for p in original_train_dir.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class subfolders found in {original_train_dir}")
    return classes


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Enhance dataset with StyleGAN3 images (per-class).")
    p.add_argument("--original_train_dir", type=str, required=True,
                   help="Path to the original train folder (contains class subfolders).")
    p.add_argument("--synthetic_images_dir", type=str, required=True,
                   help="Path to the GAN images root (mirrors class subfolders).")
    p.add_argument("--output_train_dir", type=str, required=True,
                   help="Path to write the enhanced train folder.")
    p.add_argument("--target_num_images", type=int, required=True,
                   help="Target number of images per class BEFORE adding 'additional_synthetic_percentage'.")
    p.add_argument("--additional_synthetic_percentage", type=float, default=0.0,
                   help="Extra proportion of synthetic images to add per class based on target (e.g., 0.5 = +50% of target).")
    p.add_argument("--resize", type=int, nargs=2, default=[256, 256],
                   help="Resize width height (default: 256 256).")
    p.add_argument("--classes", type=str, nargs="*", default=None,
                   help="Explicit class names. If omitted, detected from original_train_dir.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    original_train_dir = Path(args.original_train_dir)
    synthetic_images_dir = Path(args.synthetic_images_dir)
    output_train_dir = Path(args.output_train_dir)
    ensure_dir(output_train_dir)

    classes = args.classes if args.classes else detect_classes(original_train_dir)

    print(f"Classes: {classes}")
    print(f"Original:  {original_train_dir}")
    print(f"Synthetic: {synthetic_images_dir}")
    print(f"Output:    {output_train_dir}")
    print(f"Target per class: {args.target_num_images}")
    print(f"Extra % synthetic: {args.additional_synthetic_percentage * 100:.1f}%")
    print(f"Resize: {tuple(args.resize)}\n")

    for cls in classes:
        orig = original_train_dir / cls
        synth = synthetic_images_dir / cls
        outc = output_train_dir / cls
        enhance_class(
            class_label=cls,
            original_class_dir=orig,
            synthetic_class_dir=synth,
            output_class_dir=outc,
            resize_dim=(args.resize[0], args.resize[1]),
            target_num_images=args.target_num_images,
            additional_synthetic_percentage=args.additional_synthetic_percentage,
        )

    print("\n[Done] Training set enhancement complete. All images resized to the same dimension.")


if __name__ == "__main__":
    main()

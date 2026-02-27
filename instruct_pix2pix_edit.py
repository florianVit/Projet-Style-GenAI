"""Instruct-Pix2Pix-style image editing with Stable Diffusion + CLIP text conditioning.

Installation:
  pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install diffusers transformers accelerate safetensors pillow

If you do not have a GPU, install CPU-only PyTorch instead:
  pip install --upgrade torch torchvision
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import torch
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline


@dataclass
class EditConfig:
    model_id: str = "timbrooks/instruct-pix2pix"
    lora_path: str | None = None
    lora_scale: float = 1.0
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    num_inference_steps: int = 30
    seed: int | None = None


def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Pick GPU if available, otherwise CPU. Use float16 on GPU for speed."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def load_pipeline(cfg: EditConfig) -> StableDiffusionInstructPix2PixPipeline:
    """Load the Instruct-Pix2Pix pipeline from Hugging Face."""
    device, dtype = get_device_and_dtype()

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
    )

    if cfg.lora_path:
        pipe.load_lora_weights(cfg.lora_path)
        pipe.fuse_lora(lora_scale=cfg.lora_scale)

    # Optional: reduce memory on CPU/GPU
    pipe.enable_attention_slicing()

    pipe = pipe.to(device)
    return pipe


def preprocess_image(image_path: str, target_size: int = 512) -> Image.Image:
    """Load and resize the input image to a square while preserving aspect ratio."""
    image = Image.open(image_path).convert("RGB")

    # Fit to square by padding, then resize to model size.
    image = ImageOps.pad(image, (target_size, target_size), color=(255, 255, 255))
    return image


def run_inference(
    pipe: StableDiffusionInstructPix2PixPipeline,
    image: Image.Image,
    instruction: str,
    cfg: EditConfig,
) -> Image.Image:
    """Generate an edited image following the text instruction."""
    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(cfg.seed)

    result = pipe(
        prompt=instruction,
        image=image,
        guidance_scale=cfg.guidance_scale,
        image_guidance_scale=cfg.image_guidance_scale,
        num_inference_steps=cfg.num_inference_steps,
        generator=generator,
    )
    return result.images[0]


def save_output(image: Image.Image, output_path: str) -> None:
    """Save the edited image to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


def compare_side_by_side(original: Image.Image, edited: Image.Image) -> Image.Image:
    """Create a simple side-by-side comparison image."""
    width, height = original.size
    canvas = Image.new("RGB", (width * 2, height), color=(255, 255, 255))
    canvas.paste(original, (0, 0))
    canvas.paste(edited, (width, 0))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instruct-Pix2Pix image editor")
    parser.add_argument(
        "--model-id",
        default="timbrooks/instruct-pix2pix",
        help="Base model ID/path (ex: timbrooks/instruct-pix2pix)",
    )
    parser.add_argument(
        "--lora-path",
        default=None,
        help="Chemin vers un dossier/fichier LoRA (.safetensors)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=1.0,
        help="Intensité LoRA (0.0 à 1.5 en général)",
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--instruction", required=True, help="Edit instruction text")
    parser.add_argument("--output", default="outputs/edited.png", help="Output image path")
    parser.add_argument("--compare", default="outputs/compare.png", help="Side-by-side output path")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Text guidance scale")
    parser.add_argument(
        "--image-guidance-scale", type=float, default=1.5, help="Image guidance scale"
    )
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = EditConfig(
        model_id=args.model_id,
        lora_path=args.lora_path,
        lora_scale=args.lora_scale,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    pipe = load_pipeline(cfg)
    original = preprocess_image(args.image)

    edited = run_inference(pipe, original, args.instruction, cfg)
    save_output(edited, args.output)

    comparison = compare_side_by_side(original, edited)
    save_output(comparison, args.compare)

    print(f"Saved edited image to: {args.output}")
    print(f"Saved comparison to: {args.compare}")


if __name__ == "__main__":
    main()

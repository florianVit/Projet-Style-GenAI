from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog

from instruct_pix2pix_edit import (
    EditConfig,
    compare_side_by_side,
    load_pipeline,
    preprocess_image,
    run_inference,
    save_output,
)


STYLE_CONFIG = {
    "anime": {
        "instruction": "convert to anime style",
        "lora_path": "models/instructpix2pix-anime-lora",
    },
    "oil_painting": {
        "instruction": "convert to oil painting style",
        "lora_path": "models/instructpix2pix-oil-lora",
    },
}


def pick_image_file(initial_dir: str = ".") -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="Choisis une image",
        initialdir=initial_dir,
        filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
            ("Tous les fichiers", "*.*"),
        ],
    )

    root.destroy()
    return file_path or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sélectionne une image et applique ton modèle Instruct-Pix2Pix + LoRA"
    )
    parser.add_argument("--style", choices=["anime", "oil_painting"], default=None)
    parser.add_argument("--instruction", default=None, help="Instruction personnalisée")
    parser.add_argument("--image", default=None, help="Chemin image (sinon fenêtre de sélection)")
    parser.add_argument("--lora-path", default=None, help="Override du chemin LoRA")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--image-guidance-scale", type=float, default=1.5)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_style_interactive() -> str | None:
    options = ["anime", "oil_painting"]
    print("\nChoisis un style :")
    for index, style_name in enumerate(options, start=1):
        print(f"  {index}. {style_name}")
    print("  q. Quitter")

    while True:
        choice = input("Ton choix (1/2/q) : ").strip().lower()
        if choice == "q":
            return None
        if choice in {"1", "2"}:
            return options[int(choice) - 1]
        print("Choix invalide. Réessaie.")


def main() -> None:
    args = parse_args()

    style = args.style
    if style is None:
        if sys.stdin is None or not sys.stdin.isatty():
            style = "oil_painting"
            print("Aucun --style fourni et mode non interactif détecté. Style par défaut: oil_painting")
        else:
            style = choose_style_interactive()
            if style is None:
                print("Annulé par l'utilisateur. Fin.")
                return

    if args.image:
        image_path = Path(args.image)
    else:
        selected = pick_image_file(initial_dir=".")
        if selected is None:
            print("Aucune image sélectionnée. Fin.")
            return
        image_path = Path(selected)

    if not image_path.exists():
        print(f"Image introuvable: {image_path}")
        return

    style_cfg = STYLE_CONFIG[style]
    lora_path = Path(args.lora_path) if args.lora_path else Path(style_cfg["lora_path"])

    if not lora_path.exists():
        print(f"LoRA introuvable pour le style '{style}': {lora_path}")
        print("Lance d'abord le fine-tuning de ce style, ou passe --lora-path")
        return

    instruction = args.instruction or style_cfg["instruction"]

    cfg = EditConfig(
        model_id="timbrooks/instruct-pix2pix",
        lora_path=str(lora_path),
        lora_scale=args.lora_scale,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.steps,
        seed=args.seed,
    )

    print("Chargement du pipeline...")
    pipe = load_pipeline(cfg)

    print(f"Traitement de: {image_path}")
    original = preprocess_image(str(image_path), target_size=args.size)
    edited = run_inference(pipe, original, instruction, cfg)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_img = output_dir / f"{style}_{image_path.stem}_{timestamp}.png"
    out_cmp = output_dir / f"compare_{style}_{image_path.stem}_{timestamp}.png"

    save_output(edited, str(out_img))
    comparison = compare_side_by_side(original, edited)
    save_output(comparison, str(out_cmp))

    print(f"Image stylisée: {out_img}")
    print(f"Comparaison: {out_cmp}")


if __name__ == "__main__":
    main()

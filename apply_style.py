"""
Script pour appliquer des styles graphiques à une image
Utilise les filtres de transformation sans nécessiter de fine-tuning

Usage:
  python apply_style.py --image photo.jpg --style anime --output result.jpg
  python apply_style.py --image photo.jpg --style cartoon --compare
"""

import argparse
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Importer la classe StyleTransformer depuis prepare_dataset
try:
    from prepare_dataset import StyleTransformer
except ImportError:
    print("❌ Erreur: impossible d'importer StyleTransformer")
    print("   Assurez-vous que prepare_dataset.py est dans le même dossier")
    sys.exit(1)


def load_and_prepare_image(image_path: str, max_size: int = 1024) -> np.ndarray:
    """Charge l'image et la prépare pour le traitement."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image: {image_path}")
    
    # Redimensionner si trop grand
    height, width = img.shape[:2]
    if height > max_size or width > max_size:
        scale = max_size / max(height, width)
        img = cv2.resize(img, (int(width*scale), int(height*scale)))
    
    return img


def save_image(image_np: np.ndarray, output_path: str):
    """Sauvegarde l'image."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(output_path, image_np)
    print(f"✅ Image sauvegardée: {output_path}")


def create_comparison(original: np.ndarray, styled: np.ndarray) -> np.ndarray:
    """Crée une image de comparaison côte à côte."""
    height = max(original.shape[0], styled.shape[0])
    width = original.shape[1] + styled.shape[1]
    
    # Créer canvas blanc
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Coller les images
    canvas[:original.shape[0], :original.shape[1]] = original
    canvas[:styled.shape[0], original.shape[1]:] = styled
    
    # Ajouter une ligne de séparation
    cv2.line(canvas, (original.shape[1], 0), (original.shape[1], height), (0, 0, 0), 2)
    
    # Ajouter des labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (10, 30), font, 1, (0, 0, 0), 2)
    cv2.putText(canvas, "Styled", (original.shape[1] + 10, 30), font, 1, (0, 0, 0), 2)
    
    return canvas


def list_available_styles():
    """Affiche la liste des styles disponibles."""
    styles = [
        ("anime", "Style animé japonais avec couleurs vives"),
        ("cartoon", "Style cartoon avec contours nets"),
        ("watercolor", "Effet aquarelle douce"),
        ("oil_painting", "Peinture à l'huile texturée"),
        ("sketch", "Croquis noir et blanc"),
        ("pixel_art", "Effet rétro pixelisé"),
        ("comic_book", "Bande dessinée avec contours épais")
    ]
    
    print("\n📋 Styles disponibles:")
    print("=" * 60)
    for style, description in styles:
        print(f"  {style:15} - {description}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Applique un style graphique à une image")
    parser.add_argument("--image", "-i", required=True, 
                       help="Chemin vers l'image d'entrée")
    parser.add_argument("--style", "-s", required=True,
                       choices=["anime", "cartoon", "watercolor", "oil_painting", 
                               "sketch", "pixel_art", "comic_book"],
                       help="Style à appliquer")
    parser.add_argument("--output", "-o", default=None,
                       help="Chemin de sortie (défaut: outputs/<style>_<nom>.jpg)")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Créer une image de comparaison côte à côte")
    parser.add_argument("--list-styles", "-l", action="store_true",
                       help="Afficher la liste des styles disponibles")
    
    args = parser.parse_args()
    
    # Afficher les styles si demandé
    if args.list_styles:
        list_available_styles()
        return
    
    # Vérifier que l'image existe
    if not Path(args.image).exists():
        print(f"❌ Erreur: l'image '{args.image}' n'existe pas")
        return
    
    print(f"\n🎨 Application du style '{args.style}' à {args.image}")
    print("=" * 60 + "\n")
    
    # Charger l'image
    print("📸 Chargement de l'image...")
    original = load_and_prepare_image(args.image)
    
    # Initialiser le transformer
    print(f"🖌️  Application du filtre {args.style}...")
    transformer = StyleTransformer()
    
    # Appliquer le style
    styled = transformer.stylize_image(original, args.style)
    
    # Définir le chemin de sortie
    if args.output is None:
        input_path = Path(args.image)
        output_filename = f"{args.style}_{input_path.stem}.jpg"
        output_path = f"outputs/{output_filename}"
    else:
        output_path = args.output
    
    # Sauvegarder l'image stylisée
    save_image(styled, output_path)
    
    # Créer et sauvegarder la comparaison si demandé
    if args.compare:
        print("📊 Création de l'image de comparaison...")
        comparison = create_comparison(original, styled)
        compare_path = str(Path(output_path).parent / f"compare_{Path(output_path).name}")
        save_image(comparison, compare_path)
    
    print("\n" + "=" * 60)
    print("✨ Terminé!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

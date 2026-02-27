"""
Script de préparation de dataset pour fine-tuning Instruct-Pix2Pix
Convertit images réalistes en multiples styles graphiques.

Installation requise:
  pip install torch torchvision pillow opencv-python tqdm numpy scipy
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from tqdm import tqdm
from scipy.ndimage import convolve


class StyleTransformer:
    """Transforme les images en différents styles graphiques."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"🎨 Utilisation du device: {self.device}")
        
    def cartoon_filter(self, image_cv2) -> np.ndarray:
        """
        Filtre cartoon amélioré avec meilleure qualité.
        Version optimisée avec contours plus nets.
        """
        # Bilateral filter multiple passes pour meilleur lissage
        temp = image_cv2.copy()
        for _ in range(3):
            temp = cv2.bilateralFilter(temp, 9, 90, 90)
        
        # Détection des contours multi-échelle
        gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)
        
        # Quantization des couleurs avec plus de nuances
        Z = temp.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        K = 12  # Plus de couleurs pour meilleur résultat
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        center = np.uint8(center)
        res = center[label.flatten()]
        quantized = res.reshape((temp.shape))
        
        # Amélioration de la saturation
        hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)  # Saturation +30%
        quantized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Fusion avec contours
        edges_inv = cv2.bitwise_not(edges)
        cartoon = cv2.bitwise_and(quantized, quantized, mask=edges_inv)
        
        return cartoon
    
    def anime_filter(self, image_cv2) -> np.ndarray:
        """
        Filtre anime amélioré avec couleurs vives et contours nets.
        """
        # Lissage préservant les contours
        temp = image_cv2.copy()
        for _ in range(4):
            temp = cv2.bilateralFilter(temp, 9, 100, 100)
        
        # Augmentation de la saturation et luminosité (look anime)
        hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)  # Saturation +50%
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Luminosité +10%
        temp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Quantization douce pour effet dessiné
        Z = temp.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, label, center = cv2.kmeans(Z, 16, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result = res.reshape((temp.shape))
        
        # Détection contours fins (style anime)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilater légèrement les contours
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Appliquer les contours noirs
        result[edges == 255] = [0, 0, 0]
        
        return result
    
    def watercolor_filter(self, image_cv2) -> np.ndarray:
        """
        Filtre aquarelle - effet peinture douce.
        """
        # Réduction de détails
        temp = cv2.stylization(image_cv2, sigma_s=60, sigma_r=0.6)
        
        # Effet de diffusion
        temp = cv2.edgePreservingFilter(temp, flags=1, sigma_s=60, sigma_r=0.4)
        
        # Ajout de texture aquarelle
        noise = np.random.normal(0, 3, temp.shape).astype(np.float32)
        temp = np.clip(temp.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Légère désaturation pour effet aquarelle
        hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * 0.85
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def oil_painting_filter(self, image_cv2) -> np.ndarray:
        """
        Filtre peinture à l'huile - effet épais et texturé.
        """
        # Effet peinture à l'huile d'OpenCV
        result = cv2.xphoto.oilPainting(image_cv2, 7, 1)
        
        # Augmentation de la saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def sketch_filter(self, image_cv2) -> np.ndarray:
        """
        Filtre sketch/croquis - noir et blanc avec contours.
        """
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        inv_gray = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        
        # Dodge blend
        def dodge_blend(front, back):
            result = back * 255 / (255 - front + 1)
            result[result > 255] = 255
            return result.astype(np.uint8)
        
        sketch = dodge_blend(blur, gray)
        
        # Convertir en BGR pour cohérence
        result = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def _fallback_stylize(self, image_cv2, style: str) -> np.ndarray:
        """Fallback simple si les fonctions avancées ne sont pas disponibles."""
        # Simple bilateral filter + quantization
        temp = cv2.bilateralFilter(image_cv2, 9, 75, 75)
        
        Z = temp.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8 if style in ["pixel_art", "comic_book"] else 12
        _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        center = np.uint8(center)
        res = center[label.flatten()]
        return res.reshape(image_cv2.shape)
    
    def stylize_image(self, img_cv2, style: str) -> np.ndarray:
        """Applique un style selon le style demandé."""
        try:
            if style == "cartoon":
                result = self.cartoon_filter(img_cv2)
            elif style == "anime":
                result = self.anime_filter(img_cv2)
            elif style == "watercolor":
                result = self.watercolor_filter(img_cv2)
            elif style == "oil_painting":
                result = self.oil_painting_filter(img_cv2)
            elif style == "sketch":
                result = self.sketch_filter(img_cv2)
            elif style == "pixel_art":
                result = self.pixel_art_filter(img_cv2)
            elif style == "comic_book":
                result = self.comic_book_filter(img_cv2)
            else:
                raise ValueError(f"Style inconnu: {style}")
        except cv2.error as e:
            # Fallback si une fonction spécifique n'est pas disponible
            print(f"⚠️  Erreur OpenCV pour {style}, utilisation du fallback")
            result = self._fallback_stylize(img_cv2, style)
        
        return result
    
    def pixel_art_filter(self, image_cv2) -> np.ndarray:
        """
        Filtre pixel art - downscale puis upscale avec pixelisation.
        """
        height, width = image_cv2.shape[:2]
        
        # Downscale drastique
        small_size = (width // 16, height // 16)
        small = cv2.resize(image_cv2, small_size, interpolation=cv2.INTER_LINEAR)
        
        # Quantization des couleurs (palette limitée)
        Z = small.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, label, center = cv2.kmeans(Z, 32, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        quantized = res.reshape(small.shape)
        
        # Upscale sans antialiasing (nearest neighbor pour effet pixel)
        result = cv2.resize(quantized, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return result
    
    def comic_book_filter(self, image_cv2) -> np.ndarray:
        """
        Filtre comic book - effet bande dessinée avec demi-teintes.
        """
        # Bilateral filter pour lissage
        temp = cv2.bilateralFilter(image_cv2, 9, 80, 80)
        temp = cv2.bilateralFilter(temp, 9, 80, 80)
        
        # Quantization couleurs vives
        Z = temp.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, label, center = cv2.kmeans(Z, 10, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        quantized = res.reshape(temp.shape)
        
        # Contours épais style BD
        gray = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Épaissir les contours
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Appliquer contours noirs
        quantized[edges == 255] = [0, 0, 0]
        
        # Boost saturation
        hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return result
    
    def process_image_to_style(self, image_path: str, style: str) -> Image.Image:
        """
        Traite une image et l'applique au style.
        
        Args:
            image_path: Chemin vers l'image source
            style: 'anime' ou 'cartoon'
        
        Returns:
            PIL Image du résultat
        """
        # Charger l'image
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Redimensionner si trop grand (évite les problèmes mémoire)
        height, width = img_cv2.shape[:2]
        if height > 1024 or width > 1024:
            scale = 1024 / max(height, width)
            img_cv2 = cv2.resize(img_cv2, (int(width*scale), int(height*scale)))
        
        # Appliquer le filtre
        result = self.stylize_image(img_cv2, style)
        
        # Convertir vers PIL
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)


class DatasetPreparer:
    """Prépare le dataset pour le fine-tuning."""
    
    def __init__(self, source_dir: str, output_dir: str = "datasets"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.transformer = StyleTransformer()
        
        # Styles sélectionnés pour le projet
        self.styles = ["anime", "oil_painting"]
        self.metadata_per_style = {style: [] for style in self.styles}
        
    def create_dataset_structure(self):
        """Crée la structure de dossiers pour le dataset."""
        for style in self.styles:
            train_original = self.output_dir / style / "train" / "images_original"
            train_edited = self.output_dir / style / "train" / "images_edited"
            
            train_original.mkdir(parents=True, exist_ok=True)
            train_edited.mkdir(parents=True, exist_ok=True)
            
            print(f"✅ Dossiers créés pour {style}")
    
    def process_all_images(self):
        """Traite toutes les images source."""
        # Trouver toutes les images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        if self.source_dir.is_dir():
            # Si c'est un dossier, chercher les images dedans
            image_files = [f for f in self.source_dir.rglob('*') 
                          if f.suffix.lower() in image_extensions]
        else:
            # Sinon chercher dans des sous-dossiers
            image_files = [f for f in self.source_dir.glob('**/*') 
                          if f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise ValueError(f"❌ Aucune image trouvée dans {self.source_dir}")
        
        print(f"📸 {len(image_files)} images trouvées")
        
        # Traiter chaque image
        for idx, image_path in enumerate(tqdm(image_files, desc="Traitement images")):
            try:
                self._process_single_image(image_path, idx)
            except Exception as e:
                print(f"⚠️  Erreur avec {image_path}: {e}")
    
    def _process_single_image(self, image_path: Path, idx: int):
        """Traite une image unique."""
        # Charger l'image originale
        original = Image.open(image_path).convert("RGB")
        
        # Redimensionner si nécessaire
        max_size = 768
        if original.width > max_size or original.height > max_size:
            original.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Créer paires pour chaque style
        for style in self.styles:
            try:
                # Appliquer la transformation
                styled = self.transformer.process_image_to_style(str(image_path), style)
                
                # Redimensionner styled si nécessaire
                if styled.width > max_size or styled.height > max_size:
                    styled.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Sauvegarder les images
                filename = f"image_{idx:06d}.jpg"
                
                original_path = self.output_dir / style / "train" / "images_original" / filename
                styled_path = self.output_dir / style / "train" / "images_edited" / filename
                
                original.save(original_path, quality=95)
                styled.save(styled_path, quality=95)
                
                # Ajouter aux métadatas avec variations de prompts
                prompts = [
                    f"Convert to {style} style",
                    f"Transform into {style.replace('_', ' ')}",
                    f"Apply {style.replace('_', ' ')} filter",
                    f"Make it look like {style.replace('_', ' ')}",
                ]
                
                metadata_entry = {
                    "file_name": str(styled_path.relative_to(self.output_dir / style)),
                    "text": prompts[idx % len(prompts)],  # Varier les prompts
                    "original": str(original_path.relative_to(self.output_dir / style)),
                    "edited": str(styled_path.relative_to(self.output_dir / style))
                }
                self.metadata_per_style[style].append(metadata_entry)
                
            except Exception as e:
                print(f"⚠️  Erreur lors du traitement de {image_path} en {style}: {e}")
    
    def save_metadata(self):
        """Sauvegarde les fichiers metadata.jsonl pour chaque style."""
        for style in self.styles:
            metadata_path = self.output_dir / style / "metadata.jsonl"
            
            with open(metadata_path, 'w') as f:
                for entry in self.metadata_per_style[style]:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"✅ Metadata sauvegardé: {metadata_path} ({len(self.metadata_per_style[style])} entrées)")
    
    def generate_info(self):
        """Génère un fichier d'information sur le dataset."""
        info = {
            "styles": self.styles,
            "total_images_per_style": {style: len(self.metadata_per_style[style]) 
                                       for style in self.styles},
            "structure": {
                style: {
                    "train_original": f"{style}/train/images_original/",
                    "train_edited": f"{style}/train/images_edited/",
                    "metadata": f"{style}/metadata.jsonl"
                }
                for style in self.styles
            }
        }
        
        info_path = self.output_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n📋 Info dataset: {info_path}")
        print(f"   Styles: {', '.join(self.styles)}")
        for style in self.styles:
            print(f"   - {style}: {len(self.metadata_per_style[style])} paires")
    
    def run(self):
        """Exécute le pipeline complet."""
        print("\n" + "="*60)
        print("🚀 Préparation du dataset pour fine-tuning")
        print("="*60 + "\n")
        
        self.create_dataset_structure()
        self.process_all_images()
        self.save_metadata()
        self.generate_info()
        
        print("\n" + "="*60)
        print("✨ Dataset prêt pour le fine-tuning!")
        print("="*60 + "\n")


def main():
    import argparse
    allowed_styles = [
        "anime", "cartoon", "watercolor", "oil_painting",
        "sketch", "pixel_art", "comic_book"
    ]
    
    parser = argparse.ArgumentParser(description="Prépare le dataset pour fine-tuning")
    parser.add_argument("--source", type=str, default="testA", # si on veut changer le dossier source, sinon utiliser "testA" par défaut
                       help="Dossier contenant les images réalistes")
    parser.add_argument("--output", type=str, default="datasets",
                       help="Dossier de sortie pour le dataset")
    parser.add_argument("--styles", type=str, nargs="+", 
                       default=["anime", "oil_painting"])
    
    args = parser.parse_args()
    
    # Vérifier que le dossier source existe
    if not Path(args.source).exists():
        print(f"❌ Erreur: {args.source} n'existe pas")
        print(f"   Créez un dossier '{args.source}' et remplissez-le d'images!")
        return
    
    # Lancer la préparation
    preparer = DatasetPreparer(source_dir=args.source, output_dir=args.output)
    
    # Filtrer les styles si spécifiés
    if args.styles:
        preparer.styles = [s for s in args.styles if s in allowed_styles]
        if not preparer.styles:
            print("❌ Aucun style valide sélectionné")
            print(f"   Styles disponibles: {', '.join(allowed_styles)}")
            return
        preparer.metadata_per_style = {s: [] for s in preparer.styles}
        print(f"🎨 Styles sélectionnés: {', '.join(preparer.styles)}")
    
    preparer.run()


if __name__ == "__main__":
    main()

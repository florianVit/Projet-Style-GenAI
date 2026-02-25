"""
Script pour télécharger un dossier dataset depuis le dépôt GitHub.
Utilise git sparse-checkout pour télécharger uniquement le dossier souhaité.

Usage:
  python download_dataset.py
  python download_dataset.py --dataset oil_painting
  python download_dataset.py --dataset anime --output mon_dossier
"""

import argparse
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/florianVit/Projet-Style-GenAI.git"
AVAILABLE_DATASETS = ["oil_painting", "anime"]


def download_dataset(dataset: str, output_dir: str):
    """Télécharge un dossier dataset via git sparse-checkout."""
    output_path = Path(output_dir)

    if output_path.exists():
        print(f"⚠️  Le dossier '{output_dir}' existe déjà. Supprimez-le pour relancer.")
        return

    tmp_dir = Path(f"/tmp/dataset_download_{dataset}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    try:
        print(f"📥 Téléchargement du dataset '{dataset}' depuis le dépôt...")

        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--sparse", REPO_URL, str(tmp_dir)],
            check=True,
        )

        subprocess.run(
            ["git", "sparse-checkout", "set", f"datasets/{dataset}"],
            cwd=str(tmp_dir),
            check=True,
        )

        src = tmp_dir / "datasets" / dataset
        shutil.copytree(str(src), str(output_path))

        print(f"✅ Dataset '{dataset}' téléchargé dans : {output_path}")

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge un dossier dataset du projet Style-GenAI"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="oil_painting",
        choices=AVAILABLE_DATASETS,
        help=f"Dataset à télécharger. Choix : {', '.join(AVAILABLE_DATASETS)} (défaut : oil_painting)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Dossier de destination (défaut : datasets/<dataset>)",
    )

    args = parser.parse_args()
    output_dir = args.output or f"datasets/{args.dataset}"

    download_dataset(args.dataset, output_dir)


if __name__ == "__main__":
    main()

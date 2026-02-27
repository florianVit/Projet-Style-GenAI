# Projet IA Générative — Fine-tuning style (Anime + Oil Painting)

## Objectif du projet
Ce projet sert à **appliquer un style graphique à une image** via IA, en fine-tunant un modèle Instruct-Pix2Pix avec **LoRA**.

Objectifs visés:
- style **anime japonais**
- style **oil painting**

Approche:
1. Générer un dataset de paires `original -> stylisé`
2. Fine-tuner un adaptateur LoRA par style
3. Utiliser l’adaptateur pour styliser n’importe quelle image

---

## État actuel
- Modèle LoRA `anime` entraîné: `models/instructpix2pix-anime-lora/`
- Modèle LoRA `oil_painting` entraîné: `models/instructpix2pix-oil-lora/`
- Script one-click pour test image: `run_style_one_click.py`

---

## Structure du projet
- `prepare_dataset.py` : prépare le dataset pour l’entraînement (métadata + paires)
- `instruct_pix2pix_edit.py` : inférence Instruct-Pix2Pix (+ support LoRA)
- `run_style_one_click.py` : sélection d’image via fenêtre + génération automatique
- `datasets/` : datasets de fine-tuning (`anime`, `oil_painting`)
- `models/` : poids LoRA entraînés
- `outputs/` : images générées
- `testA/` : dossier source utilisé pour créer le dataset

---

## 1) Préparer le dataset
Le script est déjà configuré pour `testA` et 2 styles (`anime`, `oil_painting`) par défaut.

```bash
python prepare_dataset.py
```

Sortie attendue:
- `datasets/anime/...`
- `datasets/oil_painting/...`

---

## 2) Fine-tuning LoRA (anime)
Commande utilisée avec paramètres VRAM-friendly:

```bash
python diffusers/examples/research_projects/instructpix2pix_lora/train_instruct_pix2pix_lora.py \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
  --train_data_dir datasets/anime \
  --original_image_column original \
  --edited_image_column edited \
  --edit_prompt_column text \
  --resolution 256 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --max_train_steps 800 \
  --checkpointing_steps 100 \
  --mixed_precision fp16 \
  --seed 42 \
  --rank 4 \
  --output_dir models/instructpix2pix-anime-lora
```

---

## 3) Fine-tuning LoRA (oil painting)
Même logique, en changeant juste le dataset + output:

```bash
python diffusers/examples/research_projects/instructpix2pix_lora/train_instruct_pix2pix_lora.py \
  --pretrained_model_name_or_path timbrooks/instruct-pix2pix \
  --train_data_dir datasets/oil_painting \
  --original_image_column original \
  --edited_image_column edited \
  --edit_prompt_column text \
  --resolution 256 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --max_train_steps 800 \
  --checkpointing_steps 100 \
  --mixed_precision fp16 \
  --seed 42 \
  --rank 4 \
  --output_dir models/instructpix2pix-oil-lora
```

---

## 4) Tester ton modèle (one-click)
### Sans argument (anime par défaut)
```bash
python run_style_one_click.py
```
Puis sélectionne une image dans la fenêtre.

### Forcer un style
```bash
python run_style_one_click.py --style anime
python run_style_one_click.py --style oil_painting
```

### Résultats
Les images sont sauvegardées automatiquement dans `outputs/`:
- image stylisée
- image comparaison (original | stylisée)

---

## Notes importantes
- Le fine-tuning complet Instruct-Pix2Pix (sans LoRA) est trop lourd pour GPU 6 GB, d’où l’approche LoRA.
- Le mode one-click utilise par défaut les LoRA dans `models/instructpix2pix-anime-lora/` et `models/instructpix2pix-oil-lora/`.

---

## Commande de test direct (sans GUI)
### Anime
```bash
python instruct_pix2pix_edit.py \
  --model-id timbrooks/instruct-pix2pix \
  --lora-path models/instructpix2pix-anime-lora \
  --image 1.png \
  --instruction "convert to anime style" \
  --steps 20 \
  --guidance-scale 7 \
  --image-guidance-scale 1.5 \
  --seed 42 \
  --output outputs/anime_test_lora.png \
  --compare outputs/anime_test_compare.png
```

### Oil painting
```bash
python instruct_pix2pix_edit.py \
  --model-id timbrooks/instruct-pix2pix \
  --lora-path models/instructpix2pix-oil-lora \
  --image 1.png \
  --instruction "convert to oil painting style" \
  --steps 20 \
  --guidance-scale 7 \
  --image-guidance-scale 1.5 \
  --seed 42 \
  --output outputs/oil_test_lora.png \
  --compare outputs/oil_test_compare.png
```

# -*- coding: utf-8 -*-
"""
config.py

Archivo de configuración central.
Contiene todas las constantes, rutas y parámetros globales del proyecto.
"""

from pathlib import Path
import torch

# --- Rutas de Archivos ---
BASE_DIR = Path(r"C:\Users\nonit\Desktop\Universidad\melanoma detection\data")
OUTPUT_DIR = Path("./outputs")

TEST_CSV_PATH = BASE_DIR / "test.csv"
TEST_IMG_PATH = BASE_DIR / "test"
EXTERNAL_TEST_PATH = BASE_DIR / "kaggle_external_data" / "test"

MODEL_SAVE_PATH = OUTPUT_DIR / 'final_model_kfold.pth'
SUBMISSION_SAVE_PATH = OUTPUT_DIR / 'submission.csv'

# --- Archivos de Salida ---
MODEL_SAVE_PATH = 'final_model_kfold.pth'
SUBMISSION_SAVE_PATH = 'submission.csv'

# --- Reproducibilidad ---
SEED = 42

# --- Parámetros de Preprocesamiento ---
IMG_SIZE = 224 # tamaño estándar
BATCH_SIZE = 32 # tamaño del lote (aunque Optuna lo puede variar)
NORM_MEAN = [0.485, 0.456, 0.406] # ImageNet mean
NORM_STD = [0.229, 0.224, 0.225] # ImageNet std

# --- Dispositivo ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 3. Optimización (Optuna) ---

# Para ResNet Simple (NO EJECUTAR)
OPTUNA_N_TRIALS_RESNET_SIMPLE = 30
OPTUNA_STUDY_NAME_RESNET_SIMPLE = "melanoma-resnet-simple-study"
DB_NAME_RESNET_SIMPLE = f"{OPTUNA_STUDY_NAME_RESNET_SIMPLE}.db"
OPTUNA_STORAGE_RESNET_SIMPLE = f"sqlite:///{OUTPUT_DIR.resolve() / DB_NAME_RESNET_SIMPLE}"

# Para EfficientNet (NO EJECUTAR)
OPTUNA_N_TRIALS_EFFNET = 50
OPTUNA_STUDY_NAME_EFFNET = "melanoma-effnetstudy"
DB_NAME_EFFNET = f"{OPTUNA_STUDY_NAME_EFFNET}.db"
OPTUNA_STORAGE_EFFNET = f"sqlite:///{OUTPUT_DIR.resolve() / DB_NAME_EFFNET}"

# Para EfficientNet Fine-Tune (NO EJECUTAR)
OPTUNA_N_TRIALS_EFFNET_FT = 40
OPTUNA_STUDY_NAME_EFFNET_FT = "melanoma-effnet-finetune-study"
DB_NAME_EFFNET_FT = f"{OPTUNA_STUDY_NAME_EFFNET_FT}.db"
OPTUNA_STORAGE_EFFNET_FT = f"sqlite:///{OUTPUT_DIR.resolve() / DB_NAME_EFFNET_FT}"

# Para ResNet K-Fold (PRINCIPAL)
OPTUNA_N_TRIALS_RESNET_KFOLD = 30
OPTUNA_STUDY_NAME_RESNET_KFOLD = "melanoma-resnet-kfold-study"
DB_NAME_RESNET_KFOLD = f"{OPTUNA_STUDY_NAME_RESNET_KFOLD}.db"
OPTUNA_STORAGE_RESNET_KFOLD = f"sqlite:///{OUTPUT_DIR.resolve() / DB_NAME_RESNET_KFOLD}"


# --- Entrenamiento Final ---
N_SPLITS = 5 # 5 pliegues para K-Fold
FINAL_EPOCHS = 13 # Épocas para el modelo final

# --- Evaluación y Submission ---
FINAL_THRESHOLD = 0.5
EXTERNAL_TEST_THRESHOLD = 0.65
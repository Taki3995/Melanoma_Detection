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
TEST_CSV_PATH = BASE_DIR / "test.csv"
TEST_IMG_PATH = BASE_DIR / "test"
EXTERNAL_TEST_PATH = BASE_DIR / "kaggle_external_data" / "test"

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

# --- Optimización (Optuna) ---
OPTUNA_N_TRIALS_RESNET_SIMPLE = 30
OPTUNA_STUDY_NAME_RESNET_SIMPLE = "melanoma-resnet-simple-study"
OPTUNA_STORAGE_RESNET_SIMPLE = f"sqlite:///{OPTUNA_STUDY_NAME_RESNET_SIMPLE}.db"

OPTUNA_N_TRIALS_EFFNET = 50
OPTUNA_STUDY_NAME_EFFNET = "melanoma-effnetstudy"
OPTUNA_STORAGE_EFFNET = f"sqlite:///{OPTUNA_STUDY_NAME_EFFNET}.db"

OPTUNA_N_TRIALS_EFFNET_FT = 40
OPTUNA_STUDY_NAME_EFFNET_FT = "melanoma-effnet-finetune-study"
OPTUNA_STORAGE_EFFNET_FT = f"sqlite:///{OPTUNA_STUDY_NAME_EFFNET_FT}.db"

OPTUNA_N_TRIALS_RESNET_KFOLD = 30
OPTUNA_STUDY_NAME_RESNET_KFOLD = "melanoma-resnet-kfold-study"
OPTUNA_STORAGE_RESNET_KFOLD = f"sqlite:///{OPTUNA_STUDY_NAME_RESNET_KFOLD}.db"

# --- Entrenamiento Final ---
N_SPLITS = 5 # 5 pliegues para K-Fold
FINAL_EPOCHS = 13 # Épocas para el modelo final

# --- Evaluación y Submission ---
FINAL_THRESHOLD = 0.5
EXTERNAL_TEST_THRESHOLD = 0.65
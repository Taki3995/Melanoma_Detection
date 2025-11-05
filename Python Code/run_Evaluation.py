# -*- coding: utf-8 -*-
"""
run_evaluation.py

Script para ejecutar la evaluación del modelo (Sección 7).
Carga el modelo final y genera reportes, matriz de confusión
y análisis de errores en un fold de validación.
"""

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import KFold

# Importar constantes, datos, modelos y utilidades
import config
from dataset import get_data
from models import get_resnet_model
from utils import get_all_preds_tta # Usamos la TTA de utils

def main():
    print("========================== TESTING ==========================")

    # Cargar el estudio para obtener 'best_value' y 'best_params'
    try:
        study = optuna.load_study(
            study_name=config.OPTUNA_STUDY_NAME_RESNET_KFOLD,
            storage=config.OPTUNA_STORAGE_RESNET_KFOLD
        )
        best_params = study.best_params
    except Exception as e:
        print(f"Error cargando el estudio: {e}")
        return

    # Cargar datos
    _, _, full_dataset = get_data()

    # --- Análisis (Sección 7.1) ---
    print("--- Análisis Completo del Rendimiento del Modelo ---")
    print(f"El MEJOR F1-SCORE PROMEDIO alcanzado en la búsqueda fue: {study.best_value:.4f}")

    # --- Preparar modelo (Sección 7.2) ---
    print("\n--- Cargando el modelo final para evaluación en un fold representativo ---")
    eval_model = get_resnet_model(
        dropout_rate=best_params['dropout'],
        unfreeze_layers=best_params['unfreeze_layers']
    ).to(config.DEVICE)
    eval_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    eval_model.eval()

    # Preparar el DataLoader del fold de validación
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    _, val_idx = next(iter(kf.split(full_dataset)))
    val_sampler_eval = SubsetRandomSampler(val_idx)
    val_loader_eval = DataLoader(
        full_dataset,
        batch_size=best_params['batch_size'],
        sampler=val_sampler_eval,
        num_workers=2
    )

    # --- Predicciones TTA (Sección 7.3 y 7.4) ---
    print("\nObteniendo predicciones con Test-Time Augmentation (TTA)...")
    y_true_eval, y_probs_eval = get_all_preds_tta(eval_model, val_loader_eval, config.DEVICE)

    precision, recall, thresholds = precision_recall_curve(y_true_eval, y_probs_eval)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_threshold_idx = np.argmax(f1_scores[:-1])
    best_threshold_eval = thresholds[best_threshold_idx]
    best_f1_eval = f1_scores[best_threshold_idx]
    print(f"Se encontró un umbral óptimo de {best_threshold_eval:.4f} que resulta en un F1-Score de {best_f1_eval:.4f}")

    # --- Predicciones finales (Sección 7.5) ---
    y_preds_eval = (y_probs_eval >= best_threshold_eval).astype(int)
    print("\n--- REPORTE DE MÉTRICAS (Umbral Óptimo) ---")
    print(classification_report(y_true_eval, y_preds_eval, target_names=['No Melanoma (0)', 'Melanoma (1)']))

    print("\n--- MATRIZ DE CONFUSIÓN (Umbral Óptimo) ---")
    cm = confusion_matrix(y_true_eval, y_preds_eval)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred No-Melanoma', 'Pred Melanoma'],
                yticklabels=['Real No-Melanoma', 'Real Melanoma'])
    plt.title('Matriz de Confusión - Fold Representativo (Umbral Óptimo)', fontsize=14)
    plt.ylabel('Clase Real', fontsize=12)
    plt.xlabel('Clase Predicha', fontsize=12)
    plt.show()

    # --- Análisis de Errores (Sección 7.5) ---
    print("\n--- ANÁLISIS DE ERRORES ---")
    misclassified_indices_eval = np.where(y_true_eval != y_preds_eval)[0]
    print(f"Se encontraron {len(misclassified_indices_eval)} imágenes mal clasificadas.")

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(misclassified_indices_eval[:5]):
        original_idx = val_idx[idx]
        image, label = full_dataset[original_idx] # Obtenemos el tensor
        
        # Desnormalizar la imagen para visualización
        image = image.permute(1, 2, 0).cpu().numpy()
        mean = np.array(config.NORM_MEAN)
        std = np.array(config.NORM_STD)
        image = std * image + mean
        image = np.clip(image, 0, 1)

        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(image)
        ax.set_title(f"Real: {label} | Pred: {y_preds_eval[idx]}")
        ax.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
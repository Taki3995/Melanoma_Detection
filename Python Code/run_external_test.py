# -*- coding: utf-8 -*-
"""
run_external_test.py

Script para probar el modelo final en un dataset externo (Sección 10).
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Importar constantes, datos, modelos y utilidades
import config
from dataset import valid_transforms # Usamos transforms de validación
from models import get_resnet_model
from utils import get_all_preds_tta

def main():
    print("# 10.- Prueba con Data Externa (Kaggle)")
    
    # Cargar el estudio para 'best_params' (batch_size, arquitectura)
    try:
        study = optuna.load_study(
            study_name=config.OPTUNA_STUDY_NAME_RESNET_KFOLD,
            storage=config.OPTUNA_STORAGE_RESNET_KFOLD
        )
        best_params = study.best_params
    except Exception as e:
        print(f"Error cargando el estudio: {e}")
        return

    # --- Cargar Datos Externos ---
    print(f"Cargando dataset externo desde: {config.EXTERNAL_TEST_PATH}")
    external_dataset = datasets.ImageFolder(
        config.EXTERNAL_TEST_PATH, 
        transform=valid_transforms
    )
    print(f"Mapeo de clases del dataset externo: {external_dataset.class_to_idx}")
    external_loader = DataLoader(
        external_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # --- Cargar Modelo ---
    print("\nCargando tu mejor modelo ResNet entrenado...")
    eval_model = get_resnet_model(
        dropout_rate=best_params['dropout'],
        unfreeze_layers=best_params['unfreeze_layers']
    ).to(config.DEVICE)
    eval_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    eval_model.eval()

    # --- Obtener Predicciones ---
    print("\nObteniendo predicciones con Test-Time Augmentation (TTA)...")
    external_y_true, external_y_probs = get_all_preds_tta(eval_model, external_loader, config.DEVICE)

    # --- Aplicar Umbral y Evaluar ---
    print(f"Usando el umbral fijo de {config.EXTERNAL_TEST_THRESHOLD} para la evaluación externa.")
    external_y_preds = (external_y_probs >= config.EXTERNAL_TEST_THRESHOLD).astype(int)

    print("\n-------- Reporte de Clasificación en Dataset Externo ---------")
    final_f1_score = f1_score(external_y_true, external_y_preds, zero_division=0)
    print(f"F1-Score Final en el Dataset Externo: {final_f1_score:.4f}")
    print(classification_report(external_y_true, external_y_preds, target_names=['Benign (0)', 'Malign (1)'], zero_division=0))

    print("\n-------- Matriz de Confusión en Dataset Externo --------")
    cm = confusion_matrix(external_y_true, external_y_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Benign', 'Pred Malign'],
                yticklabels=['Real Benign', 'Real Malign'])
    plt.title('Matriz de Confusión - Dataset Externo', fontsize=14)
    plt.ylabel('Clase Real'); plt.xlabel('Clase Predicha'); plt.show()

if __name__ == "__main__":
    main()
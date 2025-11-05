# -*- coding: utf-8 -*-
"""
run_visualization.py

Script para ejecutar las visualizaciones avanzadas (Sección 8).
- Gráfica de consistencia K-Fold
- Análisis de errores (imágenes)
- Interpretabilidad con Grad-CAM
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import optuna
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from PIL import Image

# Imports de Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Importar constantes, datos, modelos y utilidades
import config
from dataset import get_data
from models import get_resnet_model
from utils import (
    train_one_epoch, 
    evaluate_model, 
    get_all_preds_tta, 
    plot_predictions, 
    apply_grad_cam
)

def main():
    print("========================== RESULTS VISUALIZATION ==========================")

    # --- Cargar Estudio y Datos ---
    study = optuna.load_study(
        study_name=config.OPTUNA_STUDY_NAME_RESNET_KFOLD,
        storage=config.OPTUNA_STORAGE_RESNET_KFOLD
    )
    best_params = study.best_params
    params = best_params
    _, _, full_dataset = get_data()

    # --- 1. Gráficas Comparativas de Métricas (K-Fold) ---
    print("--- Analizando la consistencia del mejor trial en los 5 folds ---")
    fold_scores_viz = []
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"   --- Evaluando Fold {fold+1}/{config.N_SPLITS} ---")
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(full_dataset, batch_size=params['batch_size'], sampler=train_sampler, num_workers=2)
        val_loader = DataLoader(full_dataset, batch_size=params['batch_size'], sampler=val_sampler, num_workers=2)
        
        model_fold = get_resnet_model(dropout_rate=params['dropout'], unfreeze_layers=params['unfreeze_layers']).to(config.DEVICE)
        optimizer_fold = getattr(optim, params['optimizer'])(model_fold.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.BCEWithLogitsLoss()

        best_fold_f1 = 0.0; epochs_no_improve = 0
        for epoch in range(25):
            train_one_epoch(model_fold, train_loader, optimizer_fold, criterion, config.DEVICE)
            _, val_f1 = evaluate_model(model_fold, val_loader, criterion, config.DEVICE, epoch + 1, 25)
            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1; epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= 5: 
                break
        fold_scores_viz.append(best_fold_f1)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[f"Fold {i+1}" for i in range(config.N_SPLITS)], y=fold_scores_viz, palette="viridis")
    plt.title('Consistencia del F1-Score en el Mejor Trial (K-Fold)', fontsize=16)
    plt.ylim(min(fold_scores_viz) - 0.005, max(fold_scores_viz) + 0.005)
    for index, value in enumerate(fold_scores_viz):
        plt.text(index, value, f"{value:.4f}", ha='center', va='bottom', fontsize=10)
    plt.show()

    # --- 2. Análisis de errores (Visual) ---
    print("--- Cargando modelo final para visualización de predicciones ---")
    viz_model = get_resnet_model(
        dropout_rate=best_params['dropout'],
        unfreeze_layers=best_params['unfreeze_layers']
    ).to(config.DEVICE)
    viz_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    viz_model.eval()

    print("Preparando datos del fold representativo...")
    _, val_idx = next(iter(kf.split(full_dataset))) # Mismo fold que en evaluación
    val_sampler_viz = SubsetRandomSampler(val_idx)
    val_loader_viz = DataLoader(full_dataset, batch_size=best_params['batch_size'], sampler=val_sampler_viz, num_workers=2)

    print("Generando predicciones en el fold representativo...")
    y_true_viz, y_probs_viz = get_all_preds_tta(viz_model, val_loader_viz, config.DEVICE)
    y_preds_viz = (y_probs_viz >= 0.5).astype(int) # Usando umbral 0.5
    print("¡Predicciones generadas!")

    correct_indices = np.where(y_true_viz == y_preds_viz)[0]
    misclassified_indices = np.where(y_true_viz != y_preds_viz)[0]

    print("\n--- Ejemplos de Predicciones Correctas ---")
    plot_predictions(correct_indices, "Predicciones Correctas (Umbral 0.5)", val_idx, full_dataset, y_true_viz, y_preds_viz)
    
    print("\n--- Ejemplos de Predicciones Incorrectas ---")
    plot_predictions(misclassified_indices, "Predicciones Incorrectas (Umbral 0.5)", val_idx, full_dataset, y_true_viz, y_preds_viz)

    # --- 3. Interpretabilidad con Grad-CAM ---
    print("\n--- Visualización con Grad-CAM ---")
    target_layers = [viz_model.layer4[-1]]
    cam = GradCAM(model=viz_model, target_layers=target_layers)

    print("\n--- Visualización con Grad-CAM en Predicciones Correctas ---")
    apply_grad_cam(correct_indices, "Grad-CAM - Predicciones Correctas", val_idx, full_dataset, y_true_viz, y_preds_viz, cam, config.IMG_SIZE, config.DEVICE)
    
    print("\n--- Visualización con Grad-CAM en Predicciones Incorrectas ---")
    apply_grad_cam(misclassified_indices, "Grad-CAM - Predicciones Incorrectas", val_idx, full_dataset, y_true_viz, y_preds_viz, cam, config.IMG_SIZE, config.DEVICE)

if __name__ == "__main__":
    main()
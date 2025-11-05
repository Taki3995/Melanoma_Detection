# -*- coding: utf-8 -*-
"""
run_training.py

Script para ejecutar el entrenamiento final (Sección 6).
Carga los mejores hiperparámetros del estudio K-Fold y
entrena el modelo sobre el 100% de los datos.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import random
import numpy as np

# Importar constantes, datos, modelos y utilidades
import config
from dataset import get_data
from models import get_resnet_model
from utils import train_one_epoch

def main():
    print("========================== TRAINING ==========================")
    
    # Setear seeds
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Cargar el estudio para obtener los mejores parámetros
    try:
        study = optuna.load_study(
            study_name=config.OPTUNA_STUDY_NAME_RESNET_KFOLD,
            storage=config.OPTUNA_STORAGE_RESNET_KFOLD
        )
        best_params = study.best_params
        print("Mejores hiperparámetros cargados del estudio K-Fold:")
        print(best_params)
    except Exception as e:
        print(f"Error cargando el estudio: {e}")
        print(f"Asegúrate de haber ejecutado 'run_optuna.py' con 'run_study_resnet_kfold()' activo.")
        return

    # Cargar el dataset completo
    print("Cargando el 100% de los datos para el entrenamiento final...")
    _, _, full_dataset = get_data()

    # Entrenamiento del Modelo Final
    print("\n--- Entrenando el modelo final sobre el 100% de los datos ---")

    final_train_loader = DataLoader(
        full_dataset, 
        batch_size=best_params['batch_size'], 
        shuffle=True, 
        num_workers=2
    )

    final_model = get_resnet_model(
        dropout_rate=best_params['dropout'],
        unfreeze_layers=best_params['unfreeze_layers']
    ).to(config.DEVICE)

    optimizer = getattr(optim, best_params['optimizer'])(
        final_model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(config.FINAL_EPOCHS):
        print(f"   --- Época Final {epoch+1}/{config.FINAL_EPOCHS} ---")
        # Aquí no necesitamos pasar epoch/total_epochs porque es solo train
        train_one_epoch(final_model, final_train_loader, optimizer, criterion, config.DEVICE)

    torch.save(final_model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"\nModelo final entrenado y guardado en '{config.MODEL_SAVE_PATH}'")

if __name__ == "__main__":
    main()
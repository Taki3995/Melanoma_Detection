# -*- coding: utf-8 -*-
"""
run_optuna.py

Script para ejecutar la optimización de hiperparámetros (Sección 5).
Define y ejecuta todos los estudios de Optuna.


!!! IMPORTANTE !!! 
No es necesario hacer una busqueda nueva, está el archivo con los mejores hiperparámetros en el repositorio
(melanoma-resnet-kfold-study.db)

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import optuna
import random

# Importar constantes, datos, modelos y utilidades
import config
from dataset import get_data
import models # Importa el archivo completo
from utils import train_one_epoch, evaluate_model

# --- Configuración Global ---
# Seteamos la semilla aquí para KFold y Samplers
random.seed(config.SEED)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Cargar datos UNA VEZ
print("Cargando datasets para Optuna...")
train_dataset, val_dataset, full_dataset = get_data()
print(f"Usando dispositivo para la optimización: {config.DEVICE}")

# --- 1. Optuna para ResNet18 (NO EJECUTAR) ---
def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    print(f"\n--- Trial #{trial.number} ---")
    print(f"Parámetros: lr={lr:.6f}, batch_size={batch_size}, dropout={dropout:.4f}, optimizer={optimizer_name}")
    try:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    except RuntimeError:
        print(f"Error de memoria con batch_size={batch_size}. Saltando trial.")
        raise optuna.exceptions.TrialPruned()
    
    # Usamos el modelo 'simple' de models.py
    model = models.get_resnet_model_simple(dropout_rate=dropout).to(config.DEVICE) 
    optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    MAX_EPOCHS = 30; PATIENCE = 4; epochs_no_improve = 0; best_val_f1 = 0.0

    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        # Pasamos epoch+1 y MAX_EPOCHS a evaluate_model
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, config.DEVICE, epoch+1, MAX_EPOCHS) 
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        trial.report(val_f1, epoch)
        if trial.should_prune():
            print("  Trial podado por bajo rendimiento.")
            raise optuna.exceptions.TrialPruned()
        if epochs_no_improve >= PATIENCE:
            print(f"  Early Stopping.")
            break
    return best_val_f1

def run_study_resnet_simple():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        direction='maximize',
        study_name=config.OPTUNA_STUDY_NAME_RESNET_SIMPLE,
        storage=config.OPTUNA_STORAGE_RESNET_SIMPLE,
        load_if_exists=True,
        pruner=pruner
    )
    print(f"Iniciando/Reanudando estudio '{config.OPTUNA_STUDY_NAME_RESNET_SIMPLE}'")
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS_RESNET_SIMPLE)
    
    # ___ MOSTRAR RESULTADOS ___
    print("\n\nBúsqueda (ResNet Simple) finalizada.")
    best_trial = study.best_trial
    # ... (copia todo tu código de 'Imprimir resultados' aquí) ...
    print(study.trials_dataframe().sort_values(by="value", ascending=False).head(10))


# --- 2. Optuna para EfficientNet-B2 (NO EJECUTAR) ---
def objective_effnet(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    # ... (copia el resto de la función objective_effnet) ...
    # Asegúrate de llamar a models.get_efficientnet_model
    model = models.get_efficientnet_model(dropout_rate=dropout).to(config.DEVICE)
    # ... (copia el resto del bucle de entrenamiento) ...
    # val_loss, val_f1 = evaluate_model(model, val_loader, criterion, config.DEVICE, epoch + 1, MAX_EPOCHS)
    return best_val_f1 # Reemplaza el ...

def run_study_effnet():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6)
    study_final = optuna.create_study(
        direction='maximize',
        study_name=config.OPTUNA_STUDY_NAME_EFFNET,
        storage=config.OPTUNA_STORAGE_EFFNET,
        load_if_exists=True,
        pruner=pruner
    )
    study_final.optimize(objective_effnet, n_trials=config.OPTUNA_N_TRIALS_EFFNET)
    # ... (copia todo tu código de 'Imprimir resultados' aquí) ...
    print(study_final.trials_dataframe().sort_values(by="value", ascending=False).head(10))


# --- 3. Optuna para Fine Tuning (NO EJECUTAR) ---
def objective_finetune(trial):
    unfreeze_blocks = trial.suggest_int('unfreeze_blocks', 0, 2)
    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    # ... (copia el resto de la función objective_finetune) ...
    # Asegúrate de llamar a models.get_efficientnet_finetune_model
    model = models.get_efficientnet_finetune_model(dropout_rate=dropout, unfreeze_blocks=unfreeze_blocks).to(config.DEVICE)
    # ... (copia el resto del bucle de entrenamiento) ...
    # val_loss, val_f1 = evaluate_model(model, val_loader, criterion, config.DEVICE, epoch + 1, MAX_EPOCHS)
    return best_val_f1 # Reemplaza el ...

def run_study_finetune():
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6)
    study_finetune = optuna.create_study(
        direction='maximize',
        study_name=config.OPTUNA_STUDY_NAME_EFFNET_FT,
        storage=config.OPTUNA_STORAGE_EFFNET_FT,
        load_if_exists=True,
        pruner=pruner
    )
    study_finetune.optimize(objective_finetune, n_trials=config.OPTUNA_N_TRIALS_EFFNET_FT)
    # ... (copia todo tu código de 'Imprimir resultados' y el 'loaded_study.trials[16]') ...
    loaded_study = optuna.load_study(study_name=config.OPTUNA_STUDY_NAME_EFFNET_FT, storage=config.OPTUNA_STORAGE_EFFNET_FT)
    print(loaded_study.trials_dataframe().sort_values(by="value", ascending=False).head(10))
    # print(loaded_study.trials[16]) # Descomenta si lo necesitas


# --- 4. Optuna para Resnet Mejorada (ESTUDIO PRINCIPAL) ---
def objective_kfold(trial):
    unfreeze_layers = trial.suggest_int('unfreeze_layers', 1, 2)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam'])

    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    fold_f1_scores = []
    print(f"\n--- Trial #{trial.number} ---")
    print(f"Params: Unfreeze={unfreeze_layers}, lr={lr:.6f}, bs={batch_size}, dropout={dropout:.4f}, optim={optimizer_name}, wd={weight_decay:.6f}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"   --- Fold {fold+1}/{config.N_SPLITS} ---")
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
        val_loader = DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)
        
        # Usamos el modelo 'mejorado' de models.py
        model = models.get_resnet_model(dropout_rate=dropout, unfreeze_layers=unfreeze_layers).to(config.DEVICE)
        optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        MAX_EPOCHS = 25; PATIENCE = 5; epochs_no_improve = 0; best_fold_f1 = 0.0

        for epoch in range(MAX_EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
            val_loss, val_f1 = evaluate_model(model, val_loader, criterion, config.DEVICE, epoch + 1, MAX_EPOCHS)
            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"     Early Stopping en la época {epoch+1}.")
                break
        fold_f1_scores.append(best_fold_f1)
        print(f"     Mejor F1 para este fold: {best_fold_f1:.4f}")

    average_f1 = np.mean(fold_f1_scores)
    print(f"   --> Trial finalizado. F1 Promedio en {config.N_SPLITS} folds: {average_f1:.4f}")
    return average_f1

def run_study_resnet_kfold():
    study = optuna.create_study(
        direction='maximize',
        study_name=config.OPTUNA_STUDY_NAME_RESNET_KFOLD,
        storage=config.OPTUNA_STORAGE_RESNET_KFOLD,
        load_if_exists=True
    )
    print(f"Iniciando/Reanudando estudio '{config.OPTUNA_STUDY_NAME_RESNET_KFOLD}'")
    study.optimize(objective_kfold, n_trials=config.OPTUNA_N_TRIALS_RESNET_KFOLD)
    
    print("\n\nBúsqueda (ResNet K-Fold) finalizada.")
    best_trial = study.best_trial
    print(f"  F1 Score: {best_trial.value:.4f}")
    print("  Mejores Hiperparámetros: ")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")
    return study # Retornamos el estudio para el siguiente script

if __name__ == "__main__":
    print("--- Ejecutando todos los estudios de Optuna ---")
    
    # --- Descomenta el estudio que quieras ejecutar ---
    
    # print("\n--- Ejecutando ResNet Simple (NO EJECUTAR) ---")
    # run_study_resnet_simple()
    
    # print("\n--- Ejecutando EfficientNet (NO EJECUTAR) ---")
    # run_study_effnet()
    
    # print("\n--- Ejecutando EfficientNet Fine-Tuning (NO EJECUTAR) ---")
    # run_study_finetune()
    
    print("\n--- Ejecutando ResNet K-Fold (ESTUDIO PRINCIPAL) ---")
    run_study_resnet_kfold()
    
    print("\n--- Todos los estudios seleccionados han finalizado ---")
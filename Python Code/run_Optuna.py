# -*- coding: utf-8 -*-
"""
run_optuna.py

Script para ejecutar la optimización de hiperparámetros (Sección 5).
Define y ejecuta todos los estudios de Optuna.

!!! IMPORTANTE !!!
No es necesario ejecutar este archivo, ya que el estudio con el mejor modelo para el modelo final está guardado
en el repositorio, y el código más adelante simplemente llama al database en vez de correr todo denuevo.

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
import models # Importa el archivo completo de modelos
from utils import train_one_epoch, evaluate_model # Importamos el "engine"

# --- Configuración Global ---
# Seteamos la semilla aquí para KFold y Samplers
random.seed(config.SEED)
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if config.DEVICE.type == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Cargar datos UNA VEZ al inicio del script
print("Cargando datasets para Optuna...")
# train_dataset y val_dataset se usarán en los estudios simples
# full_dataset se usará en el estudio K-Fold
train_dataset, val_dataset, full_dataset = get_data()
print(f"Usando dispositivo para la optimización: {config.DEVICE}")

# ===================================================================
# --- 1. Optuna para ResNet18 (NO EJECUTAR) ---
# ===================================================================
def objective(trial):
    """
    Función objetivo para el estudio ResNet18 simple (Sección 5.2).
    """
    # Sugerir hiperparámetros
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    print(f"\n--- Trial #{trial.number} (ResNet Simple) ---")
    print(f"Parámetros: lr={lr:.6f}, batch_size={batch_size}, dropout={dropout:.4f}, optimizer={optimizer_name}")

    # Configurar DataLoaders, Modelo y Optimizador
    try:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    except RuntimeError:
        print(f"Error de memoria con batch_size={batch_size}. Saltando trial.")
        raise optuna.exceptions.TrialPruned()

    # Usamos el modelo 'simple' de models.py
    model = models.get_resnet_model_simple(dropout_rate=dropout).to(config.DEVICE) 
    optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Bucle de entrenamiento y evaluación
    MAX_EPOCHS = 30
    PATIENCE = 4
    epochs_no_improve = 0
    best_val_f1 = 0.0

    for epoch in range(MAX_EPOCHS):
        #entrenar modelo en cada epoca
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)

        # evaluar
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, config.DEVICE, epoch + 1, MAX_EPOCHS)

        #imprimir resultados para monitorear
        print(f"  Epoch {epoch+1}/{MAX_EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Guardar el mejor F1 de este trial
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Reportar a Optuna para Pruning (poda)
        trial.report(val_f1, epoch)
        if trial.should_prune():
            print("  Trial podado por bajo rendimiento.")
            raise optuna.exceptions.TrialPruned()

        if epochs_no_improve >= PATIENCE:
            print(f"  Early Stopping.")
            break

    return best_val_f1

def run_study_resnet_simple():
    """Ejecuta el estudio para ResNet18 Simple."""
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
    print("Mejor trial:")
    best_trial = study.best_trial
    print(f"  F1 Score: {best_trial.value:.4f}")
    print("  Mejores Hiperparámetros: ")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")
        
    # Imprimir resultados (código de la sección 5.2.2)
    print("\n--- Tabla con los 10 Mejores Experimentos ---")
    results_df = study.trials_dataframe()
    print(results_df.sort_values(by="value", ascending=False).head(10))


# ===================================================================
# --- 2. Optuna para EfficientNet-B2 (NO EJECUTAR) ---
# ===================================================================
def objective_effnet(trial):
    """
    Función objetivo para el estudio EfficientNet-B2 (Sección 5.3).
    """
    # Optimizamos los hiperparámetros clave
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])

    print(f"\n--- Trial #{trial.number} (EfficientNet) ---")
    print(f"Parámetros: lr={lr:.6f}, bs={batch_size}, dropout={dropout:.4f}, optim={optimizer_name}, wd={weight_decay:.6f}")

    try:
        # Si da error, cambiar num_workers a 0
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    except RuntimeError:
        raise optuna.exceptions.TrialPruned()

    model = models.get_efficientnet_model(dropout_rate=dropout).to(config.DEVICE)
    optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    MAX_EPOCHS = 50; PATIENCE = 6; best_val_f1 = 0.0; epochs_no_improve = 0
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, config.DEVICE, epoch + 1, MAX_EPOCHS)
        scheduler.step()

        if val_f1 > best_val_f1: best_val_f1 = val_f1; epochs_no_improve = 0
        else: epochs_no_improve += 1

        trial.report(val_f1, epoch)
        if trial.should_prune(): print("  --> Trial podado."); raise optuna.exceptions.TrialPruned()
        if epochs_no_improve >= PATIENCE: print(f"  --> Early Stopping en la época {epoch+1}."); break

    print(f"  --> Trial finalizado. Mejor Val F1: {best_val_f1:.4f}")
    return best_val_f1

def run_study_effnet():
    """Ejecuta el estudio para EfficientNet-B2."""
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6)
    study_final = optuna.create_study(
        direction='maximize',
        study_name=config.OPTUNA_STUDY_NAME_EFFNET,
        storage=config.OPTUNA_STORAGE_EFFNET,
        load_if_exists=True,
        pruner=pruner
    )

    print(f"Iniciando/Reanudando estudio final '{config.OPTUNA_STUDY_NAME_EFFNET}'.")
    study_final.optimize(objective_effnet, n_trials=config.OPTUNA_N_TRIALS_EFFNET)

    # --- Mostrar Resultados ---
    print("\n\nBúsqueda (EfficientNet) finalizada.")
    trial = study_final.best_trial
    print(f"Mejor F1 Score: {trial.value:.4f}")
    print("Mejores Hiperparámetros: ")
    for key, value in trial.params.items():
        print(f"  - {key}: {value}")
        
    # Imprimir resultados (código de la sección 5.3.2)
    print("\n--- Tabla con los 10 Mejores Experimentos ---")
    results_df = study_final.trials_dataframe()
    print(results_df.sort_values(by="value", ascending=False).head(10))

# ===================================================================
# --- 3. Optuna para Fine Tuning (NO EJECUTAR) ---
# ===================================================================
def objective_finetune(trial):
    """
    Función objetivo para el estudio EfficientNet Fine-Tuning (Sección 5.4).
    """
    # Bloques de capas descongelar (de 1 a 3)
    unfreeze_blocks = trial.suggest_int('unfreeze_blocks', 0, 2)

    # Tasas de aprendizaje más bajas para fine tuning
    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)

    # Hiperparámetros
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.4, 0.7)
    weight_decay = trial.suggest_float('weight_decay', 5e-3, 5e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])

    print(f"\n--- Trial #{trial.number} (Fine-Tuning) ---")
    print(f"Params: Unfreeze={unfreeze_blocks}, lr={lr:.6f}, bs={batch_size}, dropout={dropout:.4f}, optim={optimizer_name}, wd={weight_decay:.6f}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.get_efficientnet_finetune_model(dropout_rate=dropout, unfreeze_blocks=unfreeze_blocks).to(config.DEVICE)
    optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    MAX_EPOCHS = 50; PATIENCE = 6; best_val_f1 = 0.0; epochs_no_improve = 0
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, config.DEVICE, epoch + 1, MAX_EPOCHS)
        scheduler.step()

        if val_f1 > best_val_f1: best_val_f1 = val_f1; epochs_no_improve = 0
        else: epochs_no_improve += 1

        trial.report(val_f1, epoch)
        if trial.should_prune(): print("  --> Trial podado."); raise optuna.exceptions.TrialPruned()
        if epochs_no_improve >= PATIENCE: print(f"  --> Early Stopping en la época {epoch+1}."); break

    print(f"  --> Trial finalizado. Mejor Val F1: {best_val_f1:.4f}")

    return best_val_f1

def run_study_finetune():
    """Ejecuta el estudio para EfficientNet Fine-Tuning."""
    # --- CONFIGURACIÓN Y EJECUCIÓN DEL ESTUDIO DE FINE-TUNING ---
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6)
    study_finetune = optuna.create_study(
        direction='maximize',
        study_name=config.OPTUNA_STUDY_NAME_EFFNET_FT,
        storage=config.OPTUNA_STORAGE_EFFNET_FT,
        load_if_exists=True,
        pruner=pruner
    )
    study_finetune.optimize(objective_finetune, n_trials=config.OPTUNA_N_TRIALS_EFFNET_FT)
    
    # --- Imprimir resultados (Sección 5.4.3) ---
    print("\n\nBúsqueda (Fine-Tune) finalizada.")
    loaded_study = optuna.load_study(
        study_name=config.OPTUNA_STUDY_NAME_EFFNET_FT,
        storage=config.OPTUNA_STORAGE_EFFNET_FT
    )

    best_trial = loaded_study.best_trial
    print("--- Mejor Resultado Encontrado ---")
    print(f"F1 Score: {best_trial.value:.4f}")
    print("Mejores Hiperparámetros:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value}")

    print("\n--- Tabla con los 10 Mejores Experimentos ---")
    results_df = loaded_study.trials_dataframe()
    print(results_df.sort_values(by="value", ascending=False).head(10))
    
    # Código para imprimir el trial 16
    try:
        print("\n--- Analizando Trial 16 (como en el notebook) ---")
        print(loaded_study.trials[16])
    except IndexError:
        print("El Trial 16 no existe (aún).")

# ===================================================================
# --- 4. Optuna para Resnet Mejorada (ESTUDIO PRINCIPAL) ---
# ===================================================================
def objective_kfold(trial):
    """
    Función objetivo para el estudio ResNet Mejorado con K-Fold (Sección 5.5).
    """
    # Sugerir hiperparámetros
    unfreeze_layers = trial.suggest_int('unfreeze_layers', 1, 2)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam'])

    # Configuración de K-Fold
    kf = KFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    fold_f1_scores = []

    print(f"\n--- Trial #{trial.number} (ResNet K-Fold) ---")
    print(f"Params: Unfreeze={unfreeze_layers}, lr={lr:.6f}, bs={batch_size}, dropout={dropout:.4f}, optim={optimizer_name}, wd={weight_decay:.6f}")

    # Bucle de validación cruzada
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"  --- Fold {fold+1}/{config.N_SPLITS} ---")

        # Crear DataLoaders específicos para este fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
        val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)

        # Crear una nueva instancia del modelo
        model = models.get_resnet_model(dropout_rate=dropout, unfreeze_layers=unfreeze_layers).to(config.DEVICE)
        optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Bucle de entrenamiento para este fold
        MAX_EPOCHS = 25
        PATIENCE = 5
        epochs_no_improve = 0
        best_fold_f1 = 0.0

        for epoch in range(MAX_EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
            val_loss, val_f1 = evaluate_model(model, val_loader, criterion, config.DEVICE, epoch + 1, MAX_EPOCHS)

            if val_f1 > best_fold_f1:
                best_fold_f1 = val_f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"    Early Stopping en la época {epoch+1}.")
                break

        fold_f1_scores.append(best_fold_f1)
        print(f"    Mejor F1 para este fold: {best_fold_f1:.4f}")

    # El resultado es el promedio de los F1 de todos los folds
    average_f1 = np.mean(fold_f1_scores)
    print(f"  --> Trial finalizado. F1 Promedio en {config.N_SPLITS} folds: {average_f1:.4f}")
    return average_f1

def run_study_resnet_kfold():
    """Ejecuta el estudio principal para ResNet Mejorado con K-Fold."""
    # _____ CONFIGURACIÓN Y EJECUCIÓN DEL ESTUDIO______
    study = optuna.create_study(
        direction='maximize',
        study_name=config.OPTUNA_STUDY_NAME_RESNET_KFOLD,
        storage=config.OPTUNA_STORAGE_RESNET_KFOLD,
        load_if_exists=True
    )
    
    print(f"Iniciando/Reanudando estudio '{config.OPTUNA_STUDY_NAME_RESNET_KFOLD}'")
    study.optimize(objective_kfold, n_trials=config.OPTUNA_N_TRIALS_RESNET_KFOLD)

    # ------MOSTRAR RESULTADOS ------
    print("\n\nBúsqueda (ResNet K-Fold) finalizada.")
    print("Mejor trial:")
    trial = study.best_trial
    print(f"  F1 Score: {trial.value:.4f}")
    print("  Mejores Hiperparámetros: ")
    for key, value in trial.params.items():
        print(f"    - {key}: {value}")
    
    return study # Retornamos el estudio por si otro script lo necesita

# ===================================================================
# --- Bloque Principal de Ejecución ---
# ===================================================================
if __name__ == "__main__":
    print("--- Ejecutando script de optimización (run_optuna.py) ---")
    
    # --- Descomenta el estudio que quieras ejecutar ---
    # --- (Puedes ejecutar varios en secuencia) ---
    
    # print("\n--- 1. Iniciando: ResNet Simple (NO EJECUTAR) ---")
    # run_study_resnet_simple()
    
    # print("\n--- 2. Iniciando: EfficientNet (NO EJECUTAR) ---")
    # run_study_effnet()
    
    # print("\n--- 3. Iniciando: EfficientNet Fine-Tuning (NO EJECUTAR) ---")
    # run_study_finetune()
    
    print("\n--- 4. Iniciando: ResNet K-Fold (ESTUDIO PRINCIPAL) ---")
    run_study_resnet_kfold()
    
    print("\n--- Todos los estudios seleccionados han finalizado ---")
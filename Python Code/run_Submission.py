# -*- coding: utf-8 -*-
"""
run_submission.py

Script para generar el archivo de submission (Sección 9).
"""
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
import optuna

# Importar constantes, datos, modelos
import config
from dataset import TestDataset, valid_transforms # Usamos valid_transforms para test
from models import get_resnet_model

def main():
    print("# 9.- Generación de submission")
    
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

    # Preparar el DataLoader de Test
    print("--- Preparando el conjunto de datos de Test ---")
    test_dataset = TestDataset(
        root_dir=config.TEST_IMG_PATH, 
        csv_file=config.TEST_CSV_PATH, 
        transform=valid_transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Cargar el Modelo Final
    print("\nCargando modelo final para la predicción...")
    submission_model = get_resnet_model(
        dropout_rate=best_params['dropout'],
        unfreeze_layers=best_params['unfreeze_layers']
    ).to(config.DEVICE)
    submission_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    submission_model.eval()

    predictions = []
    image_ids = []
    print(f"Usando el umbral fijo de {config.FINAL_THRESHOLD} para las predicciones finales.")

    with torch.no_grad():
        for inputs, fnames in tqdm(test_loader, desc="Generando predicciones para Kaggle"):
            inputs = inputs.to(config.DEVICE)
            outputs = submission_model(inputs)
            probs = torch.sigmoid(outputs).squeeze(-1)
            
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)

            preds = (probs.cpu().numpy() >= config.FINAL_THRESHOLD).astype(int)
            
            predictions.extend(preds)
            image_ids.extend(fnames)

    # Crear y Guardar el DataFrame
    submission_df = pd.DataFrame({
        'ID': image_ids,
        'predicted': predictions
    })
    submission_df.to_csv(config.SUBMISSION_SAVE_PATH, index=False)

    print(f"\nArchivo '{config.SUBMISSION_SAVE_PATH}' creado exitosamente.")
    print(submission_df.head())

if __name__ == "__main__":
    main()
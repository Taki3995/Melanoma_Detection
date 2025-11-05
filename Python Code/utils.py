# -*- coding: utf-8 -*-
"""
utils.py

Funciones de utilidad reutilizables para el proyecto:
- Helpers de EDA (list_images, show_examples)
- Engine de entrenamiento (train_one_epoch, evaluate_model)
- Funciones de Evaluación (get_all_preds_tta)
- Funciones de Visualización (plot_predictions, apply_grad_cam)

"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

# Imports de Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Funciones de EDA (Sección 2) ---

def list_images(path):
    return [f for f in path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

def show_examples(images, title, n=5):
    plt.figure(figsize=(15,3))
    for i in range(n):
        img_path = random.choice(images)
        img = Image.open(img_path)
        plt.subplot(1,n,i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
    plt.show()

# --- Funciones de Engine (Sección 5) ---

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Entrenando", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(-1), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)

def evaluate_model(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    val_loss = 0.0
    all_outputs, all_labels = [], []
    pbar = tqdm(loader, desc=f"Evaluando (Época {epoch}/{total_epochs})", leave=False)

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.squeeze(-1), labels.float())
            val_loss += loss.item() * inputs.size(0)

            all_outputs.append(outputs.squeeze(-1).cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels).numpy()
    probabilities = torch.sigmoid(all_outputs).numpy()
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (probabilities >= threshold).astype(int)
        current_f1 = f1_score(all_labels, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1

    return val_loss / len(loader.dataset), best_f1

# --- Función TTA (Sección 7) ---

def get_all_preds_tta(model, loader, device):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Obteniendo predicciones con TTA"):
            inputs = inputs.to(device)

            # Predicción 1: imagen original
            outputs_original = model(inputs)
            probs_original = torch.sigmoid(outputs_original).squeeze(-1)

            # Predicción 2: imagen volteada horizontal
            inputs_flipped = torch.flip(inputs, [3])
            outputs_flipped = model(inputs_flipped)
            probs_flipped = torch.sigmoid(outputs_flipped).squeeze(-1)

            # Promediar probabilidades
            avg_probs = (probs_original + probs_flipped) / 2.0

            if avg_probs.dim() == 0:
                avg_probs = avg_probs.unsqueeze(0)

            all_probs.extend(avg_probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)

# --- Funciones de Visualización (Sección 8) ---

def plot_predictions(indices, title, val_idx, full_dataset, y_true_viz, y_preds_viz):
    plt.figure(figsize=(20, 6))
    plt.suptitle(title, fontsize=16)

    for i, idx in enumerate(indices[:5]):
        original_idx = val_idx[idx]
        image, label = full_dataset.datasets[0].samples[original_idx]
        image = Image.open(image).convert('RGB')

        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(image)
        ax.set_title(f"Real: {y_true_viz[idx]} | Pred: {y_preds_viz[idx]}")
        ax.axis("off")
    plt.show()

def apply_grad_cam(indices, title, val_idx, full_dataset, y_true_viz, y_preds_viz, cam, img_size, device):
    plt.figure(figsize=(20, 6))
    plt.suptitle(title, fontsize=16)
    for i, idx in enumerate(indices[:5]):
        original_idx = val_idx[idx]
        tensor_image, label = full_dataset[original_idx] # Tensor normalizado

        # Encontrar la ruta de la imagen original para visualización
        if original_idx < len(full_dataset.datasets[0]):
            local_idx = original_idx
            original_image_path, _ = full_dataset.datasets[0].samples[local_idx]
        else:
            local_idx = original_idx - len(full_dataset.datasets[0])
            original_image_path, _ = full_dataset.datasets[1].samples[local_idx]
        
        # Cargar y preparar la imagen original (sin normalizar)
        rgb_img = np.array(Image.open(original_image_path).convert('RGB').resize((img_size, img_size))) / 255.0

        targets = [ClassifierOutputTarget(0)] # 0 es la clase 'melanoma' (según tu mapeo)
        
        # Asegúrate de que el tensor_image tenga el batch dimension y esté en el dispositivo
        grayscale_cam = cam(input_tensor=tensor_image.unsqueeze(0).to(device), targets=targets)
        grayscale_cam = grayscale_cam[0, :] # Quitar batch dim

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(visualization)
        ax.set_title(f"Real: {y_true_viz[idx]} | Pred: {y_preds_viz[idx]}")
        ax.axis("off")
    plt.show()
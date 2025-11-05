# -*- coding: utf-8 -*-
"""
models.py

Contiene todas las arquitecturas de modelos definidas en el notebook.
(CNN, ResNet, EfficientNet, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- 1. CNN (Baseline) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # capa convolucional 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # entra una imagen RGB (3 canales), salen 16 mapas de características

        # capa convolucional 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # entra 16 mapas de características, salen 32

        # capa convolucional 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # entra 32 mapas de características, salen 64

        # capa de pooling para reducir dimensionalidad (queda con lo mas importante)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # definir clasificador
        self.fc1 = nn.Linear(64 * 28 * 28, 512) # capa totalmente conectada 1
        self.fc2 = nn.Linear(512, 1) # capa totalmente conectada 2 (salida binaria)

        self.dropout = nn.Dropout(0.5) # capa de dropout para evitar sobreajuste (apaga neuronas aleatoriamente para evitar sobreajuste)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # aplicar conv1 + ReLU + pooling
        x = self.pool(F.relu(self.conv2(x))) # aplicar conv2 + ReLU + pooling
        x = self.pool(F.relu(self.conv3(x))) # aplicar conv3 + ReLU + pooling

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x

# --- 2. Transfer learning (ResNet) (NO EJECUTAR) ---
# (Renombrado a 'get_resnet_model_simple' para evitar conflicto con el modelo final)

"""
def get_resnet_model_simple(dropout_rate):
    # Cargar modelo resnet18 pre entrenado en imagenet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # congelar parametros para no reentrenar, solo adaptar
    for param in model.parameters():
        param.requires_grad = False

    # reemplazar la capa final (fully connected)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate), # Usamos el dropout que nos pasará Optuna
        nn.Linear(512, 1)          # Salida binaria final
    )
    # Descongelar solo la nueva capa clasificadora para el entrenamiento
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
"""


# --- 3. EfficientNet B2 (NO EJECUTAR) ---

"""
def get_efficientnet_model(dropout_rate):
    # Cargar EfficientNet-B2 preentrenado en ImageNet
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

    # Congelar los parámetros del modelo base
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar el clasificador final
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_ftrs, 1) # Salida única para clasificación binaria
    )

    # Descongelar solo los parámetros del nuevo clasificador
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
"""


# --- 4. Efficientnet con fine tuning (NO EJECUTAR) ---

"""
def get_efficientnet_finetune_model(dropout_rate, unfreeze_blocks):
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)

    # Congelar todo por defecto
    for param in model.parameters():
        param.requires_grad = False

    # EfficientNet-B2 tiene 8 bloques de capas convolucionales (features[0] a features[7])
    # Vamos a descongelar los últimos 'unfreeze_blocks'
    if unfreeze_blocks > 0:
        # Descongelamos desde el final hacia el principio
        for i in range(unfreeze_blocks):
            for param in model.features[-(i+1)].parameters():
                param.requires_grad = True

    # Reemplazar y descongelar el clasificador
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_ftrs, 1)
    )
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
"""


# --- 5. ResNet Mejorado (El modelo final) ---

def get_resnet_model(dropout_rate, unfreeze_layers):
    # Cargar modelo resnet18 pre-entrenado
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Congelar todos los parámetros por defecto
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar las últimas capas convolucionales según el parámetro
    if unfreeze_layers >= 1:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if unfreeze_layers >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True

    # Reemplazar y descongelar la capa final (clasificador)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(256, 1)
    )
    # El clasificador siempre se entrena
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
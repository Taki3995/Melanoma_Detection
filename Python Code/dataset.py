# -*- coding: utf-8 -*-
"""
dataset.py

Contiene las transformaciones (data augmentation) y las
definiciones de los Datasets (ImageFolder, ConcatDataset, TestDataset).
"""

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, ConcatDataset
import pandas as pd
from PIL import Image
import os

# Importar configuraciones
import config

# --- Data Augmentation ---

train_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=25, shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
])

valid_transforms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD)
])

# --- Función de Carga de Datos (Train/Val) ---

def get_data():
    """
    Carga los datasets de train y validación, corrige el mapeo
    de clases y los retorna, junto con un ConcatDataset.
    """
    
    # ---------CREAR DATASETS---------
    train_dataset_with_aug = datasets.ImageFolder(
        config.BASE_DIR / "train", 
        transform=train_transforms
    )
    valid_dataset_no_aug = datasets.ImageFolder(
        config.BASE_DIR / "valid", 
        transform=valid_transforms
    )

    # ---------CORREGIR MAPEO DE CLASES---------
    correct_mapping = {'nomel': 0, 'mel': 1}

    # reemplazar el mapeo de clases en los datasets
    train_dataset_with_aug.class_to_idx = correct_mapping
    valid_dataset_no_aug.class_to_idx = correct_mapping
    train_dataset_with_aug.classes = list(correct_mapping.keys())
    valid_dataset_no_aug.classes = list(correct_mapping.keys())

    # cambiar etiquetas
    new_train_samples = [(path, 1 - label) for path, label in train_dataset_with_aug.samples]
    new_val_samples = [(path, 1 - label) for path, label in valid_dataset_no_aug.samples]

    train_dataset_with_aug.samples = new_train_samples
    valid_dataset_no_aug.samples = new_val_samples

    # actualizar targets
    train_dataset_with_aug.targets = [s[1] for s in new_train_samples]
    valid_dataset_no_aug.targets = [s[1] for s in new_val_samples]

    # ------ Unir en un solo dataset -----------
    full_dataset = ConcatDataset([train_dataset_with_aug, valid_dataset_no_aug])
    
    print(f"Tamaño total del dataset combinado para K-Fold: {len(full_dataset)} imágenes.")
    print(f"Tamaño del dataset de entrenamiento: {len(train_dataset_with_aug)}")
    print(f"Tamaño del dataset de validación: {len(valid_dataset_no_aug)}")
    print(f"Mapeo de clases: {train_dataset_with_aug.class_to_idx}")
    
    return train_dataset_with_aug, valid_dataset_no_aug, full_dataset

# --- Definición del Dataset para el Conjunto de Prueba ---

class TestDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = pd.read_csv(csv_file)['ID'].values

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Devuelve la imagen transformada y su nombre de archivo (ID)
        return image, img_name
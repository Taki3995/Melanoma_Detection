"""
1.- Introduccion y objetivo
Este trabajo se centra en detectar según imagenes el cáncer de piel de tipo melanoma. Ésta es una de las enfermedades mas 
graves en dermatología, y la detección temprana a través de imágenes es clave para reducir riesgos y mejorar tratamientos. 
El objetivo es entrenar un modelo de clasifiación binario de imágenes capaz de distinguir entre melanoma y no melanoma, 
exporando como distintos hiperparametros (como la tasa de aprendizaje, numero de epocas, tamaño de batch, regularizacion, 
entre otros) afectan el desempeño. La metrica principal para evaluar es el F1 Score.

"""

# Importar librerías
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from pathlib import Path
import numpy as np
import copy
import random
from PIL import Image

import torch
from torchvision import transforms, datasets, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import os

from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from tqdm.auto import tqdm
import optuna

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset, SubsetRandomSampler

"""# 2.- EDA

## Estructura del dataset y distribucion de datos
"""

# Rutas Archivos

BASE_DIR = Path(r"C:\Users\nonit\Desktop\Universidad\melanoma detection\data")

train_mel_path   = BASE_DIR / "train" / "mel"
train_nomel_path = BASE_DIR / "train" / "nomel"
val_mel_path     = BASE_DIR / "valid" / "mel"
val_nomel_path   = BASE_DIR / "valid" / "nomel"
test_path        = BASE_DIR / "test"

test_csv = pd.read_csv(BASE_DIR / "test.csv")

# Conteo
def list_images(path):
    return [f for f in path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

train_mel_imgs = list_images(train_mel_path)
train_nomel_imgs = list_images(train_nomel_path)
val_mel_imgs = list_images(val_mel_path)
val_nomel_imgs = list_images(val_nomel_path)
test_imgs = list_images(test_path)

# Resumen

print("Estructura del dataset:")
print("train : imágenes para entrenamiento (subcarpetas 'mel', 'nomel')")
print("val   : imágenes para validación (subcarpetas 'mel', 'nomel')")
print("test  : imágenes para evaluación final (sin etiquetas, IDs en test.csv)")

print("\n___________ TRAIN ___________")
print(f"Melanoma    : {len(train_mel_imgs)}")
print(f"No Melanoma : {len(train_nomel_imgs)}")
print(f"Total       : {len(train_mel_imgs) + len(train_nomel_imgs)}")

print("\n___________ VALIDATION ___________")
print(f"Melanoma    : {len(val_mel_imgs)}")
print(f"No Melanoma : {len(val_nomel_imgs)}")
print(f"Total       : {len(val_mel_imgs) + len(val_nomel_imgs)}")

print("\n___________ TEST ___________")
print(f"Total imágenes en carpeta test : {len(test_imgs)}")
print(f"Total IDs en test.csv          : {len(test_csv)}")
if len(test_imgs) == len(test_csv):
    print("El número de imágenes en la carpeta test coincide con el número de IDs en test.csv.")

else:
    print("El número de imágenes en la carpeta test NO coincide con el número de IDs en test.csv.")


# Tipo de archivos

ext_train = set([img.suffix.lower() for img in train_mel_imgs + train_nomel_imgs])
ext_val = set([img.suffix.lower() for img in val_mel_imgs + val_nomel_imgs])
ext_test = set([img.suffix.lower() for img in test_imgs])

print("\n _______ FORMATOS _______")
print(f"Train: {ext_train}")
print(f"Val  : {ext_val}")
print(f"Test : {ext_test}")

"""## Visualización de ejemplos"""

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

show_examples(train_mel_imgs, "Melanoma")
show_examples(train_nomel_imgs, "No Melanoma")

"""## Balance de clases"""

data_summary = pd.DataFrame({
    "Conjunto": ["Train", "Train", "Val", "Val"],
    "Clase": ["Mel", "NoMel", "Mel", "NoMel"],
    "Cantidad": [
        len(train_mel_imgs),
        len(train_nomel_imgs),
        len(val_mel_imgs),
        len(val_nomel_imgs)
    ]
})

plt.figure(figsize=(6,4))
sns.barplot(data=data_summary, x="Conjunto", y="Cantidad", hue="Clase")
plt.title("Distribución de clases en Train y Val")
plt.ylabel("Número de imágenes")
plt.show()

# Ratios para ver balance
ratio_train = len(train_mel_imgs) / len(train_nomel_imgs)
ratio_val = len(val_mel_imgs) / len(val_nomel_imgs)

print(f"Relación Train Mel/NoMel: {ratio_train:.2f}")
print(f"Relación Val Mel/NoMel: {ratio_val:.2f}")

"""## Análisis

Podemos observar que se tienen 7288 datos de training y 2080 datos de validation, ambos distribuidos equitativamente en melanoma y n melanoma (50% de cada uno). Se observa con el ejemplo visual que no todas las imagenes estan derechas, que algunas esta, por ejemplo, rotadas. No se tiene un desbalance de clases, estan balanceadas.

# 3.- Preprocesamiento

### Redimensionamiento, normalización y preparación de imágenes
"""

# REPRODUCIBILIDAD
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

IMG_SIZE = 224 # tamaño estándar. Las imagenes se convertirán a 224x224 píxeles
BATCH_SIZE = 32 # tamaño del lote. El modelo aprendera en grupos de 32 imágenes

# Data Augmentation

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=25, shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
])

valid_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # normalizar valores RGB
])

"""Esto nos deja con las imagenes redimensionadas para que todas sean de 224x224, además de aplicarle cambios a algunas imagenes aleatoriamente como voltearla o rotarla, para que el modelo pueda reconocer melanomas del lado que sea y angulo que sea. Además se convierte la imagen a tensor y se normalizan los valores.

### Cargar datos y organizarlos en batches
"""

# ---------CREAR DATASETS---------

train_dataset_with_aug = datasets.ImageFolder(BASE_DIR / "train", transform=train_transforms)
valid_dataset_no_aug = datasets.ImageFolder(BASE_DIR / "valid", transform=valid_transforms)

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

# verificar los tamaños de los datasets y dataloaders
print(f"Tamaño del dataset de entrenamiento: {len(train_dataset_with_aug)}")
print(f"Tamaño del dataset de validación: {len(valid_dataset_no_aug)}")

# verificar etiquetas
class_map = train_dataset_with_aug.class_to_idx
print(f"Mapeo de clases: {class_map}")

"""# 4.- Modelos a probar

## CNN

**CNN desde cero**: Red Neuronal Convolucional (CNN), especializada en datos con estructura de malla, como imagenes. Las primeras capas detectan cosas simples como bordes, esquinas o colores, y las capas mas profundas combinan esos patrones simples para detectar características mas complejas como texturas, formas, etc.
Este modelo sirve a modo de baseliine, ya que el modelo de transfer learning deberia poder superarlo (si no lo hace algo está mal). Al implementarlo lo puedo usar de comparativa para saber si mi otro modelo esta funcionando correctamente. Además, al implementarlo desde cero, se va a especializar totalmente en detectar melanomas, sin sesgos de otros conocimientos.
"""

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

"""## Transfer learning (ResNet) (NO EJECUTAR)

**Transfer learning**: Modelo ya entrenado con un dataset enorme, que se adaptará al problema (deteccion de melanomas). Utilizaré ResNet (Residual Networks) ya que es una arquitectura conocida y potente para detectar caracteristicas visuales. Este modelo es adecuado ya que es una ventaja enorme que este pre-entrenado en deteccion de caracteristicas con una enorme cantidad de datos, por lo que aprender de los datos dados de melanomas no será dificil. Además, puede usar conocimiento de distincion de otras cosas y aplicarlo para distinguir lunares normales de melanomas. Este modelo es mas adecuado para el problema, ya que contamos con datos limitados y entrenar un modelo desde cero tiene un alto riesgo de sobreajuste.

Luego de buscar hiperparámetros, entrenarlo y testearlo, me dio como resultado un F1-Score de 0.98095, y al aplicarle tva dio 0.98113. Aunque el resultado es muy bueno, quería obtener algo mejor.
"""

def get_resnet_model(dropout_rate):
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

"""## EfficientNet B2 (NO EJECUTAR)

Modelo mas moderno y potente que resnet18, CNN mas eficiente y preciso en tareas de visión por computadora, principalmente en clasificación de imágenes. Escala profundida, ancho y resolución equilibrada y simultáneamente.

Cuando implementé este modelo, tambien aumenté la dificultad del transfer learning y esto provocó que el modelo se comportara de peor manera que el ResNet... bajando el F1-score a 0.97. Esto desanimó un poco por el esfuerzo y el tiempo dedicado en un modelo que pensé se comportaría mejor por ser mas eficiente que resnet, pero que tuvo un resultado peor. No me quise rendir con EfficientNet asi que probé disminuir la dificultad del data augmentation, pero aunque dió un resultado mejor que la vez anterior, aún asi era mas bajo que con resnet (0.98076).
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

"""## Efficientnet con fine tuning (NO EJECUTAR)

Al no tener buenos resultados anteriormente, probé una ultima cosa, fine tuning. Consistió en descongelar algunas de las ultimas capas del modelo pre-entrenado y re entrenarlas con una tasa de aprendizaje muy baja. Esto permitió que el modelo adapte sus detectores de características para que sean más específicas a las imagenes que estamos analizando en vez de las de ImageNet. Esto me dio un F1-score de 1.0, pero descubrí que el modelo estaba con overfitting.
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

"""## ResNet Mejorado

Como con efficientnet el modelo se me sobreajustaba, probé devolverme a un modelo mas simple: ResNet.
A diferencia que la primera vez que lo implementé, ahora recibe datos con data augmentation mas difícil, además de contar con k-folds y fine tuning.
"""

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

"""# 5.- Optimización de hiperparámetros

## Dispositivo, train_one_epoch y evaluate_model

Parámetros y funciones que ocupan todos los modelos ocupados
"""

# ----Dispositivo----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo para la optimización: {device}")

# ---Funciones de Entrenamiento y Evaluación---
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

"""## Optuna para ResNet18 (NO EJECUTAR)

Antes de usar optuna, use búsqueda bayesiana, pero no implementé que guardara un database con el estudio ni una visualización de resultados, por lo que al terminar la búsqueda no tuve como ver los mejores hiperparámetros encontrados, ni algo que me permitiera reanudar la búsqueda si se interrumpía. Por ésto, busque de que se trataba Optuna y lo implementé, asegurando un backup en caso de interrupciones y una mejor visualizacion de lo que encontró el programa.

### Implementación bayesiana con optuna

ahorrar tiempo, guardar historial para no perder todo (como me pasó anteriormente), busqueda mas eficiente
"""

# ______FUNCION OBJECTIVE DE OPTUNA______
# Define un experimento que optuna llamara N_TRIALS veces

def objective(trial):
    # Sugerir hiperparámetros
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])

    print(f"\n--- Trial #{trial.number} ---")
    print(f"Parámetros: lr={lr:.6f}, batch_size={batch_size}, dropout={dropout:.4f}, optimizer={optimizer_name}")

    # Configurar DataLoaders, Modelo y Optimizador
    try:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    except RuntimeError:
        print(f"Error de memoria con batch_size={batch_size}. Saltando trial.")
        raise optuna.exceptions.TrialPruned()

    model = get_resnet_model(dropout_rate=dropout).to(device)
    optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Bucle de entrenamiento y evaluación
    MAX_EPOCHS = 30
    PATIENCE = 4
    epochs_no_improve = 0
    best_val_f1 = 0.0

    for epoch in range(MAX_EPOCHS):
        #entrenar modelo en cada epoca
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # evaluar
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, device)

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

# _____ CONFIGURACIÓN Y EJECUCIÓN DEL ESTUDIO______
N_TRIALS = 30 # Ajustar
STUDY_NAME = "melanoma-resnet-simple-study"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

study = optuna.create_study(
    direction='maximize',
    study_name=STUDY_NAME,
    storage=STORAGE_NAME,
    load_if_exists=True,
    pruner=pruner
)

# Iniciar optimización
print(f"Iniciando/Reanudando estudio '{STUDY_NAME}'. Resultados en '{STUDY_NAME}.db'")
study.optimize(objective, n_trials=N_TRIALS)

# ___ MOSTRAR RESULTADOS ___
print("\n\nBúsqueda finalizada.")
print("Mejor trial:")
trial = study.best_trial
print(f"  F1 Score: {trial.value:.4f}")
print("  Mejores Hiperparámetros: ")
for key, value in trial.params.items():
    print(f"    - {key}: {value}")

"""### Imprimir resultados"""

# Nombre del estudio y almacenamiento
STUDY_NAME = "melanoma-resnet-simple-study"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

# Cargar el estudio completo desde la base de datos
loaded_study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=STORAGE_NAME
)

# Mostrar el mejor resultado
best_trial = loaded_study.best_trial
print("--- Mejor Resultado Encontrado ---")
print(f"F1 Score: {best_trial.value:.4f}")
print("Mejores Hiperparámetros:")
for key, value in best_trial.params.items():
    print(f"  - {key}: {value}")

# Tabla con mejores experimentos
print("\n--- Tabla con los 10 Mejores Experimentos ---")
results_df = loaded_study.trials_dataframe()
# Ordenar por el valor (F1-Score) de mayor a menor
print(results_df.sort_values(by="value", ascending=False).head(10))

"""## Optuna para EfficientNet-B2 (AdamW) (NO EJECUTAR)

Aquí se implementó optuna especificamente para effnet, y no puse opciones de modelos, solo AdamW ya que pensé que al ser mas nuevo que adam y sdg, sería mejor. No funcionó como esperaba.

### Implementacion EfficientNet
"""

def objective_effnet(trial):
    # Optimizamos los hiperparámetros clave
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])

    print(f"\n--- Trial #{trial.number} ---")
    print(f"Parámetros: lr={lr:.6f}, bs={batch_size}, dropout={dropout:.4f}, optim={optimizer_name}, wd={weight_decay:.6f}")

    try:
        # Si da error, cambiar num_workers a 0
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    except RuntimeError:
        raise optuna.exceptions.TrialPruned()

    model = get_efficientnet_model(dropout_rate=dropout).to(device)
    optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    MAX_EPOCHS = 50; PATIENCE = 6; best_val_f1 = 0.0; epochs_no_improve = 0
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, device, epoch + 1, MAX_EPOCHS)
        scheduler.step()

        if val_f1 > best_val_f1: best_val_f1 = val_f1; epochs_no_improve = 0
        else: epochs_no_improve += 1

        trial.report(val_f1, epoch)
        if trial.should_prune(): print("  --> Trial podado."); raise optuna.exceptions.TrialPruned()
        if epochs_no_improve >= PATIENCE: print(f"  --> Early Stopping en la época {epoch+1}."); break

    print(f"  --> Trial finalizado. Mejor Val F1: {best_val_f1:.4f}")
    return best_val_f1

# --- CONFIGURACIÓN Y EJECUCIÓN DEL ESTUDIO FINAL ---
N_TRIALS = 50
STUDY_NAME = "melanoma-effnetstudy"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6)
study_final = optuna.create_study(
    direction='maximize',
    study_name=STUDY_NAME,
    storage=STORAGE_NAME,
    load_if_exists=True,
    pruner=pruner
)

print(f"Iniciando/Reanudando estudio final '{STUDY_NAME}'.")
study_final.optimize(objective_effnet, n_trials=N_TRIALS)

# --- Mostrar Resultados ---
print("\n\nBúsqueda finalizada.")
trial = study_final.best_trial
print(f"Mejor F1 Score: {trial.value:.4f}")
print("Mejores Hiperparámetros: ")
for key, value in trial.params.items():
    print(f"  - {key}: {value}")

"""### Imprimir Resultados"""

# Nombre del estudio y almacenamiento
STUDY_NAME = "melanoma-effnetstudy"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

# Cargar el estudio completo desde la base de datos
loaded_study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=STORAGE_NAME
)

# Mostrar el mejor resultado
best_trial = loaded_study.best_trial
print("--- Mejor Resultado Encontrado ---")
print(f"F1 Score: {best_trial.value:.4f}")
print("Mejores Hiperparámetros:")
for key, value in best_trial.params.items():
    print(f"  - {key}: {value}")

# Tabla con mejores experimentos
print("\n--- Tabla con los 10 Mejores Experimentos ---")
results_df = loaded_study.trials_dataframe()
# Ordenar por el valor (F1-Score) de mayor a menor
print(results_df.sort_values(by="value", ascending=False).head(10))

"""## Optuna para Fine Tuning (NO EJECUTAR)

En esta última funcion de optuna, volví a poner mas de una opcion de modelo (Adam y AdamW) para que compitieran, y ademas agregue el fine tuning descongelando bloques.
"""

def objective_finetune(trial):
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

    model = get_efficientnet_finetune_model(dropout_rate=dropout, unfreeze_blocks=unfreeze_blocks).to(device)
    optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    MAX_EPOCHS = 50; PATIENCE = 6; best_val_f1 = 0.0; epochs_no_improve = 0
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate_model(model, val_loader, criterion, device, epoch + 1, MAX_EPOCHS)
        scheduler.step()

        if val_f1 > best_val_f1: best_val_f1 = val_f1; epochs_no_improve = 0
        else: epochs_no_improve += 1

        trial.report(val_f1, epoch)
        if trial.should_prune(): print("  --> Trial podado."); raise optuna.exceptions.TrialPruned()
        if epochs_no_improve >= PATIENCE: print(f"  --> Early Stopping en la época {epoch+1}."); break

    print(f"  --> Trial finalizado. Mejor Val F1: {best_val_f1:.4f}")

    return best_val_f1

"""### EJECUCION

Esta con error porque al ya encontrar un modelo con score 1.0, detuve la busqueda.
"""

# --- CONFIGURACIÓN Y EJECUCIÓN DEL ESTUDIO DE FINE-TUNING ---
N_TRIALS = 40
STUDY_NAME = "melanoma-effnet-finetune-study"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=6)
study_finetune = optuna.create_study(
    direction='maximize',
    study_name=STUDY_NAME,
    storage=STORAGE_NAME,
    load_if_exists=True,
    pruner=pruner
)
study_finetune.optimize(objective_finetune, n_trials=N_TRIALS)

"""### Imprimir resultados"""

# Nombre del estudio y almacenamiento
STUDY_NAME = "melanoma-effnet-finetune-study"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

# Cargar el estudio completo desde la base de datos
loaded_study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=STORAGE_NAME
)

# Mostrar el mejor resultado
best_trial = loaded_study.best_trial
print("--- Mejor Resultado Encontrado ---")
print(f"F1 Score: {best_trial.value:.4f}")
print("Mejores Hiperparámetros:")
for key, value in best_trial.params.items():
    print(f"  - {key}: {value}")

# Tabla con mejores experimentos
print("\n--- Tabla con los 10 Mejores Experimentos ---")
results_df = loaded_study.trials_dataframe()
# Ordenar por el valor (F1-Score) de mayor a menor
print(results_df.sort_values(by="value", ascending=False).head(10))

"""Se observa que el modelo se sobreentrenó, probablemente por ser un modelo tan potente y tener pocas imagenes a disposicion para entrenar. Un análisis es que quizás pasar de resnet a effnet no fue una buena idea teniendo en cuenta el dataset dado, y estoy usando una herramienta demasiado potente para una tarea que no lo requiere.

Tomaré el tercer mejor resultado, ya que tiene solo 2 bloques descongelados en ves de 3, y aun así un f1-score bueno.
"""

loaded_study.trials[16]

"""## Optuna para Resnet Mejorada (estudio guardado)

Para la optimización de hiperparámetros del modelo ResNet, se utiliza optuna. Este realiza una búsqueda inteligente para encontrar la combinación de parámetros que maximiza el F1 Score promedio.

Los hiperparámetros explorados son la tasa de aprendizaje (lr), el dropout, el weight_decay, el batch_size y el número de capas a descongelar para el fine-tuning. Para cada combinación, el rendimiento se evalúa de manera robusta mediante Validación Cruzada (K-Fold), asegurando que el resultado no dependa de una única división de datos.

Al finalizar la búsqueda, Optuna entrega la configuración óptima, la cual se utiliza para entrenar el modelo final sobre la totalidad de los datos.

NOTA: Se encontraron varios modelos buenos, por lo que se pausó la búsqueda tempranamente
"""

def objective_kfold(trial):
    # Sugerir hiperparámetros
    unfreeze_layers = trial.suggest_int('unfreeze_layers', 1, 2)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'Adam'])

    # Configuración de K-Fold
    N_SPLITS = 5 # 5 pliegues
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_f1_scores = []

    print(f"\n--- Trial #{trial.number} ---")
    print(f"Params: Unfreeze={unfreeze_layers}, lr={lr:.6f}, bs={batch_size}, dropout={dropout:.4f}, optim={optimizer_name}, wd={weight_decay:.6f}")

    # Bucle de validación cruzada
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"  --- Fold {fold+1}/{N_SPLITS} ---")

        # Crear DataLoaders específicos para este fold
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
        val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2)

        # Crear una nueva instancia del modelo para que cada fold empiece de cero
        model = get_resnet_model(dropout_rate=dropout, unfreeze_layers=unfreeze_layers).to(device)
        optimizer = getattr(optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # Bucle de entrenamiento para este fold
        MAX_EPOCHS = 25
        PATIENCE = 5
        epochs_no_improve = 0
        best_fold_f1 = 0.0

        for epoch in range(MAX_EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_f1 = evaluate_model(model, val_loader, criterion, device, epoch + 1, MAX_EPOCHS)

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
    print(f"  --> Trial finalizado. F1 Promedio en {N_SPLITS} folds: {average_f1:.4f}")
    return average_f1

# _____ CONFIGURACIÓN Y EJECUCIÓN DEL ESTUDIO______
N_TRIALS = 30
STUDY_NAME = "melanoma-resnet-kfold-study"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

study = optuna.create_study(
    direction='maximize',
    study_name=STUDY_NAME,
    storage=STORAGE_NAME,
    load_if_exists=True
)


# Iniciar optimización
print(f"Iniciando/Reanudando estudio '{STUDY_NAME}'. Resultados en '{STUDY_NAME}.db'")
study.optimize(objective_kfold, n_trials=N_TRIALS)

# ------MOSTRAR RESULTADOS ------
print("\n\nBúsqueda finalizada.")
print("Mejor trial:")
trial = study.best_trial
print(f"  F1 Score: {trial.value:.4f}")
print("  Mejores Hiperparámetros: ")
for key, value in trial.params.items():
    print(f"    - {key}: {value}")

"""# 6.- Entrenamiento (mejor modelo guardado)

Se toman los mejores hiperparámetros (lr, dropout, weight_decay, etc.) encontrados por optuna en la búsqueda con Validación Cruzada (K-Fold). Con estos valores, se inicializa una instancia final del modelo ResNet.

Este modelo se entrena sobre el 100% de los datos disponibles (entrenamiento y validación combinados) por un número de épocas fijo y reducido, determinado por el rendimiento promedio en los pliegues. Esta técnica, junto al data augmentation y la regularización, es la principal estrategia ocupada contra el sobreajuste, ya que no se utiliza early stopping.

Finalmente, los pesos del modelo resultante se guardan para generar la predicción definitiva en el conjunto de prueba.
"""

# Mejores hiperparámetros del estudio
best_params = study.best_params

# Entrenamiento del Modelo Final
print("\n--- Entrenando el modelo final sobre el 100% de los datos ---")

# Usar todos los datos combinados
final_train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2)

# Instanciamos el modelo con los mejores parámetros
final_model = get_resnet_model(
    dropout_rate=best_params['dropout'],
    unfreeze_layers=best_params['unfreeze_layers']
).to(device)

# Instanciamos el optimizador y la función de pérdida
optimizer = getattr(optim, best_params['optimizer'])(
    final_model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)
criterion = nn.BCEWithLogitsLoss()

FINAL_EPOCHS = 13
for epoch in range(FINAL_EPOCHS):
    print(f"   --- Época Final {epoch+1}/{FINAL_EPOCHS} ---")
    train_one_epoch(final_model, final_train_loader, optimizer, criterion, device)

# Guardar el modelo final que se usará para la submission y evaluación
torch.save(final_model.state_dict(), 'final_model_kfold.pth')
print("\nModelo final entrenado y guardado en 'final_model_kfold.pth'")

"""# 7.- Evaluación

### Análisis
"""

# --- Reporte del Análisis de Rendimiento Principal ---
print("--- Análisis Completo del Rendimiento del Modelo ---")
print("\nLa métrica principal de evaluación es el F1-Score promedio obtenido durante la Validación Cruzada de 5 pliegues.")
print(f"Este método proporciona una estimación robusta del rendimiento del modelo en datos no vistos.")
print(f"\nEl MEJOR F1-SCORE PROMEDIO alcanzado en la búsqueda fue: {study.best_value:.4f}")

"""### Preparar modelo para evaluación"""

# --- Preparación del Modelo y Datos para Evaluación Detallada ---
print("\n--- Cargando el modelo final para evaluación en un fold representativo ---")

# Instanciar y cargar modelo
eval_model = get_resnet_model(
    dropout_rate=best_params['dropout'],
    unfreeze_layers=best_params['unfreeze_layers']
).to(device)
eval_model.load_state_dict(torch.load('final_model_kfold.pth'))
eval_model.eval() # Poner el modelo en modo de evaluación

# Preparar el DataLoader del fold de validación de ejemlplo
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
# Usar los índices del primer fold como nuestro conjunto de validación representativo
_, val_idx = next(iter(kf.split(full_dataset)))
val_sampler_eval = SubsetRandomSampler(val_idx)
val_loader_eval = torch.utils.data.DataLoader(
    full_dataset, # dataset completo
    batch_size=best_params['batch_size'],
    sampler=val_sampler_eval # Solo tomamos las muestras del fold de validación
)

"""### Funcion TTA

Se aplica Test Time Augmentation, para asegurar la buena clasificación (aplicar data augmentation al momento de evaluar y promediar resultados)
"""

# Funcion para obtener todas las predicciones
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

"""### Predicciones"""

# Obtención de Predicciones y Ajuste de Umbral
print("\nObteniendo predicciones con Test-Time Augmentation (TTA) para el análisis...")
y_true_eval, y_probs_eval = get_all_preds_tta(eval_model, val_loader_eval, device)

# Encontrar el umbral óptimo que maximiza el F1-Score en este fold
precision, recall, thresholds = precision_recall_curve(y_true_eval, y_probs_eval)
# Sumar 1e-9 para evitar divisiones por cero en f1
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
# Encontramos el indice del mejor f1score
best_threshold_idx = np.argmax(f1_scores[:-1])
best_threshold_eval = thresholds[best_threshold_idx]
best_f1_eval = f1_scores[best_threshold_idx]

print(f"\nAJUSTE DE UMBRAL:")
print(f"Se encontró un umbral óptimo de {best_threshold_eval:.4f} que resulta en un F1-Score de {best_f1_eval:.4f} en este fold.")

"""### Predicciones finales"""

# Usar el umbral óptimo para generar las predicciones finales
y_preds_eval = (y_probs_eval >= best_threshold_eval).astype(int)

print("\n--- REPORTE DE MÉTRICAS (Precision, Recall, F1-Score) ---")
print(classification_report(y_true_eval, y_preds_eval, target_names=['No Melanoma (0)', 'Melanoma (1)']))

# Matriz de Confusión y Análisis de Errores
print("\n--- MATRIZ DE CONFUSIÓN ---")
cm = confusion_matrix(y_true_eval, y_preds_eval)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred No-Melanoma', 'Pred Melanoma'],
            yticklabels=['Real No-Melanoma', 'Real Melanoma'])
plt.title('Matriz de Confusión - Fold Representativo (Umbral Óptimo)', fontsize=14)
plt.ylabel('Clase Real', fontsize=12)
plt.xlabel('Clase Predicha', fontsize=12)
plt.show()

# --- Análisis de Errores (Ejemplos mal clasificados) ---
print("\n--- ANÁLISIS DE ERRORES ---")
misclassified_indices_eval = np.where(y_true_eval != y_preds_eval)[0]
print(f"Se encontraron {len(misclassified_indices_eval)} imágenes mal clasificadas en este fold de validación.")
print("A continuación se muestran algunos ejemplos:")

plt.figure(figsize=(15, 5))
for i, idx in enumerate(misclassified_indices_eval[:5]):
    original_idx = val_idx[idx]
    image, label = full_dataset.datasets[0][original_idx]

    # Desnormalizar la imagen para visualización
    image = image.permute(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image.numpy() + mean
    image = np.clip(image, 0, 1)

    ax = plt.subplot(1, 5, i + 1)
    ax.imshow(image)
    ax.set_title(f"Real: {label} | Pred: {y_preds_eval[idx]}")
    ax.axis("off")
plt.show()

"""# 8.- Visualización de resultados

### Gráficas Comparativas de Métricas

Dado que el entrenamiento final se realizó sobre todos los datos juntos, no se tiene una curva de train vs. val para ese modelo. La visualización más representativa es analizar la consistencia del F1-Score en los 5 folds del mejor trial encontrado por optuna, ya que esto justifica por qué se eligió esa configuración de hiperparámetros.

Para obtener estos 5 scores, es necesario re-ejecutar un único entrenamiento del mejor trial
"""

STUDY_NAME = "melanoma-resnet-kfold-study"
STORAGE_NAME = f"sqlite:///{STUDY_NAME}.db"

# cargar estudio
print(f"Cargando los resultados del estudio '{STUDY_NAME}'...")
study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=STORAGE_NAME
)
print("¡Estudio cargado exitosamente!")

# Cargar mejores resultados
best_trial = study.best_trial
best_params = study.best_params

print("--- Analizando la consistencia del mejor trial en los 5 folds ---")

params = study.best_params
fold_scores_viz = []

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
    print(f"  --- Evaluando Fold {fold+1}/5 ---")

    # DataLoaders para este fold
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=params['batch_size'], sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(full_dataset, batch_size=params['batch_size'], sampler=val_sampler)

    # Entrenar un modelo desde cero para este fold con los mejores parámetros
    model_fold = get_resnet_model(dropout_rate=params['dropout'], unfreeze_layers=params['unfreeze_layers']).to(device)
    optimizer_fold = getattr(optim, params['optimizer'])(model_fold.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    best_fold_f1 = 0.0
    epochs_no_improve = 0
    # Entrenamos hasta que se active el Early Stopping
    for epoch in range(25): # Un número máximo de épocas
        train_one_epoch(model_fold, train_loader, optimizer_fold, criterion, device)
        _, val_f1 = evaluate_model(model_fold, val_loader, criterion, device, epoch + 1, 25)
        if val_f1 > best_fold_f1:
            best_fold_f1 = val_f1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= 5:
            break

    fold_scores_viz.append(best_fold_f1)

# Visualizar la distribución de los F1-Scores
plt.figure(figsize=(10, 6))
sns.barplot(x=[f"Fold {i+1}" for i in range(5)], y=fold_scores_viz, palette="viridis")
plt.title('Consistencia del F1-Score en el Mejor Trial (K-Fold)', fontsize=16)
plt.xlabel('Fold de Validación Cruzada', fontsize=12)
plt.ylabel('Mejor F1-Score Obtenido', fontsize=12)
plt.ylim(min(fold_scores_viz) - 0.005, max(fold_scores_viz) + 0.005)
for index, value in enumerate(fold_scores_viz):
    plt.text(index, value, f"{value:.4f}", ha='center', va='bottom', fontsize=10)
plt.show()

"""### Analisis de errores"""

# Cargar el modelo final entrenado en la sección anterior
print("--- Cargando modelo final para visualización de predicciones ---")
viz_model = get_resnet_model(
    dropout_rate=best_params['dropout'],
    unfreeze_layers=best_params['unfreeze_layers']
).to(device)
viz_model.load_state_dict(torch.load('final_model_kfold.pth'))
viz_model.eval()

# Preparar el DataLoader del fold de validación
print("Preparando datos del fold representativo...")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
_, val_idx = next(iter(kf.split(full_dataset)))
val_sampler_viz = SubsetRandomSampler(val_idx)
val_loader_viz = torch.utils.data.DataLoader(full_dataset, batch_size=best_params['batch_size'], sampler=val_sampler_viz)

# Generar las predicciones
print("Generando predicciones en el fold representativo...")
y_true_viz, y_probs_viz = get_all_preds_tta(viz_model, val_loader_viz, device)

# Usamos un umbral de 0.5 para crear las predicciones finales (0 o 1)
y_preds_viz = (y_probs_viz >= 0.5).astype(int)
print("¡Predicciones generadas!")

# Preparar el DataLoader del fold de validación que usaremos como ejemplo
print("Preparando datos del fold representativo...")

# Identificar Índices de Predicciones
correct_indices = np.where(y_true_viz == y_preds_viz)[0]
misclassified_indices = np.where(y_true_viz != y_preds_viz)[0]

def plot_predictions(indices, title):
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

print("\n--- Ejemplos de Predicciones Correctas ---")
plot_predictions(correct_indices, "Predicciones Correctas del Modelo Final")

print("\n--- Ejemplos de Predicciones Incorrectas ---")
plot_predictions(misclassified_indices, "Predicciones Incorrectas del Modelo Final")

"""### Interpretabilidad con Grad-CAM (Mapas de Calor)"""

target_layers = [viz_model.layer4[-1]]
cam = GradCAM(model=viz_model, target_layers=target_layers)

def apply_grad_cam(indices, title):
    plt.figure(figsize=(20, 6))
    plt.suptitle(title, fontsize=16)
    for i, idx in enumerate(indices[:5]):
        original_idx = val_idx[idx]

        tensor_image, label = full_dataset[original_idx]

        if original_idx < len(full_dataset.datasets[0]):
            local_idx = original_idx
            original_image_path, _ = full_dataset.datasets[0].samples[local_idx]
        else:
            local_idx = original_idx - len(full_dataset.datasets[0])
            original_image_path, _ = full_dataset.datasets[1].samples[local_idx]

        rgb_img = np.array(Image.open(original_image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))) / 255.0

        targets = [ClassifierOutputTarget(0)]

        grayscale_cam = cam(input_tensor=tensor_image.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        ax = plt.subplot(1, 5, i + 1)
        ax.imshow(visualization)
        ax.set_title(f"Real: {y_true_viz[idx]} | Pred: {y_preds_viz[idx]}")
        ax.axis("off")
    plt.show()

print("\n--- Visualización con Grad-CAM en Predicciones Correctas ---")
apply_grad_cam(correct_indices, "Grad-CAM - Predicciones Correctas")

print("\n--- Visualización con Grad-CAM en Predicciones Incorrectas ---")
apply_grad_cam(misclassified_indices, "Grad-CAM - Predicciones Incorrectas")

"""# 9.- Generación de submission"""

# Definición del Dataset para el Conjunto de Prueba

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

# Preparar el DataLoader de Test
print("--- Preparando el conjunto de datos de Test ---")
test_csv_path = os.path.join(BASE_DIR, 'test.csv')
test_dataset = TestDataset(root_dir=test_path, csv_file=test_csv_path, transform=valid_transforms)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=best_params['batch_size'], # Usamos el batch_size del mejor trial
    shuffle=False,
    num_workers=0
)

# Cargar el Modelo Final y Hacer Predicciones
print("\nCargando modelo final para la predicción en el conjunto de test...")

# Cargamos la arquitectura ResNet con los mejores hiperparámetros
submission_model = get_resnet_model(
    dropout_rate=best_params['dropout'],
    unfreeze_layers=best_params['unfreeze_layers']
).to(device)

# Cargamos los pesos del modelo que entrenamos con el 100% de los datos
submission_model.load_state_dict(torch.load('final_model_kfold.pth'))
submission_model.eval() # Poner el modelo en modo de evaluación

predictions = []
image_ids = []

FINAL_THRESHOLD = 0.5
print(f"Usando el umbral fijo de {FINAL_THRESHOLD} para las predicciones finales.")

with torch.no_grad():
    for inputs, fnames in tqdm(test_loader, desc="Generando predicciones para Kaggle"):
        inputs = inputs.to(device)

        # Obtener probabilidades del modelo
        outputs = submission_model(inputs)
        # Usamos .squeeze(-1) para que sea robusto contra lotes de tamaño 1
        probs = torch.sigmoid(outputs).squeeze(-1)

        # Código de seguridad para manejar el caso de un solo elemento
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        # Aplicar el umbral que definimos
        preds = (probs.cpu().numpy() >= FINAL_THRESHOLD).astype(int)

        # Guardar predicciones y nombres de archivo
        predictions.extend(preds)
        image_ids.extend(fnames)

# Crear y Guardar el DataFrame de Submission
submission_df = pd.DataFrame({
    'ID': image_ids,
    'predicted': predictions
})

# Guardamos el archivo con un nuevo nombre para reflejar la metodología
submission_df.to_csv('submission.csv', index=False)

print("\nArchivo 'submission.csv' creado exitosamente.")
print("Formato del archivo y primeras 5 filas:")
print(submission_df.head())

"""# Prueba con Data Externa (Kaggel) Para observar rendimiento del modelo

Para revisar como se comporta el modelo con datos externos a los que entrenamos, descargué un dataset desde kaggle. Lamentablemente, el modelo no lo sabe clasificar bien, probablemente debiendose a un sobreajuste.
"""

external_test_path = BASE_DIR / "kaggle_external_data" / "test"
print(f"Cargando dataset externo desde: {external_test_path}")

external_dataset = datasets.ImageFolder(external_test_path, transform=valid_transforms)
print(f"Mapeo de clases del dataset externo: {external_dataset.class_to_idx}")

external_loader = torch.utils.data.DataLoader(external_dataset,
                                              batch_size=best_params['batch_size'],
                                              shuffle=False,
                                              num_workers=0)

# Cargar Mejor Modelo
print("\nCargando tu mejor modelo ResNet entrenado...")

eval_model = get_resnet_model(
    dropout_rate=best_params['dropout'],
    unfreeze_layers=best_params['unfreeze_layers']
).to(device)

eval_model.load_state_dict(torch.load('final_model_kfold.pth'))
eval_model.eval()

# Obtener Predicciones
print("\nObteniendo predicciones con Test-Time Augmentation (TTA)...")
external_y_true, external_y_probs = get_all_preds_tta(eval_model, external_loader, device)

# Aplicar un Umbral Pre-definido
FINAL_THRESHOLD = 0.65
print(f"Usando el umbral fijo y pre-definido de {FINAL_THRESHOLD} para la evaluación.")

# Generar las predicciones finales con este umbral
external_y_preds = (external_y_probs >= FINAL_THRESHOLD).astype(int)

# Reporte Final de Rendimiento
print("\n-------- Reporte de Clasificación en Dataset Externo ---------")
# Calcular el F1-Score final con las predicciones
final_f1_score = f1_score(external_y_true, external_y_preds)
print(f"F1-Score Final en el Dataset Externo: {final_f1_score:.4f}")
print(classification_report(external_y_true, external_y_preds, target_names=['Benign (0)', 'Malign (1)']))

print("\n-------- Matriz de Confusión en Dataset Externo --------")
cm = confusion_matrix(external_y_true, external_y_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Benign', 'Pred Malign'],
            yticklabels=['Real Benign', 'Real Malign'])
plt.title('Matriz de Confusión - Dataset Externo', fontsize=14)
plt.ylabel('Clase Real'); plt.xlabel('Clase Predicha'); plt.show()
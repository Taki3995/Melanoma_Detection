# -*- coding: utf-8 -*-
"""
run_eda.py

Script para ejecutar el Análisis Exploratorio de Datos (Sección 2).

"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Importar constantes y utilidades
import config
from utils import list_images, show_examples

def main():
    print("=============================== EDA ===============================")
    
    # Rutas Archivos
    train_mel_path   = config.BASE_DIR / "train" / "mel"
    train_nomel_path = config.BASE_DIR / "train" / "nomel"
    val_mel_path     = config.BASE_DIR / "valid" / "mel"
    val_nomel_path   = config.BASE_DIR / "valid" / "nomel"
    test_path        = config.BASE_DIR / "test"
    test_csv = pd.read_csv(config.TEST_CSV_PATH)

    # Conteo
    train_mel_imgs = list_images(train_mel_path)
    train_nomel_imgs = list_images(train_nomel_path)
    val_mel_imgs = list_images(val_mel_path)
    val_nomel_imgs = list_images(val_nomel_path)
    test_imgs = list_images(test_path)


    # Resumen

    print("---------- Estructura del dataset: ----------")
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

if __name__ == "__main__":
    main()
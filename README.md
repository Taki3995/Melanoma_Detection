# Clasificación de Melanomas
Este proyecto fue desarrollado para el curso "Optimización 2" en la Pontificia Universidad Católica de Chile. El problema dado es una clasificacion binaria de imágenes 
específicamente la detección de Melanoma (1) frente a No Melanoma (0) a partir de fotografías clínicas de lesiones en la piel. El objetivo es desarrollar un modelo capaz 
de clasificar imágenes médicas con la mayor precisión posible, optimizando hiperparámetros mediante algoritmos de optimización y evaluando el desempeño con la métrica F1 Score.
La clasificación automática de melanoma es una aplicación relevante del aprendizaje automático con un fuerte impacto en la salud pública, ya que la detección temprana puede 
mejorar significativamente el pronóstico y tratamiento de los pacientes.

## Descripción del Desafío
La tarea consiste en **clasificar imágenes médicas de lesiones cutáneas** en dos categorías:  
- **Melanoma (1)**  
- **No Melanoma (0)**  

Se trata de un problema de **clasificación binaria**, donde el objetivo es detectar correctamente los casos positivos (melanoma) sin perder precisión en los negativos.  

El rendimiento de los modelos será evaluado utilizando la métrica **F1 Score**, que equilibra precisión (*precision*) y exhaustividad (*recall*), y resulta especialmente útil en contextos donde puede existir desbalance de clases.  

## Estructura del Jupyter Notebook

1. **Introducción y objetivo**  
   Breve contexto del problema de melanoma y el objetivo del laboratorio: entrenar un clasificador binario (Melanoma / No Melanoma) y optimizar hiperparámetros para maximizar el **F1 Score**.

2. **Explicación de los datos**
   - Descripción la estructura del dataset (`train/`, `val/`, `test/`).  
   - Conteo de imágenes por clase (Mel / NoMel).  
   - Visualización ejemplos de imágenes de ambas categorías.  
   - Análisis posibles desbalances de clases.

4. **Preprocesamiento**  
   - Redimensionamiento, normalización y preparación de las imágenes.  
   - Uso de *data augmentation* 
   - Manejo de semillas para asegurar reproducibilidad.

5. **Modelos a probar**  
   - Descripción detallada de los diferentes modelos que se explorararon (CNN desde cero y *transfer learning* con ResNet).  
   - Justificación de por qué cada modelo es apropiado para la tarea de clasificación binaria.

6. **Optimización de hiperparámetros**  
   - Estrategia de búsqueda (búsqueda bayesiana, Optuna).  
   - Hiperparámetros considerados (learning rate, batch size, número de épocas, regularización, dropout, capas congeladas, etc.).  
   - Explicación de cómo se seleccionan los mejores hiperparámetros usando el conjunto de validación.

7. **Entrenamiento**  
   - Descripción del proceso de entrenamiento del modelo seleccionado.  
   - Detalles de la función de pérdida, optimizador, *scheduler*, técnicas contra sobreajuste (*early stopping*, regularización).  
   - Evolución del entrenamiento (curvas de loss y métricas en train/val).

8. **Evaluación**  CORREGIR SEGUN ELEGIDO
   - Reportar análisis completo del rendimiento de los modelos sobre el conjunto de validación.  
   - Métricas: **F1 Score**, *precision* y *recall*.  
   - Matriz de confusión y análisis de errores (ejemplos mal clasificados).  
   - (Opcional) Ajuste de umbral de decisión para mejorar el F1.

9. **Visualización de resultados**  CORREGIR SEGUN ELEGIDO
   - Gráficas comparativas de las métricas (train vs val).  
   - Visualización de predicciones correctas e incorrectas.  
   - (Opcional) Interpretabilidad con Grad-CAM o mapas de calor para observar qué partes de la imagen influyen más en la predicción.

## Notas

Con la primera versión del código (usando resnet) se llego a un f1-score de 0.98095, por lo que se implementó TTA, lo que dio una mejora. El resultado fue de 0.98113, que aunque fue poco, mejoró. 

En una segunda versión se implementó EfficientNet y se hizo un Data Augmentation más robusto.

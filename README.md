# Instrucciones´
Este trabajo se puede revisar / correr tanto desde el archivo "melanoma_detection.ipynb" como corriendo el archivo "main.py" de la carpeta "Python Code". Ambos tienen lo mismo, pero desde el Jupyter Notebook se puede visualizar las celdas ya ejecutadas. 

Están guardados en la carpeta "outputs" el estudio de hiperparámetros, el mejor modelo y el submission, por lo que no es necesario correr la búsqueda de hiperparámetros y de mejor modelo nuevamente.

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


# Bitácora de Experimentación y Descubrimientos del Proyecto

El desarrollo de este proyecto consistió en un proceso iterativo de experimentación para optimizar la clasificación binaria de imágenes de melanoma, utilizando el F1-Score como métrica principal. A continuación, se detallan los hallazgos clave y la evolución de la metodología.

### 1. Línea Base Inicial: ResNet18
El punto de partida fue la implementación de una arquitectura ResNet18 pre-entrenada, utilizando únicamente *transfer learning* (congelando el *backbone* y entrenando solo el clasificador).
* **Resultado:** Se alcanzó un F1-Score notable de **0.98095**.
* **Mejora con TTA:** Se introdujo la técnica de *Test Time Augmentation* (TTA), promediando las predicciones de la imagen original y su volteo horizontal. Esto resultó en una mejora marginal pero consistente, elevando el F1-Score a **0.98113**.

### 2. Pivote Estratégico: EfficientNet-B2
Buscando superar la alta línea base, la segunda fase del proyecto se centró en una arquitectura más moderna y potente, EfficientNet-B2. La hipótesis era que un modelo más avanzado, combinado con un *Data Augmentation* más robusto (incluyendo rotaciones, cambios de color y *Random Erasing*), capturaría características más complejas.
* **Resultado Inesperado:** La implementación inicial, utilizando solo el optimizador AdamW, tuvo un rendimiento inferior al de ResNet18. El F1-Score decayó significativamente a aproximadamente **0.97**.
* **Ajuste (Data Augmentation):** Se teorizó que la combinación de un modelo complejo y un *Data Augmentation* agresivo dificultaba el aprendizaje. Al suavizar el *Data Augmentation* (haciéndolo menos complejo), el rendimiento de EfficientNet mejoró, pero aún se mantenía por debajo del F1-Score obtenido con ResNet18.

### 3. Introducción del Fine-Tuning y Detección de Sobreajuste
El siguiente paso fue explorar el *Fine-Tuning* (descongelamiento de las últimas capas convolucionales) en la arquitectura EfficientNet.
* **Falso Positivo:** Esta técnica arrojó resultados inicialmente extraordinarios, alcanzando un **F1-Score perfecto de 1.0** en el conjunto de validación.
* **Descubrimiento Crítico:** Un análisis más profundo reveló que este resultado no era producto de la generalización, sino de un **severo sobreajuste** (*overfitting*). El modelo había memorizado eficazmente el conjunto de validación, un riesgo exacerbado por la alta capacidad del modelo EfficientNet frente a un conjunto de datos limitado.

### 4. Síntesis y Modelo Final: ResNet18 con K-Fold y Fine-Tuning
Habiendo identificado el sobreajuste como el principal adversario, el proyecto retornó estratégicamente a la arquitectura ResNet18, que había demostrado ser más estable. Esta implementación final no fue una simple regresión, sino una síntesis de todas las lecciones aprendidas:
* **Arquitectura Estable:** Se utilizó **ResNet18**.
* **Fine-Tuning Controlado:** Se descongelaron las últimas capas convolucionales (`layer3` y `layer4`) para permitir que el modelo ajustara sus detectores de características de alto nivel.
* **Data Augmentation Robusto:** Se mantuvo el *Data Augmentation* más complejo implementado en la fase 2.
* **Validación Robusta (K-Fold):** Para combatir el sobreajuste a un único conjunto de validación, se implementó una **Validación Cruzada de 5 Pliegues (K-Fold)**. La optimización de hiperparámetros (Optuna) se realizó promediando el F1-Score de los 5 pliegues, forzando al modelo a generalizar.
* **Resultado:** Este enfoque híbrido demostró ser el más exitoso, generando el modelo con el mejor rendimiento promedio y la mayor robustez en los datos de entrenamiento y validación combinados.

### 5. Conclusión y Análisis de Generalización
El modelo final, si bien fue el de mejor desempeño en el set de datos proporcionado, exhibió dificultades al ser evaluado contra un **conjunto de datos externo** (de Kaggle), donde su rendimiento disminuyó significativamente.
Este hallazgo final subraya una conclusión clave: el conjunto de datos original, aunque balanceado, es probablemente **limitado en su diversidad**. Las imágenes pueden carecer de variabilidad en iluminación, calidad, ruido o artefactos presentes en datos "salvajes". Como resultado, el modelo desarrolló un sobreajuste al *dominio* específico de los datos de entrenamiento, limitando su generalización en escenarios de inferencia completamente nuevos.
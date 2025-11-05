# -*- coding: utf-8 -*-
"""
main.py

Script principal para ejecutar el pipeline completo de
detección de melanoma en el orden correcto.
"""

import subprocess
import sys
import time

def run_script(script_name):
    """
    Función de ayuda para ejecutar un script de Python y
    manejar errores.
    """
    # Usar sys.executable asegura que se use el mismo intérprete de Python que está ejecutando este script
    python_executable = sys.executable
    
    print(f"\n--- Iniciando: [ {script_name} ] ---")
    start_time = time.time()
    
    try:
        # subprocess.run es la forma moderna y recomendada.
        # check=True: Lanza un error si el script falla.
        # capture_output=True: Atrapa lo que el script imprima.
        # text=True: Decodifica la salida como texto (en lugar de bytes).
        result = subprocess.run(
            [python_executable, script_name],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8' # Forzar utf-8 para los caracteres especiales
        )
        
        end_time = time.time()
        print(f"--- Finalizado: [ {script_name} ] (Duración: {end_time - start_time:.2f}s) ---")
        
        # Imprimir la salida estándar del script (útil para ver logs)
        if result.stdout:
            print("--- Salida del script (stdout): ---")
            print(result.stdout)

        # Imprimir si hubo alguna salida de error (warnings, etc.)
        if result.stderr:
            print("--- Salida de error (stderr): ---")
            print(result.stderr)
            
        return True # Indicar éxito

    except subprocess.CalledProcessError as e:
        # Esto se activa si check=True y el script retorna un código de error
        end_time = time.time()
        print(f"--- ERROR: [ {script_name} ] falló (Duración: {end_time - start_time:.2f}s) ---")
        print("--- Salida del script (stdout): ---")
        print(e.stdout)
        print("--- Error (stderr): ---")
        print(e.stderr)
        return False # Indicar fallo
    
    except FileNotFoundError:
        print(f"--- ERROR: No se pudo encontrar el script [ {script_name} ] ---")
        print("Asegúrate de que el archivo existe en el mismo directorio.")
        return False # Indicar fallo

if __name__ == "__main__":
    
    # --- Definir el orden del pipeline ---
    pipeline_scripts = [
        "run_eda.py",              # 1. Análisis Exploratorio
        # "run_optuna.py",           # 2. Búsqueda de Hiperparámetros
        "run_training.py",         # 3. Entrenamiento del modelo final
        "run_evaluation.py",       # 4. Evaluación del modelo
        "run_visualization.py",    # 5. Visualización y Grad-CAM
        # "run_submission.py",       # 6. Generar submission.csv
        "run_external_test.py"     # 7. Probar con datos externos
    ]
    
    print("=================================================")
    print("====== INICIANDO PIPELINE DE MELANOMA ======")
    print("=================================================")
    
    total_start_time = time.time()
    
    for script in pipeline_scripts:
        success = run_script(script)
        
        if not success:
            print(f"\n--- PIPELINE DETENIDO DEBIDO A UN ERROR EN [ {script} ] ---")
            break # Detiene la ejecución del resto del pipeline
    else:
        # Este 'else' se ejecuta solo si el bucle 'for' termina sin 'break'
        print("\n===================================================")
        print("====== PIPELINE COMPLETADO EXITOSAMENTE ======")
        print(f"Duración total: {time.time() - total_start_time:.2f}s")
        print("===================================================")
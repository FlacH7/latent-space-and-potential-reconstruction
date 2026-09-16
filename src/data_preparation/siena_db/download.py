#!/usr/bin/env python3
"""
Descarga el dataset Siena Scalp EEG de PhysioNet usando wget.
Uso: python descargar_siena.py /ruta/al/directorio/destino
"""

import argparse
import os
import subprocess
import sys
from src.utils.config import DB_SIENA_PATH


def descargar_siena(directorio_destino):
    """
    Ejecuta wget para descargar el dataset de Siena Scalp EEG
    en el directorio especificado.
    """
    # Crear el directorio si no existe
    if not os.path.exists(directorio_destino):
        print(f"Creando directorio: {directorio_destino}")
        os.makedirs(directorio_destino, exist_ok=True)

    # Verificar que sea un directorio válido
    if not os.path.isdir(directorio_destino):
        print(f"Error: '{directorio_destino}' no es un directorio válido.", file=sys.stderr)
        sys.exit(1)

    # Comando wget con los parámetros solicitados
    # -P: prefijo del directorio de destino
    comando = [
        "wget",
        "-r",           # Recursivo
        "-N",           # Solo descargar si el archivo remoto es más nuevo
        "-c",           # Continuar descargas parciales
        "-np",          # No subir al directorio padre
        "-P", directorio_destino,  # Directorio de destino
        "https://physionet.org/files/siena-scalp-eeg/1.0.0/"
    ]

    print(f"Iniciando descarga en: {os.path.abspath(directorio_destino)}")
    print(f"Ejecutando: {' '.join(comando)}")
    print("-" * 60)

    try:
        # Ejecutar el comando y mostrar la salida en tiempo real
        resultado = subprocess.run(
            comando,
            check=True,
            text=True
        )
        print("-" * 60)
        print("Descarga completada exitosamente.")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: La descarga falló con código de salida {e.returncode}.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: 'wget' no está instalado o no se encuentra en el PATH.", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Descarga el dataset Siena Scalp EEG de PhysioNet."
    )
    parser.add_argument(
        "--directorio",
        help="Directorio de destino donde se descargarán los archivos.",
        default= DB_SIENA_PATH
    )
    
    args = parser.parse_args()
    descargar_siena(args.directorio)


if __name__ == "__main__":
    main()
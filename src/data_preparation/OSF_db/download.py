#!/usr/bin/env python3
"""
download_anphy_sleep.py
Descarga automatizada de ANPHY-Sleep desde OSF.
No requiere registro ni token (proyecto público).
"""

import os
import subprocess
import sys
from src.utils.config import DB_ANPHY_PATH


def download_anphy_sleep(output_dir: str = DB_ANPHY_PATH):
    """
    Descarga la base de datos ANPHY-Sleep desde OSF usando osfclient.
    Proyecto ID: R26FH (https://doi.org/10.17605/OSF.IO/R26FH)
    """
    output_dir = os.path.abspath(output_dir)
    
    if os.path.exists(output_dir) and any(os.scandir(output_dir)):
        print(f"[INFO] El directorio {output_dir} ya existe y tiene contenido.")
        response = input("¿Deseas re-descargar? (s/N): ").strip().lower()
        if response != 's':
            print("Descarga cancelada.")
            return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Verificar si osfclient está instalado
    try:
        subprocess.run(["osf", "-h"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[INFO] osfclient no encontrado. Instalando...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "osfclient"])

    # Descargar todo el proyecto R26FH
    # osf clone descarga recursivamente todos los archivos
    print(f"\n[DOWNLOAD] Descargando ANPHY-Sleep en: {output_dir}")
    print("[DOWNLOAD] Esto puede tardar varios minutos (dataset grande)...")
    
    cmd = ["osf", "-p", "R26FH", "clone", output_dir]
    subprocess.run(cmd, check=True)
    
    print(f"\n[OK] Descarga completada en: {output_dir}")
    return output_dir


def verify_download(base_dir: str):
    """Verifica estructura esperada: subfolders por sujeto con .edf y anotaciones."""
    expected_subjects = 29
    subfolders = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("sub-")]
    
    print(f"\n[VERIFY] Sujetos encontrados: {len(subfolders)} (esperados: {expected_subjects})")
    
    edf_count = 0
    for sub in subfolders:
        sub_path = os.path.join(base_dir, sub)
        files = os.listdir(sub_path)
        edfs = [f for f in files if f.lower().endswith(".edf")]
        edf_count += len(edfs)
        if edfs:
            print(f"  {sub}: {len(edfs)} archivo(s) .edf")
    
    print(f"[VERIFY] Total archivos .edf: {edf_count}")
    return len(subfolders) == expected_subjects


if __name__ == "__main__":
    out = download_anphy_sleep()
    verify_download(out)
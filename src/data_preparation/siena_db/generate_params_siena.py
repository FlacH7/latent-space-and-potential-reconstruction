#!/usr/bin/env python3
"""
generate_batch_params.py
========================
Lee los archivos ``Seizures-list-PNXX.txt`` de cada paciente, parsea los
tiempos de registro y seizures, y genera un archivo de texto con listas
de parámetros válidos para el pipeline de IgA.

Cada línea del archivo de salida es una lista Python:
    ['PNXX', 'recording', 't_start_min', 't_end_min', 'label']

Reglas de ventanas:
    - Sin solapamiento con seizures.
    - Continuas dentro de cada segmento libre.
    - Preferiblemente 60 min; si no cabe, 30 min; si no, se descarta.
    - Etiqueta: pre-ictal / inter-ictal / post-ictal.

Uso:
    python generate_batch_params.py /path/to/siena_scalp_eeg/1.0.0
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from datetime import datetime

from src.utils.config import DB_SIENA_PATH, BASE_PARAMS_FILE


# =============================================================================
# PARSING DE TIEMPOS
# =============================================================================

TIME_RE = re.compile(r'(\d{1,2})[\.:](\d{2})[\.:](\d{2})')

def extract_times_from_line(line: str) -> list[int]:
    """
    Extrae todos los tiempos HH:MM:SS o HH.MM.SS de una línea y
    devuelve una lista de segundos desde la medianoche.
    """
    matches = TIME_RE.findall(line)
    return [int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in matches]


def parse_seizure_txt(filepath: Path) -> list[dict]:
    """
    Parsea un archivo Seizures-list-PNXX.txt y devuelve una lista de
    registros (recordings), cada uno con:
        - file_name: str
        - reg_start: int (segundos desde medianoche)
        - reg_end:   int (segundos desde medianoche, puede > 86400)
        - seizures:  list[dict] con 'start' y 'end' en segundos
    """
    recordings = []
    current = None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # --- Nuevo archivo / recording ---
        if line.lower().startswith("file name"):
            # Guardar el anterior si existe
            if current is not None:
                recordings.append(current)

            fname = line.split(":", 1)[1].strip()
            current = {
                "file_name": fname,
                "reg_start": None,
                "reg_end": None,
                "seizures": [],
            }
            i += 1
            continue

        # --- Tiempos de registro ---
        if current is not None and "registration start time" in line.lower():
            times = extract_times_from_line(line)
            if times:
                current["reg_start"] = times[0]
            i += 1
            continue

        if current is not None and "registration end time" in line.lower():
            times = extract_times_from_line(line)
            if times:
                current["reg_end"] = times[0]
            i += 1
            continue

        # --- Seizure start / end ---
        # Detectamos líneas que contengan "start time" o "end time" y
        # que NO sean las de registration.
        if current is not None and "start time" in line.lower() and "registration" not in line.lower():
            times = extract_times_from_line(line)
            if times:
                # Si ya tenemos un seizure pendiente sin end, lo cerramos
                # con el tiempo más temprano (no debería pasar, pero por robustez)
                if current["seizures"] and current["seizures"][-1].get("end") is None:
                    current["seizures"][-1]["start"] = min(times)
                else:
                    current["seizures"].append({"start": min(times), "end": None})
            i += 1
            continue

        if current is not None and "end time" in line.lower() and "registration" not in line.lower():
            times = extract_times_from_line(line)
            if times and current["seizures"] and current["seizures"][-1].get("end") is None:
                # Tomamos el tiempo más tardío para ser conservadores
                current["seizures"][-1]["end"] = max(times)
            i += 1
            continue

        i += 1

    # Guardar el último bloque
    if current is not None:
        recordings.append(current)

    # Normalizar medianoche y limpiar seizures incompletos
    out = []
    for rec in recordings:
        if rec["reg_start"] is None or rec["reg_end"] is None:
            print(f"  [WARN] {filepath.name}: {rec['file_name']} sin tiempos de registro. Omitido.")
            continue

        rs = rec["reg_start"]
        re = rec["reg_end"]

        # Si el registro cruza la medianoche
        if re < rs:
            re += 24 * 3600

        clean_seizures = []
        for sz in rec["seizures"]:
            if sz["start"] is None or sz["end"] is None:
                continue
            s = sz["start"]
            e = sz["end"]
            # Ajustar si cruzan la medianoche respecto al inicio del registro
            if s < rs:
                s += 24 * 3600
            if e < rs:
                e += 24 * 3600
            # Asegurar end > start
            if e < s:
                e += 24 * 3600
            clean_seizures.append({"start": s, "end": e})

        clean_seizures.sort(key=lambda x: x["start"])
        out.append({
            "file_name": rec["file_name"],
            "reg_start": rs,
            "reg_end": re,
            "seizures": clean_seizures,
        })

    return out


# =============================================================================
# GENERACIÓN DE VENTANAS
# =============================================================================

def build_windows(rec: dict) -> list[tuple[float, float, str]]:
    """
    Dado un registro con tiempos absolutos en segundos, genera las
    ventanas válidas en minutos relativos al inicio del registro.
    Devuelve lista de (t_start_min, t_end_min, label).
    """
    rs = rec["reg_start"]
    re = rec["reg_end"]
    total_min = (re - rs) / 60.0

    seizures_rel = [
        {"start": (sz["start"] - rs) / 60.0, "end": (sz["end"] - rs) / 60.0}
        for sz in rec["seizures"]
    ]

    # Segmentos libres: (start_min, end_min)
    free_segments = []
    if not seizures_rel:
        free_segments.append((0.0, total_min))
    else:
        # Antes del primer seizure
        first_sz = seizures_rel[0]
        if first_sz["start"] > 0:
            free_segments.append((0.0, first_sz["start"]))

        # Entre seizures
        for i in range(len(seizures_rel) - 1):
            a = seizures_rel[i]["end"]
            b = seizures_rel[i + 1]["start"]
            if b > a:
                free_segments.append((a, b))

        # Después del último seizure
        last_sz = seizures_rel[-1]
        if total_min > last_sz["end"]:
            free_segments.append((last_sz["end"], total_min))

    windows = []
    for seg_start, seg_end in free_segments:
        duration = seg_end - seg_start
        if duration < 30.0:
            continue

        # Etiqueta
        if seg_start == 0.0 and len(seizures_rel) > 0:
            label = "pre-ictal"
        elif seg_end == total_min and len(seizures_rel) > 0:
            label = "post-ictal"
        else:
            label = "inter-ictal"

        # Generar ventanas continuas de 60 min, luego 30 min si sobra
        pos = seg_start
        while True:
            remaining = seg_end - pos
            if remaining >= 60.0:
                windows.append((pos, pos + 60.0, label))
                pos += 60.0
            elif remaining >= 30.0:
                windows.append((pos, pos + 30.0, label))
                break
            else:
                break

    return windows


def extract_recording(file_name: str) -> str:
    """Extrae el identificador de recording del nombre de archivo EDF."""
    stem = Path(file_name).stem  # quita .edf
    if '-' in stem:
        return stem.split('-', 1)[1]
    return "1"


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera parámetros de batch a partir de archivos Seizures-list-PNXX.txt"
    )
    parser.add_argument("--base_dir", type=str, default= DB_SIENA_PATH,
                        help="Carpeta base que contiene las subcarpetas PNXX")
    parser.add_argument("--out", type=str, default=BASE_PARAMS_FILE + "/batch_params_siena.txt",
                        help="Archivo de salida con las listas (default: batch_params.txt)")
    parser.add_argument("--csv", type=str, default=BASE_PARAMS_FILE + "/batch_params_siena.csv",
                        help="Archivo CSV adicional (default: batch_params.csv)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"[ERROR] No existe el directorio: {base_dir}")
        return 1

    out_txt = Path(args.out)
    out_csv = Path(args.csv)

    all_params = []

    # Recorrer subcarpetas PN*
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if not subdir.name.upper().startswith("PN"):
            continue

        patient = subdir.name  # ej. "PN01"
        txt_file = subdir / f"Seizures-list-{patient}.txt"

        if not txt_file.exists():
            print(f"  [SKIP] {patient}: no encontrado {txt_file.name}")
            continue

        print(f"  [PARSING] {patient} ...")
        recordings = parse_seizure_txt(txt_file)

        for rec in recordings:
            recording = extract_recording(rec["file_name"])
            windows = build_windows(rec)

            for t_start, t_end, label in windows:
                all_params.append([
                    patient,
                    recording,
                    f"{t_start:.2f}",
                    f"{t_end:.2f}",
                    label,
                ])

    # Guardar TXT (listas de Python)
    with open(out_txt, 'w', encoding='utf-8') as f:
        for row in all_params:
            f.write(str(row) + "\n")

    # Guardar CSV
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "recording", "t_start_min", "t_end_min", "label"])
        writer.writerows(all_params)

    print(f"\n{'='*60}")
    print(f"  Total de combinaciones generadas: {len(all_params)}")
    print(f"  TXT  : {out_txt.absolute()}")
    print(f"  CSV  : {out_csv.absolute()}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
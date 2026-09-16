#!/usr/bin/env python3
"""
generate_batch_params_anphy.py
==============================
Lee los hipnogramas (.txt) y matrices de artefactos (.mat v7.3) de
ANPHY-Sleep, filtra epochs por canales válidos de Siena, y genera
batch_params_anphy.txt con las especificaciones de cada epoch a analizar.

**NUEVO**: Agrupa epochs consecutivos de la misma etiqueta en bloques.
Solo se generan bloques de ≥ 10 epochs consecutivos.

Cada línea del TXT es una lista Python:
    ['subject', 'block_id', 't_start_sec', 't_end_sec', 'stage_label']

Uso:
    python generate_batch_params_anphy.py /path/to/ANPHY-Sleep/osfstorage
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import numpy as np

from src.utils.config import DB_ANPHY_PATH, BASE_PARAMS_FILE

try:
    import h5py
except ImportError:
    print("[ERROR] h5py no está instalado. Ejecuta: pip install h5py")
    sys.exit(1)

try:
    import mne
except ImportError:
    print("[ERROR] mne no está instalado. Ejecuta: pip install mne")
    sys.exit(1)


# =============================================================================
# CANALES VÁLIDOS (mismos que Siena)
# =============================================================================
_SIENA_VALID_EEG_CHANNELS: frozenset[str] = frozenset({
    "FP1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
     "FC1","FC5","CP1","CP5","F9",
     "FZ", "CZ", "PZ",
     "FP2", "F4", "C4", "P4","O2", "F8", "T4", "T6",
     "FC2","FC6","CP2","CP6","F10"
})

# Canales que NO son EEG (EOG, EMG, ECG, etc.)
_NON_EEG_PATTERNS = frozenset({
    "EOG", "EMG", "ECG", "EKG", "RESP", "TRIG", "STATUS", "EVENT",
    "MK", "MISC", "SAO", "PLETH", "TEMP", "GSR", "EDA", "BREATH",
})


def _norm_ch(name: str) -> str:
    """Normaliza nombre de canal para comparación."""
    return name.strip().upper().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")


def _is_eeg_channel(name: str) -> bool:
    """Determina si un canal parece ser EEG (no EOG, EMG, ECG, etc.)."""
    norm = _norm_ch(name)
    for pat in _NON_EEG_PATTERNS:
        if pat in norm:
            return False
    return True


def load_artndxn(mat_path: Path) -> np.ndarray:
    """Carga artndxn desde .mat v7.3 (HDF5). h5py lee column-major de MATLAB."""
    with h5py.File(mat_path, "r") as f:
        arr = np.array(f["artndxn"])
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return arr


def get_eeg_channel_mapping(edf_path: Path) -> tuple[list[int], list[str], list[int], list[str]]:
    """
    Carga el EDF (solo headers) y devuelve:
        - eeg_indices: índices DENTRO del EDF de los canales EEG
        - eeg_names: nombres originales de esos canales EEG
        - valid_in_eeg: índices DENTRO del subconjunto EEG de los canales Siena
        - valid_names: nombres de los canales Siena válidos
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")

    eeg_indices = []
    eeg_names = []
    for idx, ch_name in enumerate(raw.ch_names):
        if _is_eeg_channel(ch_name):
            eeg_indices.append(idx)
            eeg_names.append(ch_name)

    n_eeg = len(eeg_indices)
    print(f"    → {n_eeg} canales EEG detectados en EDF (de {len(raw.ch_names)} total)")

    valid_in_eeg = []
    valid_names = []
    for i, ch_name in enumerate(eeg_names):
        if _norm_ch(ch_name) in _SIENA_VALID_EEG_CHANNELS:
            valid_in_eeg.append(i)
            valid_names.append(ch_name)

    return eeg_indices, eeg_names, valid_in_eeg, valid_names


def parse_hypnogram(txt_path: Path) -> list[tuple[str, float, float]]:
    """
    Parsea el hipnograma .txt de ANPHY-Sleep.
    Cada línea:  <STAGE>  <START_SEC>  <DURATION_SEC>
    Devuelve lista de (stage, start_sec, duration_sec).
    """
    epochs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                stage = parts[0]
                start = float(parts[1])
                duration = float(parts[2])
                epochs.append((stage, start, duration))
    return epochs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera batch_params para ANPHY-Sleep (bloques ≥ 10 epochs)"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DB_ANPHY_PATH + "/osfstorage",
        help="Carpeta base que contiene EPCTLXX/, Artifact matrix/, etc.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=BASE_PARAMS_FILE + "/batch_params_anphy.txt",
        help="Archivo de salida TXT (listas Python)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=BASE_PARAMS_FILE + "/batch_params_anphy.csv",
        help="Archivo de salida CSV",
    )
    parser.add_argument(
        "--min-clean-ratio",
        type=float,
        default=0.5,
        help="Ratio mínimo de canales Siena limpios para incluir el epoch "
             "(0.0 = incluir todos). Default: 0.5",
    )
    parser.add_argument(
        "--min-block-size",
        type=int,
        default=10,
        help="Mínimo número de epochs consecutivos con la misma etiqueta "
             "para generar un bloque. Default: 10",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        print(f"[ERROR] No existe el directorio: {base_dir}")
        return 1

    artifact_dir = base_dir / "Artifact matrix"
    if not artifact_dir.is_dir():
        print(f"[ERROR] No existe Artifact matrix en: {base_dir}")
        return 1

    out_txt = Path(args.out)
    out_csv = Path(args.csv)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_params = []

    # -----------------------------------------------------------------
    # Recorrer sujetos EPCTL01 .. EPCTL29
    # -----------------------------------------------------------------
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        name = subdir.name
        if not name.startswith("EPCTL"):
            continue

        subject = name
        print(f"  [PARSING] {subject} ...")

        # --- EDF ---
        edf_candidates = list(subdir.glob("*.edf"))
        if not edf_candidates:
            print(f"  [SKIP] {subject}: no se encontró .edf")
            continue
        edf_path = edf_candidates[0]

        # --- TXT hipnograma ---
        txt_candidates = list(subdir.glob("*.txt"))
        if not txt_candidates:
            print(f"  [SKIP] {subject}: no se encontró .txt")
            continue
        txt_path = txt_candidates[0]

        # --- MAT artefactos (hay typos en los nombres originales) ---
        mat_candidates = [
            artifact_dir / f"{subject}_artndxn.mat",
            artifact_dir / f"{subject}_artndex.mat",
            artifact_dir / f"{subject}_artdnex.mat",
        ]
        mat_candidates += [
            artifact_dir / f"{subject.lower()}_artndxn.mat",
            artifact_dir / f"{subject.lower()}_artndex.mat",
            artifact_dir / f"{subject.lower()}_artdnex.mat",
        ]
        mat_path = None
        for cand in mat_candidates:
            if cand.exists():
                mat_path = cand
                break

        if mat_path is None:
            print(f"  [SKIP] {subject}: no se encontró .mat de artefactos")
            continue

        # --- Mapeo de canales EEG → artndxn ---
        try:
            eeg_indices, eeg_names, valid_in_eeg, valid_names = get_eeg_channel_mapping(edf_path)
        except Exception as e:
            print(f"  [ERROR] {subject}: fallo leyendo EDF: {e}")
            continue

        n_eeg = len(eeg_indices)
        if n_eeg == 0:
            print(f"  [WARN] {subject}: ningún canal EEG detectado")
            continue

        if not valid_in_eeg:
            print(f"  [WARN] {subject}: ningún canal EEG coincide con la lista de Siena")
            continue

        print(f"    → {len(valid_in_eeg)} canales válidos de Siena entre los {n_eeg} EEG")

        # --- Cargar hipnograma y matriz ---
        try:
            epochs = parse_hypnogram(txt_path)
            artndxn = load_artndxn(mat_path)
        except Exception as e:
            print(f"  [ERROR] {subject}: fallo cargando anotaciones/artefactos: {e}")
            continue

        n_epochs_txt = len(epochs)
        n_epochs_mat = artndxn.shape[0]
        n_cols_mat = artndxn.shape[1]

        if n_epochs_txt != n_epochs_mat:
            print(f"    → mismatch txt={n_epochs_txt} mat={n_epochs_mat}; usando min={min(n_epochs_txt, n_epochs_mat)}")
        n_epochs = min(n_epochs_txt, n_epochs_mat)

        if any(idx >= n_cols_mat for idx in valid_in_eeg):
            print(f"  [WARN] {subject}: algunos índices válidos ({max(valid_in_eeg)}) exceden "
                  f"las columnas de artndxn ({n_cols_mat}). Recortando...")
            valid_in_eeg = [idx for idx in valid_in_eeg if idx < n_cols_mat]
            valid_names = [valid_names[i] for i in range(len(valid_in_eeg))]
            if not valid_in_eeg:
                print(f"  [SKIP] {subject}: ningún canal válido cabe en artndxn")
                continue

        # -----------------------------------------------------------------
        # 1. Agrupar epochs consecutivos con la misma etiqueta
        # -----------------------------------------------------------------
        blocks = []
        current_block = []
        current_stage = None

        for i in range(n_epochs):
            stage, start_sec, duration_sec = epochs[i]
            if stage == current_stage:
                current_block.append((i, stage, start_sec, duration_sec))
            else:
                if current_block:
                    blocks.append(current_block)
                current_block = [(i, stage, start_sec, duration_sec)]
                current_stage = stage
        if current_block:
            blocks.append(current_block)

        # -----------------------------------------------------------------
        # 2. Filtrar bloques por tamaño y calidad
        # -----------------------------------------------------------------
        kept = 0
        block_counter = 0
        for block in blocks:
            if len(block) < args.min_block_size:
                continue

            # Evaluar calidad de cada epoch del bloque
            block_ratios = []
            block_clean_masks = []
            for i, stage, start_sec, duration_sec in block:
                clean_mask = artndxn[i, valid_in_eeg]  # 1 = limpio, 0 = artefacto
                n_clean = int(np.sum(clean_mask))
                n_total = len(valid_in_eeg)
                ratio = n_clean / n_total if n_total > 0 else 0.0
                block_ratios.append(ratio)
                block_clean_masks.append(clean_mask)

            min_ratio = min(block_ratios)
            if min_ratio < args.min_clean_ratio:
                continue

            # Intersección de canales limpios (limpios en TODOS los epochs del bloque)
            combined_clean = np.all(block_clean_masks, axis=0)
            clean_ch_names = [valid_names[j] for j in range(len(valid_in_eeg)) if combined_clean[j]]
            clean_ch_str = ",".join(clean_ch_names)

            t_start = block[0][2]
            t_end = block[-1][2] + block[-1][3]
            stage = block[0][1]

            all_params.append([
                subject,
                f"{stage}_{block_counter}",
                f"{t_start:.2f}",
                f"{t_end:.2f}",
                stage,
                clean_ch_str,
            ])
            kept += 1
            block_counter += 1

        print(f"    → {kept}/{len(blocks)} bloques incluidos (≥ {args.min_block_size} epochs, ratio limpio ≥ {args.min_clean_ratio})")

    # -----------------------------------------------------------------
    # Guardar resultados
    # -----------------------------------------------------------------
    with open(out_txt, "w", encoding="utf-8") as f:
        for row in all_params:
            f.write(str(row) + "\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "block_id", "t_start_sec", "t_end_sec", "stage", "clean_channels"])
        writer.writerows(all_params)

    print(f"\n{'='*60}")
    print(f"  Total bloques generados: {len(all_params)}")
    print(f"  TXT : {out_txt.absolute()}")
    print(f"  CSV : {out_csv.absolute()}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

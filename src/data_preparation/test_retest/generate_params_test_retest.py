#!/usr/bin/env python3
"""
generate_params_test_retest.py
==============================
Escanea la base de datos test-retest (OpenNeuro, BIDS/BrainVision),
lee la duración de cada registro, y genera un archivo de parámetros de
batch donde **cada tarea completa es una ventana**.

Cada paciente tiene 3 sesiones × 5 tareas = 15 registros.
Cada línea del archivo de salida es una lista Python::

    ['sub-XX', 'sessionY_TASK', '0.00', 'duration_sec', 'TASK']

Uso
---
Desde la raíz del proyecto::

    python -m src.data_preparation.test_retest_db.generate_params_test_retest

Variables de entorno
--------------------
TEST_RETEST_DB_PATH
    Ruta raíz de la base de datos BIDS.
    Por defecto usa ``DB_TEST_RETEST_PATH`` de ``src.utils.config``.
TEST_RETEST_PARAMS_OUT
    Ruta del archivo de salida TXT. Default: ``params/batch_params_test_retest.txt``
TEST_RETEST_MAX_DURATION
    Duración máxima (en segundos) para considerar un registro válido.
    Si un registro excede este valor se emite un warning pero se incluye
    igual. Default: ``3600`` (1 hora)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Asegurar importabilidad del paquete
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from src.utils.config import BASE_PARAMS_FILE, DB_TEST_RETEST_PATH
except ImportError:
    BASE_PARAMS_FILE = "./params"
    DB_TEST_RETEST_PATH = ""

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_params_test_retest")

# ===========================================================================
# HELPERS de entorno
# ===========================================================================


def _env(var: str, default: str | None = None) -> str | None:
    return os.environ.get(var, default)


def _env_float(var: str, default: float) -> float:
    try:
        return float(os.environ[var])
    except (KeyError, ValueError):
        return default


# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

DB_PATH: Path = Path(_env("TEST_RETEST_DB_PATH", DB_TEST_RETEST_PATH or "./test_retest_dataset"))
OUT_TXT: Path = Path(_env("TEST_RETEST_PARAMS_OUT", f"{BASE_PARAMS_FILE}/batch_params_test_retest.txt"))
OUT_CSV: Path = OUT_TXT.with_suffix(".csv")
MAX_DURATION_SEC: float = _env_float("TEST_RETEST_MAX_DURATION", 3600.0)


# ===========================================================================
# MNE lazy import
# ===========================================================================


def _get_mne():
    try:
        import mne
        return mne
    except ImportError:
        logger.error("mne no está instalado. Ejecuta: pip install mne")
        sys.exit(1)


# ===========================================================================
# ESCANEO BIDS
# ===========================================================================


def scan_bids_dataset(db_path: Path) -> list[dict]:
    """Escanea BIDS y devuelve metadatos de cada .vhdr (sin cargar datos)."""
    records: list[dict] = []

    if not db_path.exists():
        logger.error("Ruta no encontrada: %s", db_path)
        return records

    for subj_dir in sorted(db_path.glob("sub-*")):
        if not subj_dir.is_dir():
            continue
        subject = subj_dir.name

        for ses_dir in sorted(subj_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            session = ses_dir.name.replace("ses-", "")

            eeg_dir = ses_dir / "eeg"
            if not eeg_dir.is_dir():
                continue

            for vhdr_file in sorted(eeg_dir.glob("*_eeg.vhdr")):
                stem = vhdr_file.stem.replace("_eeg", "")
                parts = stem.split("_")
                task_part = [p for p in parts if p.startswith("task-")]
                task = task_part[0].replace("task-", "") if task_part else "unknown"

                records.append({
                    "subject": subject,
                    "session": session,
                    "task": task,
                    "vhdr_path": vhdr_file,
                })

    return records


def get_recording_duration(vhdr_path: Path) -> float | None:
    """Carga solo el header y devuelve duración en segundos."""
    mne = _get_mne()
    try:
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=False, verbose="ERROR")
        return raw.times[-1]
    except Exception as exc:
        logger.warning("No se pudo leer %s: %s", vhdr_path.name, exc)
        return None


# ===========================================================================
# MAIN
# ===========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera batch_params para test-retest (1 ventana = 1 tarea completa)"
    )
    parser.add_argument("--db-path", type=str, default=str(DB_PATH),
                        help="Ruta raíz de la base de datos BIDS")
    parser.add_argument("--out", type=str, default=str(OUT_TXT),
                        help="Archivo de salida TXT")
    parser.add_argument("--csv", type=str, default=str(OUT_CSV),
                        help="Archivo de salida CSV adicional")
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION_SEC,
                        help="Duración máxima de warning en segundos (default: 3600)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo escanear y reportar, sin escribir archivos")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    out_txt = Path(args.out)
    out_csv = Path(args.csv)

    logger.info("=" * 60)
    logger.info("  GENERADOR DE PARÁMETROS -- Test-Retest")
    logger.info("=" * 60)
    logger.info("  DB path      : %s", db_path.absolute())
    logger.info("  Max duración : %.0f s (%.1f min)", args.max_duration, args.max_duration / 60)
    logger.info("=" * 60)

    # --- Escanear estructura ---
    records = scan_bids_dataset(db_path)
    if not records:
        logger.error("No se encontraron archivos .vhdr")
        return 1

    logger.info("Registros encontrados: %d", len(records))

    # --- Leer duraciones y generar parámetros ---
    all_params: list[list[str]] = []
    skipped = 0
    long_warnings = 0

    for rec in records:
        duration = get_recording_duration(rec["vhdr_path"])
        if duration is None:
            skipped += 1
            continue

        if duration > args.max_duration:
            logger.warning(
                "  %s/%s/%s: duración %.1f s (%.1f min) excede el umbral de %.0f s",
                rec["subject"], rec["session"], rec["task"],
                duration, duration / 60, args.max_duration,
            )
            long_warnings += 1

        logger.info(
            "  %s/%s/%s: %.1f s (%.1f min)",
            rec["subject"], rec["session"], rec["task"], duration, duration / 60,
        )

        all_params.append([
            rec["subject"],
            f"{rec['session']}_{rec['task']}",
            "0.00",
            f"{duration:.2f}",
            rec["task"],
        ])

    logger.info("")
    logger.info("Ventanas generadas: %d (1 por tarea completa)", len(all_params))
    if skipped:
        logger.info("Registros saltados (error de lectura): %d", skipped)
    if long_warnings:
        logger.info("Registros con duración > %.0f s: %d", args.max_duration, long_warnings)

    if args.dry_run:
        logger.info("[DRY-RUN] No se escribieron archivos")
        return 0

    # --- Guardar ---
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(out_txt, "w", encoding="utf-8") as f:
        for row in all_params:
            f.write(str(row) + "\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "session_task", "t_start_sec", "t_end_sec", "label"])
        writer.writerows(all_params)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  TXT : %s", out_txt.absolute())
    logger.info("  CSV : %s", out_csv.absolute())
    logger.info("  Total combinaciones: %d", len(all_params))
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

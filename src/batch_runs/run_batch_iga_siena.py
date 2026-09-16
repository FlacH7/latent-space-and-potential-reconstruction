#!/usr/bin/env python3
"""
run_batch_iga_siena.py
======================
Ejecutor de batch para el pipeline de IgA sobre la base de datos Siena Scalp EEG.

Ejecuta corridas secuenciales o paralelas del pipeline
``test_iga_from_eeg_latent_siena_db.py`` según un archivo de parámetros
(batch_params_siena.txt), con soporte de checkpointing, logging CSV y
post-procesamiento automático de asimetrías.

Diseñado exclusivamente para ejecución en consola (estaciones de cálculo).
Toda la configuración se realiza mediante variables de entorno; el script no
acepta argumentos de línea de comandos.

Variables de entorno
--------------------
SIENA_BATCH_PARAMS_FILE
    Ruta al archivo de parámetros (``.txt`` o ``.csv``). Cada línea describe
    un segmento individual::

        ['patient', 'record', 't_start_min', 't_end_min', 'period_label']

    **Requerido.**

SIENA_DB_PATH
    Ruta raíz de la base de datos Siena Scalp EEG (contiene subcarpetas por
    paciente con archivos ``.edf``).
    Por defecto usa ``DB_SIENA_PATH`` definido en ``src.utils.config``.

SIENA_OUTPUT_DIR
    Directorio base para resultados.
    Por defecto usa ``BASE_RESULTS_PATH`` definido en ``src.utils.config``.

SIENA_CACHE_DIR
    Directorio base para caché de espacios latentes.
    Por defecto usa ``BASE_CACHE_PATH`` definido en ``src.utils.config``.

SIENA_DELAY
    Pausa en segundos entre ejecuciones consecutivas (solo modo secuencial).
    Default: ``2.0``

SIENA_IGNORE_CACHE
    Si ``true``, fuerza el recálculo del espacio latente ignorando caché
    existente. Default: ``false``

SIENA_RUN_POSTPROCESS
    Si ``true``, ejecuta automáticamente el post-procesamiento de
    asimetrías al completar el batch. Default: ``true``

SIENA_MAX_WORKERS
    Número máximo de workers paralelos (``1`` = secuencial).
    **Atención:** cada worker internamente usa ``n_jobs=-1``; ajustar con
    cuidado según los cores disponibles. Default: ``1``

SIENA_LATENT_DIM
    Dimensionalidad del subespacio latente. Default: ``2``

SIENA_SCORING_METHOD
    Estrategia de selección de subespacio. Default: ``markov``

SIENA_ICA_METHOD
    Método de descomposición ICA. Default: ``picard``

SIENA_L_FREQ
    Frecuencia de corte inferior del filtro. Default: ``1.0``

SIENA_H_FREQ
    Frecuencia de corte superior del filtro. Default: ``40.0``

Uso
---
Desde la raíz del proyecto::

    export SIENA_BATCH_PARAMS_FILE=/ruta/a/batch_params_siena.txt
    python -m src.batch_runs.run_batch_iga_siena

Autor: reorganizado para el nuevo repo estructurado.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Asegurar importabilidad del paquete
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # src/batch_runs/ -> src/ -> raíz

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------
try:
    from src.utils.config import (
        BASE_CACHE_PATH, 
        BASE_RESULTS_PATH, 
        DB_SIENA_PATH, 
        BASE_PARAMS_FILE,
        LOGGING_BASE_PATH,
        LOGGING_LEVEL,
    )
    from src.post_processing.run_postprocessing import run_postprocessing
except ImportError as _exc:  # pragma: no cover
    print(
        f"[ERROR] No se pudieron importar los módulos del proyecto. "
        f"Asegúrate de ejecutar este script desde la raíz del repositorio "
        f"o de que 'src' esté en PYTHONPATH.\n{_exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOGGING_LEVEL.upper(),
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_iga_siena")

logging_path = Path(LOGGING_BASE_PATH + "/batch_iga_siena.log") if LOGGING_BASE_PATH else None
if not logging_path:
    logger.warning("No se definió LOGGING_BASE_PATH; logs no se guardarán en archivo.")

if os.environ.get("SIENA_LOG_LEVEL", LOGGING_LEVEL).upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)

# ===========================================================================
# HELPERS de lectura de entorno
# ===========================================================================


def _env(var: str, default: str | None = None) -> str | None:
    """Lee una variable de entorno."""
    return os.environ.get(var, default)


def _env_bool(var: str, default: bool = False) -> bool:
    val = os.environ.get(var, "").strip().lower()
    return val in ("1", "true", "yes", "on") if val else default


def _env_float(var: str, default: float) -> float:
    try:
        return float(os.environ[var])
    except (KeyError, ValueError):
        return default


def _env_int(var: str, default: int) -> int:
    try:
        return int(os.environ[var])
    except (KeyError, ValueError):
        return default


# ===========================================================================
# CONFIGURACIÓN GLOBAL (todas vía variables de entorno)
# ===========================================================================

PARAMS_FILE: Path = Path(_env("SIENA_BATCH_PARAMS_FILE", BASE_PARAMS_FILE + "/batch_params_siena.txt"))
SIENA_DB_PATH: Path = Path(_env("SIENA_DB_PATH", DB_SIENA_PATH))
OUTPUT_DIR: Path = Path(_env("SIENA_OUTPUT_DIR", BASE_RESULTS_PATH))
CACHE_DIR: Path = Path(_env("SIENA_CACHE_DIR", BASE_CACHE_PATH))

DELAY: float = _env_float("SIENA_DELAY", 2.0)
IGNORE_CACHE: bool = _env_bool("SIENA_IGNORE_CACHE", False)
RUN_POSTPROCESS: bool = _env_bool("SIENA_RUN_POSTPROCESS", True)
MAX_WORKERS: int = _env_int("SIENA_MAX_WORKERS", 1)

LATENT_DIM: int = _env_int("SIENA_LATENT_DIM", 2)
SCORING_METHOD: str = _env("SIENA_SCORING_METHOD", "markov")
ICA_METHOD: str = _env("SIENA_ICA_METHOD", "picard")
L_FREQ: float = _env_float("SIENA_L_FREQ", 1.0)
H_FREQ: float = _env_float("SIENA_H_FREQ", 40.0)

CHECKPOINT_FILE: Path = CACHE_DIR / _env(
    "SIENA_CHECKPOINT_FILENAME", f"batch_checkpoint_siena_{SCORING_METHOD}.json"
)

PIPELINE_MODULE: str = "src.pipelines.test_iga_from_eeg_latent_siena_db"
_PIPELINE_OUT_SUBPATH = "siena"


# ===========================================================================
# BATCH RUNNER
# ===========================================================================


class BatchRunnerSiena:
    """Orquesta la ejecución del pipeline IgA sobre Siena Scalp EEG en batch."""

    CSV_FIELDS = [
        "timestamp", "patient", "record", "t_start", "t_end", "label",
        "success", "returncode", "elapsed_s", "command",
    ]

    def __init__(self) -> None:
        if not PARAMS_FILE.exists():
            logger.error("Archivo de parámetros no encontrado: %s", PARAMS_FILE)
            sys.exit(1)
        if not _PROJECT_ROOT.exists():
            logger.error("Raíz del proyecto no encontrada: %s", _PROJECT_ROOT)
            sys.exit(1)
        if not SIENA_DB_PATH.exists():
            logger.warning("Ruta de Siena no encontrada: %s", SIENA_DB_PATH)

        self.checkpoint: set[str] = self._load_checkpoint()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logging_path / f"batch_log_siena_{ts}.csv"
        self._init_csv_log()

        logger.info("=" * 70)
        logger.info("  BATCH IgA -- Siena Scalp EEG")
        logger.info("=" * 70)
        logger.info("  Params file : %s", PARAMS_FILE)
        logger.info("  Pipeline    : %s", PIPELINE_MODULE)
        logger.info("  Siena DB    : %s", SIENA_DB_PATH)
        logger.info("  Output dir  : %s", OUTPUT_DIR)
        logger.info("  Cache dir   : %s", CACHE_DIR)
        logger.info("  Checkpoint  : %s", CHECKPOINT_FILE)
        logger.info("  Delay       : %.1f s", DELAY)
        logger.info("  Ignore cache: %s", IGNORE_CACHE)
        logger.info("  Max workers : %d (%s)", MAX_WORKERS,
                    "paralelo" if MAX_WORKERS > 1 else "secuencial")
        logger.info("  Post-process: %s", RUN_POSTPROCESS)
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> set[str]:
        if CHECKPOINT_FILE.exists():
            try:
                with open(CHECKPOINT_FILE, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ck = set(data.get("completed", []))
                logger.info("Checkpoint cargado: %d jobs previos completados", len(ck))
                return ck
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Checkpoint corrupto, empezando de cero: %s", exc)
        return set()

    def _save_checkpoint(self) -> None:
        try:
            CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as fh:
                json.dump({"completed": sorted(self.checkpoint)}, fh, indent=2)
        except OSError as exc:
            logger.warning("No se pudo guardar checkpoint: %s", exc)

    @staticmethod
    def _checkpoint_key(patient: str, record: str, t_start: str, t_end: str) -> str:
        return f"{patient}|{record}|{t_start}|{t_end}"

    # ------------------------------------------------------------------
    # Logs CSV
    # ------------------------------------------------------------------

    def _init_csv_log(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.log_file, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
        except OSError as exc:
            logger.warning("No se pudo inicializar log CSV: %s", exc)

    def _write_csv_log(
        self,
        patient: str,
        record: str,
        t_start: str,
        t_end: str,
        label: str,
        success: bool,
        returncode: int,
        elapsed: float,
        cmd: list[str],
    ) -> None:
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "patient": patient,
                    "record": record,
                    "t_start": t_start,
                    "t_end": t_end,
                    "label": label,
                    "success": success,
                    "returncode": returncode,
                    "elapsed_s": round(elapsed, 2),
                    "command": " ".join(cmd),
                })
        except OSError as exc:
            logger.warning("Fallo al escribir log CSV: %s", exc)

    # ------------------------------------------------------------------
    # Lectura de parámetros
    # ------------------------------------------------------------------

    @staticmethod
    def _read_params_txt(filepath: Path) -> list[list[str]]:
        params: list[list[str]] = []
        with open(filepath, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    row = ast.literal_eval(line)
                    if isinstance(row, list) and len(row) >= 5:
                        params.append(row)
                    else:
                        logger.warning("Línea %d ignorada (formato inválido): %s", lineno, line[:60])
                except (SyntaxError, ValueError) as exc:
                    logger.warning("Línea %d ignorada (parse error): %s -- %s", lineno, line[:60], exc)
        return params

    @staticmethod
    def _read_params_csv(filepath: Path) -> list[list[str]]:
        params: list[list[str]] = []
        with open(filepath, "r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for lineno, row in enumerate(reader, 2):
                if len(row) >= 5:
                    params.append(row[:5])
                else:
                    logger.warning("Fila %d ignorada (columnas insuficientes): %s", lineno, row)
        return params

    def _read_params(self) -> list[list[str]]:
        ext = PARAMS_FILE.suffix.lower()
        if ext == ".csv":
            logger.info("Leyendo parámetros desde CSV: %s", PARAMS_FILE)
            return self._read_params_csv(PARAMS_FILE)
        logger.info("Leyendo parámetros desde TXT: %s", PARAMS_FILE)
        return self._read_params_txt(PARAMS_FILE)

    # ------------------------------------------------------------------
    # Filtro de jobs ya completados
    # ------------------------------------------------------------------

    def _filter_todo(self, params: list[list[str]]) -> list[list[str]]:
        todo: list[list[str]] = []
        for row in params:
            patient, record, t_start, t_end, label = row[:5]
            key = self._checkpoint_key(patient, record, t_start, t_end)

            if key in self.checkpoint:
                logger.debug("SKIP (checkpoint): %s-%s [%s-%s]",
                             patient, record, t_start, t_end)
                continue

            out_sub = (
                OUTPUT_DIR
                / _PIPELINE_OUT_SUBPATH
                / f"{patient}-{record}"
                / f"{LATENT_DIM}_latent_dim"
                / f"from{t_start}_to_{t_end}_{label}"
            )
            if out_sub.exists() and any(out_sub.glob("*.png")):
                logger.debug("SKIP (output exists): %s-%s [%s-%s]", patient, record, t_start, t_end)
                self.checkpoint.add(key)
                continue

            todo.append(row)

        if skipped := len(params) - len(todo):
            logger.info("Jobs ya completados (skip): %d / %d", skipped, len(params))
        return todo

    # ------------------------------------------------------------------
    # Construcción del comando
    # ------------------------------------------------------------------

    def _build_command(
        self, patient: str, record: str, t_start: str, t_end: str, label: str
    ) -> list[str]:
        cmd = [
            sys.executable,
            "-m", PIPELINE_MODULE,
            "--patient", patient,
            "--record", record,
            "--t-start", str(t_start),
            "--t-end", str(t_end),
            "--period-label", label,
            "--latent-dim", str(LATENT_DIM),
            "--scoring-method", SCORING_METHOD,
            "--l-freq", str(L_FREQ),
            "--h-freq", str(H_FREQ),
            "--ica-method", ICA_METHOD,
        ]
        if IGNORE_CACHE:
            cmd.append("--ignore-cache")
        return cmd

    # ------------------------------------------------------------------
    # Ejecución de un job individual
    # ------------------------------------------------------------------

    def _run_single_job(self, row: list[str]) -> tuple[str, bool]:
        patient, record, t_start, t_end, label = row[:5]
        key = self._checkpoint_key(patient, record, t_start, t_end)

        try:
            cmd = self._build_command(patient, record, t_start, t_end, label)
        except FileNotFoundError as exc:
            logger.error("No se encontró .edf para %s-%s: %s", patient, record, exc)
            return key, False

        logger.info("RUN | %s-%s | [%s-%s] min | %s", patient, record, t_start, t_end, label)
        logger.debug("CMD: %s", " ".join(cmd))

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                cwd=_PROJECT_ROOT,
            )

            if result.stdout:
                for line in result.stdout.splitlines():
                    logger.info("  [PIPE] %s", line)

            elapsed = time.time() - t0
            success = result.returncode == 0

            self._write_csv_log(
                patient, record, t_start, t_end, label,
                success, result.returncode, elapsed, cmd,
            )

            if success:
                logger.info("OK | %s-%s (%.1f s)", patient, record, elapsed)
            else:
                logger.error("ERROR | %s-%s -- código %d",
                             patient, record, result.returncode)

            return key, success

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error("EXCEPTION | %s-%s: %s", patient, record, exc)
            self._write_csv_log(
                patient, record, t_start, t_end, label,
                False, -1, elapsed, cmd,
            )
            return key, False

    # ------------------------------------------------------------------
    # Post-procesamiento
    # ------------------------------------------------------------------

    def _run_postprocessing(self) -> None:
        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO POST-PROCESAMIENTO DE ASIMETRÍAS")
        logger.info("=" * 70)
        try:
            postprocess_dir = OUTPUT_DIR / "postprocess_asymmetry"
            run_postprocessing(
                db_name="siena",
                root_dir=OUTPUT_DIR,
                output_dir=postprocess_dir,
            )
            logger.info("Post-procesamiento completado. Resultados en: %s", postprocess_dir)
        except Exception:
            logger.error("ERROR en post-procesamiento:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    # Orquestación principal
    # ------------------------------------------------------------------

    def run(self) -> int:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        params = self._read_params()
        if not params:
            logger.error("No se encontraron parámetros válidos en %s", PARAMS_FILE)
            return 1

        todo = self._filter_todo(params)
        total = len(todo)

        if total == 0:
            logger.info("Todos los jobs ya están completados.")
            if RUN_POSTPROCESS:
                self._run_postprocessing()
            return 0

        logger.info("Total a ejecutar: %d / %d", total, len(params))

        completed = 0
        failed = 0

        if MAX_WORKERS > 1:
            completed, failed = self._run_parallel(todo, total)
        else:
            completed, failed = self._run_sequential(todo, total)

        self._save_checkpoint()

        logger.info("=" * 70)
        logger.info("BATCH COMPLETADO -- OK: %d | Fallos: %d | Total: %d", completed, failed, total)
        logger.info("Log CSV: %s", self.log_file)

        if RUN_POSTPROCESS and completed > 0:
            self._run_postprocessing()

        return 0 if failed == 0 else 1

    def _run_sequential(self, todo: list[list[str]], total: int) -> tuple[int, int]:
        completed = 0
        failed = 0

        for idx, row in enumerate(todo, start=1):
            logger.info("")
            logger.info("-" * 70)
            logger.info("Progreso: %d / %d", idx, total)
            logger.info("-" * 70)

            key, success = self._run_single_job(row)
            if success:
                self.checkpoint.add(key)
                completed += 1
            else:
                failed += 1

            self._save_checkpoint()

            if idx < total and DELAY > 0:
                logger.debug("Pausa %.1f s...", DELAY)
                time.sleep(DELAY)

        return completed, failed

    def _run_parallel(self, todo: list[list[str]], total: int) -> tuple[int, int]:
        completed = 0
        failed = 0

        logger.info("Modo PARALELO con %d workers", MAX_WORKERS)

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_row = {
                executor.submit(self._run_single_job, row): row for row in todo
            }

            for future in as_completed(future_to_row):
                row = future_to_row[future]
                patient, record, *_ = row
                try:
                    key, success = future.result()
                    if success:
                        self.checkpoint.add(key)
                        completed += 1
                    else:
                        failed += 1
                except Exception as exc:
                    logger.error("FUTURE EXCEPTION | %s-%s: %s", patient, record, exc)
                    failed += 1

                self._save_checkpoint()
                logger.info("Progreso: %d / %d completados", completed + failed, total)

        return completed, failed


# ===========================================================================
# ENTRY POINT
# ===========================================================================


def main() -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = BatchRunnerSiena()
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())

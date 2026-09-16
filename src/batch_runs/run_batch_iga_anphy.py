#!/usr/bin/env python3
"""
run_batch_iga_anphy.py
======================
Ejecutor de batch para el pipeline de IgA sobre la base de datos ANPHY-Sleep.

Ejecuta corridas secuenciales o paralelas del pipeline
``test_iga_from_eeg_latent_anphy.py`` según un archivo de parámetros
(batch_params_anphy.txt), con soporte de checkpointing, logging CSV y
post-procesamiento automático de asimetrías.

Diseñado exclusivamente para ejecución en consola (estaciones de cálculo).
Toda la configuración se realiza mediante variables de entorno; el script no
acepta argumentos de línea de comandos.

Variables de entorno
--------------------
ANPHY_BATCH_PARAMS_FILE
    Ruta al archivo de parámetros (``.txt`` o ``.csv``). Cada línea describe
    un epoch individual::

        ['subject', 'epoch_idx', 't_start_sec', 't_end_sec', 'stage_label']

    **Requerido.**

ANPHY_DB_PATH
    Ruta raíz de la base de datos ANPHY-Sleep (contiene subcarpetas por
    sujeto con archivos ``.edf``).
    Por defecto usa ``DB_ANPHY_PATH`` definido en ``src.utils.config``.

ANPHY_OUTPUT_DIR
    Directorio base para resultados.
    Por defecto usa ``BASE_RESULTS_PATH`` definido en ``src.utils.config``.

ANPHY_CACHE_DIR
    Directorio base para caché de espacios latentes.
    Por defecto usa ``BASE_CACHE_PATH`` definido en ``src.utils.config``.

ANPHY_DELAY
    Pausa en segundos entre ejecuciones consecutivas (solo modo secuencial).
    Default: ``2.0``

ANPHY_IGNORE_CACHE
    Si ``true``, fuerza el recálculo del espacio latente ignorando caché
    existente. Default: ``false``

ANPHY_RUN_POSTPROCESS
    Si ``true``, ejecuta automáticamente el post-procesamiento de
    asimetrías al completar el batch. Default: ``true``

ANPHY_MAX_WORKERS
    Número máximo de workers paralelos (``1`` = secuencial).
    **Atención:** cada worker internamente usa ``n_jobs=-1``; ajustar con
    cuidado según los cores disponibles. Default: ``1``

ANPHY_LATENT_DIM
    Dimensionalidad del subespacio latente. Default: ``2``

ANPHY_SCORING_METHOD
    Estrategia de selección de subespacio. Default: ``markov``

ANPHY_ICA_METHOD
    Método de descomposición ICA. Default: ``picard``

ANPHY_L_FREQ
    Frecuencia de corte inferior del filtro. Default: ``1.0``

ANPHY_H_FREQ
    Frecuencia de corte superior del filtro. Default: ``40.0``

Uso
---
Desde la raíz del proyecto::

    export ANPHY_BATCH_PARAMS_FILE=/ruta/a/batch_params_anphy.txt
    python -m src.batch_runs.run_batch_iga_anphy

O directamente::

    PYTHONPATH=/ruta/al/proyecto/src:$PYTHONPATH python src/batch_runs/run_batch_iga_anphy.py

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
        DB_ANPHY_PATH, 
        BASE_PARAMS_FILE,
        LOGGING_BASE_PATH,
        LOGGING_LEVEL,
    )
    from src.post_processing.run_postprocessing import run_postprocessing
except ImportError as _exc:  # pragma: no cover
    # Fallback amigable si se ejecuta sin el entorno del proyecto configurado
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
logger = logging.getLogger("batch_iga_anphy")

logging_path = Path(LOGGING_BASE_PATH + "/batch_iga_anphy.log") if LOGGING_BASE_PATH else None
if not logging_path:
    logger.warning("No se definió LOGGING_BASE_PATH; logs no se guardarán en archivo.")

# Permite ajustar nivel vía variable de entorno
if os.environ.get("ANPHY_LOG_LEVEL", LOGGING_LEVEL).upper() == "DEBUG":
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

PARAMS_FILE: Path = Path(_env("ANPHY_BATCH_PARAMS_FILE", BASE_PARAMS_FILE + "/batch_params_anphy.txt"))
ANPHY_DB_PATH: Path = Path(_env("ANPHY_DB_PATH", DB_ANPHY_PATH + "/osfstorage"))
OUTPUT_DIR: Path = Path(_env("ANPHY_OUTPUT_DIR", BASE_RESULTS_PATH))
CACHE_DIR: Path = Path(_env("ANPHY_CACHE_DIR", BASE_CACHE_PATH))

DELAY: float = _env_float("ANPHY_DELAY", 2.0)
IGNORE_CACHE: bool = _env_bool("ANPHY_IGNORE_CACHE", False)
RUN_POSTPROCESS: bool = _env_bool("ANPHY_RUN_POSTPROCESS", True)
MAX_WORKERS: int = _env_int("ANPHY_MAX_WORKERS", 1)

LATENT_DIM: int = _env_int("ANPHY_LATENT_DIM", 2)
SCORING_METHOD: str = _env("ANPHY_SCORING_METHOD", "markov")
ICA_METHOD: str = _env("ANPHY_ICA_METHOD", "picard")
L_FREQ: float = _env_float("ANPHY_L_FREQ", 1.0)
H_FREQ: float = _env_float("ANPHY_H_FREQ", 40.0)

CHECKPOINT_FILE: Path = CACHE_DIR / _env(
    "ANPHY_CHECKPOINT_FILENAME", f"batch_checkpoint_anphy_{SCORING_METHOD}.json"
)

# Ruta al script del pipeline (relativa a este archivo)
PIPELINE_MODULE: Path = "src.pipelines.test_iga_from_eeg_latent_anphy"

# Sub-ruta dentro de OUTPUT_DIR donde el pipeline coloca resultados
_PIPELINE_OUT_SUBPATH = "anphy"


# ===========================================================================
# BATCH RUNNER
# ===========================================================================


class BatchRunnerAnphy:
    """Orquesta la ejecución del pipeline IgA sobre ANPHY-Sleep en batch."""

    # Campos del log CSV
    CSV_FIELDS = [
        "timestamp", "subject", "epoch_idx", "t_start", "t_end", "label",
        "success", "returncode", "elapsed_s", "command",
    ]

    def __init__(self) -> None:
        # Validar que existen los componentes esenciales
        if not PARAMS_FILE.exists():
            logger.error("Archivo de parámetros no encontrado: %s", PARAMS_FILE)
            sys.exit(1)
        if not _PROJECT_ROOT.exists():
            logger.error("Raíz del proyecto no encontrada: %s", _PROJECT_ROOT)
            sys.exit(1)
        if not ANPHY_DB_PATH.exists():
            logger.warning("Ruta de ANPHY no encontrada: %s", ANPHY_DB_PATH)

        self.checkpoint: set[str] = self._load_checkpoint()

        # Archivo de log CSV con timestamp para no sobrescribir corridas previas
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logging_path / f"batch_log_anphy_{ts}.csv"
        self._init_csv_log()

        logger.info("=" * 70)
        logger.info("  BATCH IgA -- ANPHY-Sleep")
        logger.info("=" * 70)
        logger.info("  Params file : %s", PARAMS_FILE)
        logger.info("  Pipeline    : %s", PIPELINE_MODULE)
        logger.info("  ANPHY DB    : %s", ANPHY_DB_PATH)
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
    def _checkpoint_key(subject: str, epoch_idx: str, t_start: str, t_end: str) -> str:
        return f"{subject}|{epoch_idx}|{t_start}|{t_end}"

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
        subject: str,
        epoch_idx: str,
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
                    "subject": subject,
                    "epoch_idx": epoch_idx,
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
            next(reader, None)  # saltar header
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
        """Filtra epochs ya procesados (checkpoint o existencia de PNGs)."""
        todo: list[list[str]] = []
        for row in params:
            subject, epoch_idx, t_start, t_end, label = row[:5]
            key = self._checkpoint_key(subject, epoch_idx, t_start, t_end)

            if key in self.checkpoint:
                logger.debug("SKIP (checkpoint): %s epoch %s [%s-%s]",
                             subject, epoch_idx, t_start, t_end)
                continue

            # Verificación adicional: ¿existe el directorio de salida con PNGs?
            out_sub = (
                OUTPUT_DIR
                / _PIPELINE_OUT_SUBPATH
                / subject
                / f"{LATENT_DIM}_latent_dim"
                / f"epoch_{t_start}s_{t_end}s_{label}"
            )
            if out_sub.exists() and any(out_sub.glob("*.png")):
                logger.debug("SKIP (output exists): %s epoch %s", subject, epoch_idx)
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
        self, subject: str, epoch_idx: str, t_start: str, t_end: str, label: str
    ) -> list[str]:
        # El pipeline infiere --file, --cache-file y --out-dir de config;
        # aquí solo pasamos los parámetros esenciales y flags.
        cmd = [
            sys.executable,
            "-m", PIPELINE_MODULE,
            "--subject", subject,
            "--t-start", str(t_start),
            "--t-end", str(t_end),
            "--epoch-idx", str(epoch_idx),
            "--stage-label", label,
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
        """Ejecuta un epoch. Devuelve (checkpoint_key, éxito)."""
        subject, epoch_idx, t_start, t_end, label = row[:5]
        key = self._checkpoint_key(subject, epoch_idx, t_start, t_end)

        try:
            cmd = self._build_command(subject, epoch_idx, t_start, t_end, label)
        except FileNotFoundError as exc:
            logger.error("No se encontró .edf para %s: %s", subject, exc)
            return key, False

        logger.info("RUN | %s | epoch %s | [%s-%s] s | %s", subject, epoch_idx, t_start, t_end, label)
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

            # Volcar stdout del pipeline al logger
            if result.stdout:
                for line in result.stdout.splitlines():
                    logger.info("  [PIPE] %s", line)

            elapsed = time.time() - t0
            success = result.returncode == 0

            self._write_csv_log(
                subject, epoch_idx, t_start, t_end, label,
                success, result.returncode, elapsed, cmd,
            )

            if success:
                logger.info("OK | %s epoch %s (%.1f s)", subject, epoch_idx, elapsed)
            else:
                logger.error("ERROR | %s epoch %s -- código %d",
                             subject, epoch_idx, result.returncode)

            return key, success

        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            logger.error("EXCEPTION | %s epoch %s: %s", subject, epoch_idx, exc)
            self._write_csv_log(
                subject, epoch_idx, t_start, t_end, label,
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
                db_name = "anphy",
                root_dir=OUTPUT_DIR,
                output_dir=postprocess_dir,
            )
            logger.info("Post-procesamiento completado. Resultados en: %s", postprocess_dir)
        except Exception:  # noqa: BLE001
            logger.error("ERROR en post-procesamiento:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    # Orquestación principal
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Punto de entrada principal. Devuelve código de salida."""
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

        # Guardar checkpoint final
        self._save_checkpoint()

        logger.info("=" * 70)
        logger.info("BATCH COMPLETADO -- OK: %d | Fallos: %d | Total: %d", completed, failed, total)
        logger.info("Log CSV: %s", self.log_file)

        if RUN_POSTPROCESS and completed > 0:
            self._run_postprocessing()

        return 0 if failed == 0 else 1

    def _run_sequential(self, todo: list[list[str]], total: int) -> tuple[int, int]:
        """Ejecución secuencial con delay configurable."""
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
        """Ejecución paralela con ProcessPoolExecutor."""
        completed = 0
        failed = 0

        logger.info("Modo PARALELO con %d workers", MAX_WORKERS)

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_row = {
                executor.submit(self._run_single_job, row): row for row in todo
            }

            for future in as_completed(future_to_row):
                row = future_to_row[future]
                subject, epoch_idx, *_ = row
                try:
                    key, success = future.result()
                    if success:
                        self.checkpoint.add(key)
                        completed += 1
                    else:
                        failed += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error("FUTURE EXCEPTION | %s epoch %s: %s", subject, epoch_idx, exc)
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
        runner = BatchRunnerAnphy()
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())

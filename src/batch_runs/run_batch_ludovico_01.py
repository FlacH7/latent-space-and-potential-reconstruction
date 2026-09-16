#!/usr/bin/env python3
"""
run_batch_ludovico_01.py
==========================
Ejecutor de batch multi-metodo para el pipeline IgA sobre el dataset **Ludovico_01**.

A diferencia de ``run_batch_multi_method.py`` (que itera sobre sujetos/sesiones/tareas
con estructura BIDS), este batch itera sobre una **lista plana de nombres de
ficheros CSV** leida del JSON de parametros.

Diferencias clave con el batch de EEG:

1. **Sujeto = nombre del CSV** (sin extension), leido de ``"subjects"`` como lista.
2. **Sesion y tarea son fijas**: ``"session1"`` y ``"default"``.
3. **No se pasan ``--l-freq`` ni ``--h-freq``** (datos no-EEG, sin filtrado).
4. **No se pasa ``--ica-method``** (no hay ICA).
5. **El directorio de salida usa ``ludovico_01/``** en vez de ``test_retest_gedai/``.
6. **Se pasa ``--sfreq``** con el valor del JSON (default 1000 Hz).

Uso::

    # Usando defaults (lee el JSON)
    python -m src.batch_runs.run_batch_ludovico_01

    # Apuntando a otro JSON
    BATCH_LUDOVICO_PARAMS_JSON=/path/to/other.json \
        python -m src.batch_runs.run_batch_ludovico_01
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Asegurar importabilidad del paquete
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # batch_runs/ -> src/ -> raiz

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
        BASE_PARAMS_FILE,
        LOGGING_BASE_PATH,
        LOGGING_LEVEL,
        DEFAULT_BATCH_RUNS_WORKERS,
    )
except ImportError as _exc:
    print(
        f"[ERROR] No se pudieron importar los modulos del proyecto. "
        f"Asegurate de ejecutar este script desde la raiz del repositorio "
        f"o de que 'src' este en PYTHONPATH.\n{_exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# Try to import Ludovico_01 path
try:
    from src.utils.config import DB_LUDOVICO_01_PATH
except (ImportError, AttributeError):
    DB_LUDOVICO_01_PATH = None

# ---------------------------------------------------------------------------
# Configuracion de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOGGING_LEVEL.upper(),
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_ludovico_01")

logging_path = (
    Path(LOGGING_BASE_PATH + "/batch_ludovico_01") if LOGGING_BASE_PATH else None
)
if not logging_path:
    logger.warning("No se definio LOGGING_BASE_PATH; logs no se guardaran en archivo.")
else:
    os.makedirs(logging_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Ruta por defecto del JSON de parametros
# ---------------------------------------------------------------------------
DEFAULT_PARAMS_JSON = Path("./src/batch_runs/ludovico_01_call_params.json")

# Fijos para Ludovico_01
_FIXED_SESSION = "session1"
_FIXED_TASK = "default"


# ---------------------------------------------------------------------------
# Lectura del JSON de parametros
# ---------------------------------------------------------------------------


def _load_params(json_path: Path) -> dict:
    """Carga y valida el JSON de parametros del batch de Ludovico_01."""
    if not json_path.exists():
        logger.error("Archivo JSON de parametros no encontrado: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    # Validaciones minimas
    required_top_keys = ["subjects", "methods", "shared_params"]
    for key in required_top_keys:
        if key not in params:
            logger.error("Falta la clave requerida '%s' en el JSON", key)
            sys.exit(1)

    # subjects debe ser una lista de strings
    if not isinstance(params["subjects"], list):
        logger.error("'subjects' debe ser una lista de nombres de ficheros CSV")
        sys.exit(1)

    for method in params["methods"]:
        for req_key in ("label", "stage2_dynamics", "stage3_selection"):
            if req_key not in method:
                logger.error(
                    "El metodo '%s' falta la clave requerida '%s'",
                    method.get("label", "<sin label>"), req_key,
                )
                sys.exit(1)

    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spec_label(method: dict) -> str:
    s1 = method.get("stage1_embedding") or "none"
    return f"{s1}+{method['stage2_dynamics']}+{method['stage3_selection']}"


def _spec_hash(method: dict, shared: dict) -> str:
    spec = {
        "stage1_embedding": method.get("stage1_embedding"),
        "stage1_params": method.get("stage1_params", {}),
        "stage2_dynamics": method["stage2_dynamics"],
        "stage2_params": method.get("stage2_params", {}),
        "stage3_selection": method["stage3_selection"],
        "stage3_params": method.get("stage3_params", {}),
    }
    payload = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


# ===========================================================================
# GENERACION DE JOBS
# ===========================================================================


def _generate_jobs(params: dict) -> list[dict]:
    """Genera la lista de jobs.

    Cada job = (subject_csv, session_fija, task_fija, method).
    """
    subjects = params["subjects"]  # lista de strings
    methods = params["methods"]
    tw = params.get("time_window", {"t_start": 0.0, "t_end": 10.0})
    shared = params["shared_params"]

    jobs: list[dict] = []
    for subject in subjects:
        for method in methods:
            jobs.append({
                "subject": subject,
                "session": _FIXED_SESSION,
                "task": _FIXED_TASK,
                "method_label": method["label"],
                "method": method,
                "shared": shared,
                "t_start": str(tw["t_start"]),
                "t_end": str(tw["t_end"]),
            })
    return jobs


# ===========================================================================
# BATCH RUNNER
# ===========================================================================


class Ludovico01BatchRunner:
    """Orquesta la ejecucion multi-metodo del pipeline IgA para Ludovico_01."""

    CSV_FIELDS = [
        "timestamp", "subject", "method_label",
        "spec_label", "t_start", "t_end", "success", "returncode",
        "elapsed_s", "command",
    ]

    def __init__(self, params: dict) -> None:
        self.params = params
        self.pipeline_module = params.get(
            "pipeline_module",
            "src.pipelines.test_iga_from_eeg_latent_ludovico_01",
        )
        self.shared = params["shared_params"]
        self.exec_cfg = params.get("execution", {})
        self.post_cfg = params.get("postprocess", {})

        # Rutas
        self.db_path = DB_LUDOVICO_01_PATH
        self.output_dir = Path(BASE_RESULTS_PATH)
        self.cache_dir = Path(BASE_CACHE_PATH)

        # mode_map path for cdhsa_specific_modes
        self.mode_map_path: str | None = params.get("mode_map_path")
        if self.mode_map_path is not None:
            logger.info("  mode_map_path     : %s", self.mode_map_path)

        # Detectar si algun metodo usa cdhsa_specific_modes
        self._has_cdhsa_method = any(
            m["stage2_dynamics"] == "cdhsa_specific_modes"
            for m in params["methods"]
        )
        if self._has_cdhsa_method and self.mode_map_path is None:
            # Intentar resolver desde BASE_PARAMS_FILE / params
            try:
                from src.utils.config import BASE_PARAMS_FILE
                params_dir = Path(BASE_PARAMS_FILE)
                candidates = list(params_dir.glob("mode_map_*_ludovico_01_*.json"))
                if candidates:
                    self.mode_map_path = str(max(candidates, key=lambda p: p.stat().st_mtime))
                    logger.info("  mode_map_path     : %s (auto-resolved)", self.mode_map_path)
            except (ImportError, AttributeError):
                pass

        if self._has_cdhsa_method and self.mode_map_path is None:
            logger.warning(
                "  [WARN] Methods use cdhsa_specific_modes but no mode_map_path "
                "resolved. Pass 'mode_map_path' in JSON or --mode-map-path in CLI."
            )

        # Ejecucion
        self.delay: float = self.exec_cfg.get("delay", 2.0)
        self.max_workers: int = self.exec_cfg.get("max_workers", DEFAULT_BATCH_RUNS_WORKERS)
        self.run_postprocess: bool = self.exec_cfg.get("run_postprocess", True)
        self.ignore_cache: bool = self.shared.get("ignore_cache", False)

        # sfreq
        self.sfreq: float = self.shared.get("sfreq", 1000.0)

        # Logging
        log_level = self.exec_cfg.get("log_level", LOGGING_LEVEL)
        if log_level.upper() == "DEBUG":
            logger.setLevel(logging.DEBUG)

        # Imagenes para post-procesamiento
        self.potential_img = self.post_cfg.get("potential_img", "potential_2d.png")
        self.noncons_img = self.post_cfg.get(
            "nonconservative_img", "potential_2d_nonconservative_force.png"
        )
        self.composite_prefix = self.post_cfg.get("composite_prefix", "methods_comparison")

        # Checkpoint
        self.checkpoint: set[str] = self._load_checkpoint()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logging_path / f"batch_log_ludovico_01_{ts}.csv"
        self._init_csv_log()

        # Generar jobs
        self.all_jobs = _generate_jobs(params)
        self._print_banner()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        n_subjects = len({j["subject"] for j in self.all_jobs})
        n_methods = len({j["method_label"] for j in self.all_jobs})

        logger.info("=" * 70)
        logger.info("  BATCH MULTI-METHOD -- Ludovico_01 (CSV, non-EEG)")
        logger.info("=" * 70)
        logger.info("  Pipeline    : %s", self.pipeline_module)
        logger.info("  DB path     : %s", self.db_path)
        logger.info("  Output dir  : %s", self.output_dir)
        logger.info("  Cache dir   : %s", self.cache_dir)
        logger.info("  Checkpoint  : %s", self._checkpoint_path())
        logger.info("  ---")
        logger.info("  Subjects    : %d", n_subjects)
        logger.info("  Methods     : %s",
                    [m["label"] for m in self.params["methods"]])
        tw = self.params.get("time_window", {"t_start": 0.0, "t_end": 10.0})
        logger.info("  Time window : %.1f s -> %.1f s", tw["t_start"], tw["t_end"])
        logger.info("  sfreq       : %.0f Hz", self.sfreq)
        logger.info("  Latent dim  : %d", self.shared.get("latent_dim", 2))
        logger.info("  ---")
        logger.info("  Total jobs  : %d (%d subj x %d meth)",
                    len(self.all_jobs), n_subjects, n_methods)
        logger.info("  Delay       : %.1f s", self.delay)
        logger.info("  Max workers : %d (%s)",
                    self.max_workers, "paralelo" if self.max_workers > 1 else "secuencial")
        logger.info("  Post-process: %s", self.run_postprocess)
        if self.mode_map_path:
            logger.info("  mode_map    : %s", self.mode_map_path)
        if self._has_cdhsa_method:
            logger.info("  CDHSA modes : YES (cdhsa_specific_modes)")
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        return self.cache_dir / "batch_checkpoint_ludovico_01.json"

    def _load_checkpoint(self) -> set[str]:
        cp = self._checkpoint_path() if hasattr(self, "cache_dir") else \
            Path(BASE_CACHE_PATH) / "batch_checkpoint_ludovico_01.json"
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ck = set(data.get("completed", []))
                logger.info("Checkpoint cargado: %d jobs previos completados", len(ck))
                return ck
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Checkpoint corrupto, empezando de cero: %s", exc)
        return set()

    def _save_checkpoint(self) -> None:
        cp = self._checkpoint_path()
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump({"completed": sorted(self.checkpoint)}, fh, indent=2)
        except OSError as exc:
            logger.warning("No se pudo guardar checkpoint: %s", exc)

    @staticmethod
    def _checkpoint_key(subject: str, method_label: str,
                        t_start: str, t_end: str) -> str:
        return f"{subject}|{method_label}|{t_start}|{t_end}"

    # ------------------------------------------------------------------
    # Logs CSV
    # ------------------------------------------------------------------

    def _init_csv_log(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.log_file, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
        except OSError as exc:
            logger.warning("No se pudo inicializar log CSV: %s", exc)

    def _write_csv_log(self, job: dict, success: bool,
                       returncode: int, elapsed: float, cmd: list[str]) -> None:
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.CSV_FIELDS)
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "subject": job["subject"],
                    "method_label": job["method_label"],
                    "spec_label": _spec_label(job["method"]),
                    "t_start": job["t_start"],
                    "t_end": job["t_end"],
                    "success": success,
                    "returncode": returncode,
                    "elapsed_s": round(elapsed, 2),
                    "command": " ".join(cmd),
                })
        except OSError as exc:
            logger.warning("Fallo al escribir log CSV: %s", exc)

    # ------------------------------------------------------------------
    # Filtro de jobs ya completados
    # ------------------------------------------------------------------

    def _filter_todo(self, jobs: list[dict]) -> list[dict]:
        todo: list[dict] = []
        for job in jobs:
            key = self._checkpoint_key(
                job["subject"], job["method_label"],
                job["t_start"], job["t_end"],
            )
            if key in self.checkpoint:
                logger.debug(
                    "SKIP (checkpoint): %s [%s]",
                    job["subject"], job["method_label"],
                )
                continue
            todo.append(job)

        if skipped := len(jobs) - len(todo):
            logger.info("Jobs ya completados (skip): %d / %d", skipped, len(jobs))
        return todo

    # ------------------------------------------------------------------
    # Ruta de salida para un job
    # ------------------------------------------------------------------

    def _get_output_dir(self, job: dict) -> Path:
        """Directorio donde el pipeline guarda resultados."""
        method = job["method"]
        shared = job["shared"]
        latent_dim = shared.get("latent_dim", 2)
        spec_label = _spec_label(method)

        session_dir = (
            self.output_dir
            / "ludovico_01"
            / job["subject"]
            / _FIXED_SESSION
        )

        prefix = f"{latent_dim}_latent_dim_{spec_label}_"
        if session_dir.is_dir():
            candidates = [
                d for d in session_dir.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
            ]
            if len(candidates) == 1:
                method_dir = candidates[0]
            elif len(candidates) > 1:
                method_dir = max(candidates, key=lambda p: p.stat().st_mtime)
                logger.warning(
                    "Multiples carpetas para %s, usando la mas reciente: %s",
                    spec_label, method_dir.name,
                )
            else:
                method_dir = session_dir / f"{prefix}????????"
        else:
            method_dir = session_dir / f"{prefix}????????"

        return method_dir / f"from{job['t_start']}s_to_{job['t_end']}s_{_FIXED_TASK}"

    # ------------------------------------------------------------------
    # Construccion del comando
    # ------------------------------------------------------------------

    def _build_command(self, job: dict) -> list[str]:
        """Construye el comando para llamar al pipeline de Ludovico_01.

        NO pasa --l-freq, --h-freq, --ica-method.
        """
        method = job["method"]
        shared = job["shared"]

        cmd = [
            sys.executable,
            "-m", self.pipeline_module,
            "--subject", job["subject"],
            "--t-start", job["t_start"],
            "--t-end", job["t_end"],
            "--latent-dim", str(shared.get("latent_dim", 2)),
            "--sfreq", str(self.sfreq),
        ]

        # --- Stage 1: Embedding ---
        s1 = method.get("stage1_embedding")
        if s1:
            cmd.extend(["--stage1-embedding", s1])
            s1_params = method.get("stage1_params", {})
            if s1_params:
                cmd.extend(["--stage1-params", json.dumps(s1_params)])
        else:
            cmd.extend(["--stage1-embedding", "none"])

        # --- Stage 2: Dynamics ---
        cmd.extend(["--stage2-dynamics", method["stage2_dynamics"]])
        s2_params = method.get("stage2_params", {})

        # For cdhsa_specific_modes, inject mode_map_path and condition
        if method["stage2_dynamics"] == "cdhsa_specific_modes":
            if self.mode_map_path is not None:
                cmd.extend(["--mode-map-path", self.mode_map_path])
            # condition = subject name (auto-injected by pipeline, but explicit is safer)
            condition_name = s2_params.get("condition", job["subject"])
            cmd.extend(["--condition", condition_name])

        if s2_params:
            cmd.extend(["--stage2-params", json.dumps(s2_params)])

        # --- Stage 3: Selection ---
        cmd.extend(["--stage3-selection", method["stage3_selection"]])
        s3_params = method.get("stage3_params", {})
        if s3_params:
            cmd.extend(["--stage3-params", json.dumps(s3_params)])

        # --- Shared params opcionales ---
        if shared.get("analysis_dim") is not None:
            cmd.extend(["--analysis-dim", str(shared["analysis_dim"])])

        if shared.get("column") is not None:
            cmd.extend(["--column", str(shared["column"])])

        if shared.get("workers") is not None:
            cmd.extend(["--workers", str(shared["workers"])])

        if shared.get("hankel_embedding_depth") is not None:
            cmd.extend([
                "--hankel-embedding-depth",
                str(shared["hankel_embedding_depth"]),
            ])

        # Diffusion maps params
        if method["stage2_dynamics"] != "diffusion_maps":
            for dflag, dkey in [
                ("--diffusion-sigma", "diffusion_sigma"),
                ("--diffusion-k", "diffusion_k"),
                ("--diffusion-time", "diffusion_time"),
                ("--diffusion-alpha", "diffusion_alpha"),
            ]:
                val = shared.get(dkey)
                if val is not None:
                    cmd.extend([dflag, str(val)])

        # KM bins
        if shared.get("km_bins") is not None:
            cmd.extend(["--km-bins", str(shared["km_bins"])])

        # Flags booleanos
        if shared.get("ignore_cache", False) or self.ignore_cache:
            cmd.append("--ignore-cache")

        if not shared.get("verbose", True):
            cmd.append("--no-verbose")

        if shared.get("no_save_potential", False):
            cmd.append("--no-save-potential")

        return cmd

    # ------------------------------------------------------------------
    # Ejecucion de un job individual
    # ------------------------------------------------------------------

    def _run_single_job(self, job: dict) -> tuple[str, bool]:
        key = self._checkpoint_key(
            job["subject"], job["method_label"],
            job["t_start"], job["t_end"],
        )

        try:
            cmd = self._build_command(job)
        except Exception as exc:
            logger.error(
                "Error construyendo comando para %s [%s]: %s",
                job["subject"], job["method_label"], exc,
            )
            return key, False

        logger.info(
            "RUN | %s | method=%s (%s) | [%s-%s] s",
            job["subject"],
            job["method_label"], _spec_label(job["method"]),
            job["t_start"], job["t_end"],
        )
        logger.debug("CMD: %s", " ".join(cmd))

        t0 = time.time()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=_PROJECT_ROOT,
            )

            for line in proc.stdout:
                logger.info("  [PIPE] %s", line.rstrip())

            proc.wait()
            elapsed = time.time() - t0
            success = proc.returncode == 0

            self._write_csv_log(job, success, proc.returncode, elapsed, cmd)

            if success:
                logger.info(
                    "OK | %s [%s] (%.1f s)",
                    job["subject"], job["method_label"], elapsed,
                )
            else:
                logger.error(
                    "ERROR | %s [%s] -- codigo %d",
                    job["subject"], job["method_label"], proc.returncode,
                )

            return key, success

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "EXCEPTION | %s [%s]: %s",
                job["subject"], job["method_label"], exc,
            )
            self._write_csv_log(job, False, -1, elapsed, cmd if 'cmd' in dir() else [])
            return key, False

    # ------------------------------------------------------------------
    # Post-procesamiento: graficos compuestos
    # ------------------------------------------------------------------

    @staticmethod
    def _embedding_key(method: dict) -> str:
        s1 = method.get("stage1_embedding")
        return s1 if s1 and s1 != "none" else "none"

    @staticmethod
    def _embedding_display_name(key: str) -> str:
        return key if key != "none" else "No embedding"

    def _organize_method_matrix(self, jobs: list[dict]) -> dict:
        col_set: list[tuple[str, str]] = []
        seen_cols: set[tuple[str, str]] = set()
        for job in jobs:
            m = job["method"]
            pair = (m["stage2_dynamics"], m["stage3_selection"])
            if pair not in seen_cols:
                seen_cols.add(pair)
                col_set.append(pair)

        emb_set: list[str] = []
        seen_emb: set[str] = set()
        for job in jobs:
            ek = self._embedding_key(job["method"])
            if ek not in seen_emb:
                seen_emb.add(ek)
                emb_set.append(ek)

        emb_ordered = sorted(
            [e for e in emb_set if e != "none"],
            key=lambda x: x.lower(),
        ) + [e for e in emb_set if e == "none"]

        cell: dict[tuple[str, str, str], dict] = {}
        for job in jobs:
            m = job["method"]
            ek = self._embedding_key(m)
            pair = (m["stage2_dynamics"], m["stage3_selection"])
            cell[(ek, pair[0], pair[1])] = job

        col_labels = [f"{s2} + {s3}" for s2, s3 in col_set]

        return {
            "columns": col_set,
            "row_groups": emb_ordered,
            "cell": cell,
            "col_labels": col_labels,
        }

    def _run_postprocessing(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO POST-PROCESAMIENTO: GRAFICOS COMPUESTOS")
        logger.info("=" * 70)

        # Agrupar jobs por subject
        groups: dict[str, list[dict]] = {}
        for job in self.all_jobs:
            groups.setdefault(job["subject"], []).append(job)

        total_groups = len(groups)
        generated = 0
        missing = 0
        partial = 0

        for group_idx, (subject, jobs) in enumerate(groups.items(), 1):
            logger.info("[%d/%d] Procesando %s ...", group_idx, total_groups, subject)

            matrix = self._organize_method_matrix(jobs)
            n_cols = len(matrix["columns"])
            n_row_groups = len(matrix["row_groups"])
            n_rows = n_row_groups * 2

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5 * n_cols, 4 * n_rows),
                constrained_layout=True,
            )

            if n_rows == 1 and n_cols == 1:
                axes = axes.reshape(1, 1)
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            n_found = 0
            expected_images = 2 * n_cols * n_row_groups

            for rg_idx, emb_key in enumerate(matrix["row_groups"]):
                row_pot = rg_idx * 2
                row_rot = rg_idx * 2 + 1

                for col_idx, (s2, s3) in enumerate(matrix["columns"]):
                    lookup = (emb_key, s2, s3)
                    job = matrix["cell"].get(lookup)
                    col_title = matrix["col_labels"][col_idx]

                    if job is None:
                        for r in (row_pot, row_rot):
                            axes[r, col_idx].text(
                                0.5, 0.5, "N/A",
                                ha="center", va="center", fontsize=10,
                                color="lightgray",
                                transform=axes[r, col_idx].transAxes,
                            )
                            axes[r, col_idx].axis("off")
                        continue

                    out_dir = self._get_output_dir(job)

                    # Potential
                    pot_path = out_dir / self.potential_img
                    if pot_path.exists():
                        try:
                            img = Image.open(pot_path)
                            axes[row_pot, col_idx].imshow(img)
                            n_found += 1
                        except Exception as exc:
                            logger.warning("  No se pudo cargar %s: %s", pot_path, exc)
                            axes[row_pot, col_idx].text(
                                0.5, 0.5, f"ERROR\n{pot_path.name}",
                                ha="center", va="center", fontsize=8, color="red",
                                transform=axes[row_pot, col_idx].transAxes,
                            )
                    else:
                        logger.warning("  Falta: %s", pot_path)
                        axes[row_pot, col_idx].text(
                            0.5, 0.5, f"MISSING\n{pot_path.name}",
                            ha="center", va="center", fontsize=8, color="gray",
                            transform=axes[row_pot, col_idx].transAxes,
                        )
                    axes[row_pot, col_idx].axis("off")

                    # Non-conservative force
                    ncf_path = out_dir / self.noncons_img
                    if ncf_path.exists():
                        try:
                            img = Image.open(ncf_path)
                            axes[row_rot, col_idx].imshow(img)
                            n_found += 1
                        except Exception as exc:
                            logger.warning("  No se pudo cargar %s: %s", ncf_path, exc)
                            axes[row_rot, col_idx].text(
                                0.5, 0.5, f"ERROR\n{ncf_path.name}",
                                ha="center", va="center", fontsize=8, color="red",
                                transform=axes[row_rot, col_idx].transAxes,
                            )
                    else:
                        logger.warning("  Falta: %s", ncf_path)
                        axes[row_rot, col_idx].text(
                            0.5, 0.5, f"MISSING\n{ncf_path.name}",
                            ha="center", va="center", fontsize=8, color="gray",
                            transform=axes[row_rot, col_idx].transAxes,
                        )
                    axes[row_rot, col_idx].axis("off")

                # Column titles
                for col_idx, col_title in enumerate(matrix["col_labels"]):
                    axes[row_pot, col_idx].set_title(
                        col_title, fontsize=10, fontweight="bold", pad=8,
                    )

            # Row labels
            for rg_idx, emb_key in enumerate(matrix["row_groups"]):
                emb_name = self._embedding_display_name(emb_key)
                row_pot = rg_idx * 2
                row_rot = rg_idx * 2 + 1

                axes[row_pot, 0].set_ylabel(
                    f"{emb_name}\n\nPotential (U)",
                    fontsize=11, fontweight="bold", rotation=90, labelpad=15,
                )
                axes[row_rot, 0].set_ylabel(
                    "Non-cons.\nforce (v)",
                    fontsize=11, fontweight="bold", rotation=90, labelpad=15,
                )

            # Separators
            if n_row_groups > 1:
                for rg_idx in range(1, n_row_groups):
                    sep_row = rg_idx * 2 - 0.5
                    fig.add_artist(plt.matplotlib.lines.Line2D(
                        [0.02, 0.98], [sep_row / n_rows, sep_row / n_rows],
                        transform=fig.transFigure,
                        color="black", linewidth=1.5, linestyle="--",
                        alpha=0.5,
                    ))

            # Title
            fig.suptitle(
                f"{subject} -- Methods comparison",
                fontsize=14, fontweight="bold", y=1.01,
            )

            # Save
            save_dir = (
                self.output_dir
                / "ludovico_01"
                / subject
                / _FIXED_SESSION
                / "methods_comparison"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            composite_name = f"{self.composite_prefix}_{subject}.png"
            save_path = save_dir / composite_name

            try:
                fig.savefig(
                    str(save_path), dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none",
                )
                plt.close(fig)
                logger.info("  Guardado: %s", save_path)
                generated += 1
            except Exception as exc:
                logger.error("  Error guardando %s: %s", save_path, exc)
                plt.close(fig)

            if n_found == 0:
                missing += 1
            elif n_found < expected_images:
                partial += 1

        # Summary
        complete = generated - partial - missing
        logger.info("")
        logger.info("=" * 70)
        logger.info("  POST-PROCESAMIENTO COMPLETADO")
        logger.info("=" * 70)
        logger.info("  Grupos procesados   : %d", total_groups)
        logger.info("  Completos           : %d", complete)
        logger.info("  Parciales           : %d", partial)
        logger.info("  Sin imagenes        : %d", missing)
        logger.info("  Archivos generados  : %d", generated)

    # ------------------------------------------------------------------
    # Orquestacion principal
    # ------------------------------------------------------------------

    def run(self) -> int:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.all_jobs:
            logger.error("No se generaron jobs. Revisa el JSON de parametros.")
            return 1

        todo = self._filter_todo(self.all_jobs)
        total = len(todo)

        if total == 0:
            logger.info("Todos los jobs ya estan completados.")
            if self.run_postprocess:
                self._run_postprocessing()
            return 0

        logger.info("Total a ejecutar: %d / %d", total, len(self.all_jobs))

        completed = 0
        failed = 0

        if self.max_workers > 1:
            completed, failed = self._run_parallel(todo, total)
        else:
            completed, failed = self._run_sequential(todo, total)

        self._save_checkpoint()

        logger.info("=" * 70)
        logger.info(
            "BATCH COMPLETADO -- OK: %d | Fallos: %d | Total: %d",
            completed, failed, total,
        )
        logger.info("Log CSV: %s", self.log_file)

        if self.run_postprocess:
            self._run_postprocessing()

        return 0 if failed == 0 else 1

    def _run_sequential(self, todo: list[dict], total: int) -> tuple[int, int]:
        completed = 0
        failed = 0

        for idx, job in enumerate(todo, start=1):
            logger.info("")
            logger.info("-" * 70)
            logger.info("Progreso: %d / %d", idx, total)
            logger.info("-" * 70)

            key, success = self._run_single_job(job)
            if success:
                self.checkpoint.add(key)
                completed += 1
            else:
                failed += 1

            self._save_checkpoint()

            if idx < total and self.delay > 0:
                logger.debug("Pausa %.1f s...", self.delay)
                time.sleep(self.delay)

        return completed, failed

    def _run_parallel(self, todo: list[dict], total: int) -> tuple[int, int]:
        completed = 0
        failed = 0

        logger.info("Modo PARALELO con %d workers", self.max_workers)

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {
                executor.submit(self._run_single_job, job): job for job in todo
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    key, success = future.result()
                    if success:
                        self.checkpoint.add(key)
                        completed += 1
                    else:
                        failed += 1
                except Exception as exc:
                    logger.error(
                        "FUTURE EXCEPTION | %s [%s]: %s",
                        job["subject"], job["method_label"], exc,
                    )
                    failed += 1

                self._save_checkpoint()
                logger.info(
                    "Progreso: %d / %d completados", completed + failed, total
                )

        return completed, failed


# ===========================================================================
# ENTRY POINT
# ===========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch multi-metodo para el pipeline IgA (Ludovico_01, CSV)."
    )
    parser.add_argument(
        "--params-json", type=str, default=None,
        help=(
            "Ruta al archivo JSON de parametros. "
            f"Default: {DEFAULT_PARAMS_JSON}"
        ),
    )
    parser.add_argument(
        "--mode-map-path", type=str, default=None,
        help=(
            "Ruta al mode_map.json generado por el batch CDHSA de Ludovico. "
            "Se usa cuando algun metodo tiene stage2_dynamics='cdhsa_specific_modes'. "
            "Si no se pasa, se intenta auto-resolver desde BASE_PARAMS_FILE/params."
        ),
    )
    args = parser.parse_args()

    # Determinar ruta del JSON
    json_path = Path(args.params_json) if args.params_json else DEFAULT_PARAMS_JSON
    if os.environ.get("BATCH_LUDOVICO_PARAMS_JSON"):
        json_path = Path(os.environ["BATCH_LUDOVICO_PARAMS_JSON"])

    # Cargar parametros
    params = _load_params(json_path)
    logger.info("Parametros cargados desde: %s", json_path)

    # CLI override para mode_map_path
    if args.mode_map_path:
        params["mode_map_path"] = args.mode_map_path

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = Ludovico01BatchRunner(params)
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())

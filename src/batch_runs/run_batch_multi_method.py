#!/usr/bin/env python3
"""
run_batch_multi_method.py
===========================
Ejecutor de batch multi-metodo para el pipeline IgA (version nueva con API de 3 etapas).

A diferencia de la version anterior, este script:

1. **Lee toda la configuracion desde un JSON** (``multimethod_call_params.json``),
   lo que permite cambiar sujetos, sesiones, tareas, metodos y parametros de
   pipeline sin tocar el codigo.

2. **Usa la nueva API del orquestador** (``--stage1-embedding``,
   ``--stage2-dynamics``, ``--stage3-selection`` y sus respectivos
   ``--stageX-params``) en lugar del legacy ``--scoring-method``.

3. **Post-procesamiento automatico**: al finalizar todas las corridas,
   genera graficos compuestos en forma de **matriz** por cada
   (sujeto, sesion, tarea), donde:
   - **Columnas** = combinaciones unicas de (stage2_dynamics, stage3_selection).
   - **Filas** = grupos de stage1_embedding (con/sin embedding), cada grupo
     con 2 sub-filas: potencial 2D y fuerza no-conservativa.
   - Ejemplo con 4 combos de stage2/3 y 2 grupos de embedding (hankel, none):
     una matriz 4x4 donde las filas 1-2 son con hankel y las 3-4 sin hankel.

   El layout se adapta automaticamente a cualquier numero de metodos,
   combinaciones de stage2/3 y grupos de embedding.

   Los graficos se nombran dinamicamente incluyendo sujeto, sesion y tarea:
   ``methods_comparison_{subject}_{session}_{task}.png``

Configuracion
--------------
Todo se controla mediante el archivo JSON (por defecto en
``BASE_PARAMS_FILE / multimethod_call_params.json``). Los campos del JSON son:

```json
{
  "pipeline_module": "src.pipelines.test_iga_from_eeg_latent_test_retest_gedai",
  "subjects": { "start": 1, "end": 5 },
  "sessions": ["session1"],
  "tasks": ["eyesclosed", ...],
  "time_window": { "t_start": 0.0, "t_end": 300.0 },
  "methods": [
    {
      "label": "hankel_dmd",
      "stage1_embedding": "hankel",
      "stage1_params": {},
      "stage2_dynamics": "dmd",
      "stage2_params": {},
      "stage3_selection": "top_n",
      "stage3_params": {}
    }
  ],
  "shared_params": { ... },
  "execution": { "delay": 2.0, "max_workers": 1, ... },
  "postprocess": { ... }
}
```

Tambien se puede sobreescribir la ruta del JSON con la variable de entorno:
  ``BATCH_MM_PARAMS_JSON``

Uso
---
Desde la raiz del proyecto::

    # Usando defaults (lee el JSON de BASE_PARAMS_FILE)
    python -m src.batch_runs.run_batch_multi_method

    # Apuntando a otro JSON
    BATCH_MM_PARAMS_JSON=/path/to/other.json python -m src.batch_runs.run_batch_multi_method

    # Cambiar log level
    BATCH_MM_LOG_LEVEL=DEBUG python -m src.batch_runs.run_batch_multi_method
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
        DB_TEST_RETEST_GEDAI_PATH,
        BASE_PARAMS_FILE,
        LOGGING_BASE_PATH,
        LOGGING_LEVEL,
        DEFAULT_BATCH_RUNS_WORKERS
    )
except ImportError as _exc:
    print(
        f"[ERROR] No se pudieron importar los modulos del proyecto. "
        f"Asegurate de ejecutar este script desde la raiz del repositorio "
        f"o de que 'src' este en PYTHONPATH.\n{_exc}",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuracion de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOGGING_LEVEL.upper(),
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_multi_method")

logging_path = (
    Path(LOGGING_BASE_PATH + "/batch_multi_method") if LOGGING_BASE_PATH else None
)
if not logging_path:
    logger.warning("No se definio LOGGING_BASE_PATH; logs no se guardaran en archivo.")
else:
    os.makedirs(logging_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Ruta por defecto del JSON de parametros
# ---------------------------------------------------------------------------
DEFAULT_PARAMS_JSON = Path("./src/batch_runs/multimethod_call_params.json")

# ---------------------------------------------------------------------------
# Lectura del JSON de parametros
# ---------------------------------------------------------------------------


def _load_params(json_path: Path) -> dict:
    """Carga y valida el JSON de parametros del batch."""
    if not json_path.exists():
        logger.error("Archivo JSON de parametros no encontrado: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    # Validaciones minimas
    required_top_keys = ["subjects", "sessions", "tasks", "methods", "shared_params"]
    for key in required_top_keys:
        if key not in params:
            logger.error("Falta la clave requerida '%s' en el JSON", key)
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
    """Genera el label de spec igual que el orquestador: s1+s2+s3."""
    s1 = method.get("stage1_embedding") or "none"
    return f"{s1}+{method['stage2_dynamics']}+{method['stage3_selection']}"


def _spec_hash(method: dict, shared: dict) -> str:
    """Genera el hash corto igual que el orquestador."""
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
    """Genera la lista de jobs a partir del JSON de parametros.

    Cada job contiene toda la informacion necesaria para construir
    la llamada al orquestador: sujetos, sesion, tarea, metodo, y
    los parametros de pipeline completos.
    """
    subj_cfg = params["subjects"]
    start = subj_cfg["start"]
    end = subj_cfg["end"]
    sessions = params["sessions"]
    tasks = params["tasks"]
    methods = params["methods"]
    tw = params["time_window"]
    shared = params["shared_params"]

    jobs: list[dict] = []
    for subj_idx in range(start, end + 1):
        subject = f"sub-{subj_idx:02d}"
        for session in sessions:
            for task in tasks:
                for method in methods:
                    jobs.append({
                        "subject": subject,
                        "session": session,
                        "task": task,
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


class MultiMethodBatchRunner:
    """Orquesta la ejecucion multi-metodo del pipeline IgA (nueva API 3 etapas)."""

    CSV_FIELDS = [
        "timestamp", "subject", "session", "task", "method_label",
        "spec_label", "t_start", "t_end", "success", "returncode",
        "elapsed_s", "command",
    ]

    def __init__(self, params: dict) -> None:
        self.params = params
        self.pipeline_module = params.get(
            "pipeline_module",
            "src.pipelines.test_iga_from_eeg_latent_test_retest_gedai",
        )
        self.shared = params["shared_params"]
        self.exec_cfg = params.get("execution", {})
        self.post_cfg = params.get("postprocess", {})

        # Rutas
        self.db_path = Path(DB_TEST_RETEST_GEDAI_PATH)
        self.output_dir = Path(BASE_RESULTS_PATH)
        self.cache_dir = Path(BASE_CACHE_PATH)

        # Ejecucion
        self.delay: float = self.exec_cfg.get("delay", 2.0)
        self.max_workers: int = self.exec_cfg.get("max_workers", DEFAULT_BATCH_RUNS_WORKERS)
        self.run_postprocess: bool = self.exec_cfg.get("run_postprocess", True)
        self.ignore_cache: bool = self.shared.get("ignore_cache", False)

        # Logging level override
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
        self.log_file = logging_path / f"batch_log_multi_method_{ts}.csv"
        self._init_csv_log()

        # Generar jobs
        self.all_jobs = _generate_jobs(params)
        self._print_banner()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        n_subjects = len({j["subject"] for j in self.all_jobs})
        n_sessions = len({j["session"] for j in self.all_jobs})
        n_tasks = len({j["task"] for j in self.all_jobs})
        n_methods = len({j["method_label"] for j in self.all_jobs})

        logger.info("=" * 70)
        logger.info("  BATCH MULTI-METHOD -- Test-Retest Gedai (EEGLAB .set/.fdt)")
        logger.info("  [NUEVA API 3 ETAPAS]")
        logger.info("=" * 70)
        logger.info("  Pipeline    : %s", self.pipeline_module)
        logger.info("  DB path     : %s", self.db_path)
        logger.info("  Output dir  : %s", self.output_dir)
        logger.info("  Cache dir   : %s", self.cache_dir)
        logger.info("  Checkpoint  : %s", self._checkpoint_path())
        logger.info("  ---")
        subj_cfg = self.params["subjects"]
        logger.info("  Subjects    : %d (sub-%02d .. sub-%02d)",
                    n_subjects, subj_cfg["start"], subj_cfg["end"])
        logger.info("  Sessions    : %s", self.params["sessions"])
        logger.info("  Tasks       : %s", self.params["tasks"])
        logger.info("  Methods     : %s",
                    [m["label"] for m in self.params["methods"]])
        tw = self.params["time_window"]
        logger.info("  Time window : %.1f s -> %.1f s", tw["t_start"], tw["t_end"])
        logger.info("  Latent dim  : %d", self.shared.get("latent_dim", 2))
        logger.info("  ---")
        logger.info("  Total jobs  : %d (%d subj x %d sess x %d task x %d meth)",
                    len(self.all_jobs), n_subjects, n_sessions, n_tasks, n_methods)
        logger.info("  Delay       : %.1f s", self.delay)
        logger.info("  Ignore cache: %s", self.ignore_cache)
        logger.info("  Max workers : %d (%s)",
                    self.max_workers, "paralelo" if self.max_workers > 1 else "secuencial")
        logger.info("  Post-process: %s", self.run_postprocess)
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        return self.cache_dir / "batch_checkpoint_multi_method.json"

    def _load_checkpoint(self) -> set[str]:
        cp = self._checkpoint_path() if hasattr(self, "cache_dir") else \
            Path(BASE_CACHE_PATH) / "batch_checkpoint_multi_method.json"
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
    def _checkpoint_key(subject: str, session: str, task: str,
                        method_label: str, t_start: str, t_end: str) -> str:
        return f"{subject}|{session}|{task}|{method_label}|{t_start}|{t_end}"

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
                    "session": job["session"],
                    "task": job["task"],
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
                job["subject"], job["session"], job["task"],
                job["method_label"], job["t_start"], job["t_end"],
            )
            if key in self.checkpoint:
                logger.debug(
                    "SKIP (checkpoint): %s/%s/%s [%s]",
                    job["subject"], job["session"], job["task"],
                    job["method_label"],
                )
                continue
            todo.append(job)

        if skipped := len(jobs) - len(todo):
            logger.info("Jobs ya completados (skip): %d / %d", skipped, len(jobs))
        return todo

    # ------------------------------------------------------------------
    # Ruta de salida para un job (misma logica que el orquestador nuevo)
    # ------------------------------------------------------------------

    def _get_output_dir(self, job: dict) -> Path:
        """Devuelve el directorio de salida donde el orquestador guarda resultados.

        En vez de calcular el spec_hash localmente (lo cual falla porque el
        orquestador inyecta defaults del CLI antes de hashear), busca en
        disco la carpeta que coincida con el ``spec_label``.

        Patron del orquestador:
        ``test_retest_gedai/{subject}/{session}/
         {latent_dim}_latent_dim_{spec_label}_{spec_hash}/
         from{t_start}s_to_{t_end}s_{task}``
        """
        method = job["method"]
        shared = job["shared"]
        latent_dim = shared.get("latent_dim", 2)
        spec_label = _spec_label(method)

        session_dir = (
            self.output_dir
            / "test_retest_gedai"
            / job["subject"]
            / job["session"]
        )

        # Buscar la carpeta del metodo por spec_label (ignorando el hash)
        prefix = f"{latent_dim}_latent_dim_{spec_label}_"
        if session_dir.is_dir():
            candidates = [
                d for d in session_dir.iterdir()
                if d.is_dir() and d.name.startswith(prefix)
            ]
            if len(candidates) == 1:
                method_dir = candidates[0]
            elif len(candidates) > 1:
                # Multiples matches: intentar resolver por t_start/t_end/task
                task_sub = f"from{job['t_start']}s_to_{job['t_end']}s_{job['task']}"
                for c in candidates:
                    if (c / task_sub).is_dir():
                        method_dir = c
                        break
                else:
                    # Fallback: usar la mas reciente
                    method_dir = max(candidates, key=lambda p: p.stat().st_mtime)
                    logger.warning(
                        "Multiples carpetas para %s, usando la mas reciente: %s",
                        spec_label, method_dir.name,
                    )
            else:
                # No encontrada: retornar path estimado para que el warning
                # del post-procesamiento muestre que falta
                method_dir = session_dir / f"{prefix}????????"
        else:
            method_dir = session_dir / f"{prefix}????????"

        return method_dir / f"from{job['t_start']}s_to_{job['t_end']}s_{job['task']}"

    # ------------------------------------------------------------------
    # Construccion del comando (nueva API 3 etapas)
    # ------------------------------------------------------------------

    def _build_command(self, job: dict) -> list[str]:
        """Construye el comando para llamar al orquestador con la nueva API.

        Usa ``--stage1-embedding``, ``--stage2-dynamics``, ``--stage3-selection``
        y ``--stageX-params`` en lugar del legacy ``--scoring-method``.
        """
        method = job["method"]
        shared = job["shared"]

        cmd = [
            sys.executable,
            "-m", self.pipeline_module,
            "--subject", job["subject"],
            "--session", job["session"],
            "--task", job["task"],
            "--t-start", job["t_start"],
            "--t-end", job["t_end"],
            "--latent-dim", str(shared.get("latent_dim", 2)),
            "--l-freq", str(shared.get("l_freq", 1.0)),
            "--h-freq", str(shared.get("h_freq", 40.0)),
            "--ica-method", shared.get("ica_method", "picard"),
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

        # Diffusion maps params (solo se pasan si no estan ya en stage2_params)
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

        # --- KM bins ---
        if shared.get("km_bins") is not None:
            cmd.extend(["--km-bins", str(shared["km_bins"])])

        # --- Flags booleanos ---
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
            job["subject"], job["session"], job["task"],
            job["method_label"], job["t_start"], job["t_end"],
        )

        try:
            cmd = self._build_command(job)
        except Exception as exc:
            logger.error(
                "Error construyendo comando para %s/%s/%s [%s]: %s",
                job["subject"], job["session"], job["task"],
                job["method_label"], exc,
            )
            return key, False

        logger.info(
            "RUN | %s/%s/%s | method=%s (%s) | [%s-%s] s",
            job["subject"], job["session"], job["task"],
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

            # Stream stdout en tiempo real
            for line in proc.stdout:
                logger.info("  [PIPE] %s", line.rstrip())

            proc.wait()
            elapsed = time.time() - t0
            success = proc.returncode == 0

            self._write_csv_log(job, success, proc.returncode, elapsed, cmd)

            if success:
                logger.info(
                    "OK | %s/%s/%s [%s] (%.1f s)",
                    job["subject"], job["session"], job["task"],
                    job["method_label"], elapsed,
                )
            else:
                logger.error(
                    "ERROR | %s/%s/%s [%s] -- codigo %d",
                    job["subject"], job["session"], job["task"],
                    job["method_label"], proc.returncode,
                )

            return key, success

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "EXCEPTION | %s/%s/%s [%s]: %s",
                job["subject"], job["session"], job["task"],
                job["method_label"], exc,
            )
            self._write_csv_log(job, False, -1, elapsed, cmd if 'cmd' in dir() else [])
            return key, False

    # ------------------------------------------------------------------
    # Post-procesamiento: graficos compuestos en matriz
    # ------------------------------------------------------------------

    @staticmethod
    def _embedding_key(method: dict) -> str:
        """Clave normalizada del stage1 embedding para agrupar filas."""
        s1 = method.get("stage1_embedding")
        return s1 if s1 and s1 != "none" else "none"

    @staticmethod
    def _embedding_display_name(key: str) -> str:
        """Nombre legible para el grupo de embedding (labels de fila)."""
        return key if key != "none" else "No embedding"

    def _organize_method_matrix(self, jobs: list[dict]) -> dict:
        """Organiza los jobs de un grupo (subject, session, task) en una
        estructura matricial: columnas = combos stage2+stage3,
        filas = grupos de stage1 embedding.

        Returns
        -------
        dict with keys:
            - ``columns``: lista de (stage2, stage3) ordenadas
            - ``row_groups``: lista de embedding keys ordenadas
              (con embedding primero, luego sin embedding)
            - ``cell``: dict  ``(emb_key, s2, s3) -> job``
            - ``col_labels``: lista de strings para titulos de columna
        """
        # Descubrir columnas: combinaciones unicas (stage2, stage3)
        col_set: list[tuple[str, str]] = []
        seen_cols: set[tuple[str, str]] = set()
        for job in jobs:
            m = job["method"]
            pair = (m["stage2_dynamics"], m["stage3_selection"])
            if pair not in seen_cols:
                seen_cols.add(pair)
                col_set.append(pair)

        # Descubrir row groups: embedding keys unicos
        emb_set: list[str] = []
        seen_emb: set[str] = set()
        for job in jobs:
            ek = self._embedding_key(job["method"])
            if ek not in seen_emb:
                seen_emb.add(ek)
                emb_set.append(ek)

        # Ordenar: embeddings con nombre primero (hankel, ...), luego "none"
        emb_ordered = sorted(
            [e for e in emb_set if e != "none"],
            key=lambda x: x.lower(),
        ) + [e for e in emb_set if e == "none"]

        # Build lookup: (emb_key, s2, s3) -> job
        cell: dict[tuple[str, str, str], dict] = {}
        for job in jobs:
            m = job["method"]
            ek = self._embedding_key(m)
            pair = (m["stage2_dynamics"], m["stage3_selection"])
            cell[(ek, pair[0], pair[1])] = job

        # Column labels
        col_labels = [f"{s2} + {s3}" for s2, s3 in col_set]

        return {
            "columns": col_set,
            "row_groups": emb_ordered,
            "cell": cell,
            "col_labels": col_labels,
        }

    def _run_postprocessing(self) -> None:
        """Genera graficos compuestos en matriz por (sujeto, sesion, tarea).

        Layout adaptable:
          - **Columnas** = combinaciones unicas de (stage2_dynamics, stage3_selection).
          - **Filas** = grupos de stage1_embedding, cada uno con 2 sub-filas:
            sub-fila par = potencial 2D, sub-fila impar = fuerza no-conservativa.
          - Total de filas = len(row_groups) * 2.
          - Total de columnas = len(stage23_combos).

        Ejemplo con hankel/none y 4 combos stage2/3:
          Fila 0: [hankel] potential       para cada combo de columna
          Fila 1: [hankel] rotational     para cada combo de columna
          Fila 2: [none]  potential       para cada combo de columna
          Fila 3: [none]  rotational     para cada combo de columna

        El nombre del archivo incluye sujeto, sesion y tarea:
          methods_comparison_{subject}_{session}_{task}.png
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO POST-PROCESAMIENTO: GRAFICOS COMPUESTOS (MATRIZ)")
        logger.info("=" * 70)

        # Agrupar jobs por (subject, session, task)
        groups: dict[tuple[str, str, str], list[dict]] = {}
        for job in self.all_jobs:
            key = (job["subject"], job["session"], job["task"])
            groups.setdefault(key, []).append(job)

        total_groups = len(groups)
        generated = 0
        missing = 0
        partial = 0

        for group_idx, ((subject, session, task), jobs) in enumerate(groups.items(), 1):
            logger.info(
                "[%d/%d] Procesando %s/%s/%s ...",
                group_idx, total_groups, subject, session, task,
            )

            # Organizar en matriz
            matrix = self._organize_method_matrix(jobs)
            n_cols = len(matrix["columns"])
            n_row_groups = len(matrix["row_groups"])
            n_rows = n_row_groups * 2

            logger.info(
                "  Layout: %d filas x %d columnas (%d embedding groups x 2)",
                n_rows, n_cols, n_row_groups,
            )

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5 * n_cols, 4 * n_rows),
                constrained_layout=True,
            )

            # Asegurar que axes sea siempre 2D
            if n_rows == 1 and n_cols == 1:
                axes = axes.reshape(1, 1)
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            n_found = 0
            expected_images = 2 * n_cols * n_row_groups

            for rg_idx, emb_key in enumerate(matrix["row_groups"]):
                # Sub-filas para este grupo de embedding
                row_pot = rg_idx * 2      # fila de potencial
                row_rot = rg_idx * 2 + 1  # fila de rotacional

                for col_idx, (s2, s3) in enumerate(matrix["columns"]):
                    lookup = (emb_key, s2, s3)
                    job = matrix["cell"].get(lookup)
                    col_title = matrix["col_labels"][col_idx]

                    if job is None:
                        # Celda vacia: no hay job para esta combinacion
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

                    # --- Sub-fila de potencial ---
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

                    # --- Sub-fila de fuerza no-conservativa ---
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

                # --- Titulos de columna (solo en la primera sub-fila del grupo) ---
                for col_idx, col_title in enumerate(matrix["col_labels"]):
                    axes[row_pot, col_idx].set_title(
                        col_title, fontsize=10, fontweight="bold", pad=8,
                    )

            # --- Etiquetas de fila (grupo de embedding + tipo de imagen) ---
            for rg_idx, emb_key in enumerate(matrix["row_groups"]):
                emb_name = self._embedding_display_name(emb_key)
                row_pot = rg_idx * 2
                row_rot = rg_idx * 2 + 1

                # Etiqueta del grupo de embedding centrada entre las 2 sub-filas
                # Se pone en la sub-fila de potencial como ylabel
                axes[row_pot, 0].set_ylabel(
                    f"{emb_name}\n\nPotential (U)",
                    fontsize=11, fontweight="bold", rotation=90, labelpad=15,
                )
                axes[row_rot, 0].set_ylabel(
                    "Non-cons.\nforce (v)",
                    fontsize=11, fontweight="bold", rotation=90, labelpad=15,
                )

            # --- Lineas separadoras horizontales entre grupos de embedding ---
            if n_row_groups > 1:
                for rg_idx in range(1, n_row_groups):
                    sep_row = rg_idx * 2 - 0.5
                    fig.add_artist(plt.matplotlib.lines.Line2D(
                        [0.02, 0.98], [sep_row / n_rows, sep_row / n_rows],
                        transform=fig.transFigure,
                        color="black", linewidth=1.5, linestyle="--",
                        alpha=0.5,
                    ))

            # --- Titulo general ---
            fig.suptitle(
                f"{subject} | {session} | {task}  --  Methods comparison",
                fontsize=14, fontweight="bold", y=1.01,
            )

            # --- Guardar en subcarpeta methods_comparison a nivel de sesion ---
            save_dir = (
                self.output_dir
                / "test_retest_gedai"
                / subject
                / session
                / "methods_comparison"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            composite_name = f"{self.composite_prefix}_{subject}_{session}_{task}.png"
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

            # Conteo parcial
            if n_found == 0:
                missing += 1
            elif n_found < expected_images:
                partial += 1

        # --- Resumen ---
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
                        "FUTURE EXCEPTION | %s/%s/%s [%s]: %s",
                        job["subject"], job["session"], job["task"],
                        job["method_label"], exc,
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
        description="Batch multi-metodo para el pipeline IgA (nueva API 3 etapas). "
                    "Lee la configuracion desde un JSON.",
    )
    parser.add_argument(
        "--params-json", type=str, default=None,
        help=(
            "Ruta al archivo JSON de parametros. "
            f"Default: {DEFAULT_PARAMS_JSON}"
        ),
    )
    args = parser.parse_args()

    # Determinar ruta del JSON
    json_path = Path(args.params_json) if args.params_json else DEFAULT_PARAMS_JSON
    if os.environ.get("BATCH_MM_PARAMS_JSON"):
        json_path = Path(os.environ["BATCH_MM_PARAMS_JSON"])

    # Cargar parametros
    params = _load_params(json_path)
    logger.info("Parametros cargados desde: %s", json_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = MultiMethodBatchRunner(params)
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())

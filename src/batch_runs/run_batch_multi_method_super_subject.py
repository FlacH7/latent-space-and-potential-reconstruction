#!/usr/bin/env python3
"""
run_batch_super_subject.py
============================
Batch executor for the **super-subject** IgA pipeline.

A *super-subject* is the time-axis concatenation of *N* individual
subjects' EEG recordings (same session, same task), processed as a
single recording by the IgA pipeline.  This batch runner iterates over
(super-subject x session x task x method) combinations and dispatches
each as a subprocess call to::

    src.pipelines.test_iga_from_eeg_latent_test_retest_gedai_super_subject

It is a sibling of ``run_batch_multi_method.py`` and shares its
structure (checkpoint, CSV log, post-processing composite matrix),
differing only in that:

* The ``subjects`` block of the JSON is replaced by a ``super_subjects``
  block whose fields are::

      "super_subjects": {
          "selected": [1, 2, 3],         # which super-subjects to run
          "subjects_per_super_subject": 20,
          "subject_start_offset": 1,
          "total_subjects": 60,           # informational only
          "groups": {                     # OPTIONAL explicit override
              "1": [1, 2, ..., 20],
              "2": [21, ..., 40],
              "3": [41, ..., 60]
          }
      }

* Output dir root is ``test_retest_gedai_super_subject`` (instead of
  ``test_retest_gedai``), so per-super-subject results live under
  ``test_retest_gedai_super_subject/super_subject-{id}/{session}/...``.

* Checkpoint / CSV-log / composite-image keys use the super-subject id
  rather than the individual subject label.

Configuration
--------------
Everything is controlled by the JSON file (default in
``BASE_PARAMS_FILE / supersubject_call_params.json``).  The fields are::

    {
      "pipeline_module": "src.pipelines.test_iga_from_eeg_latent_test_retest_gedai_super_subject",
      "super_subjects": { ... },
      "sessions": ["session1", "session2", "session3"],
      "tasks": ["eyesclosed", ...],
      "time_window": { "t_start": 0.0, "t_end": 300.0 },
      "methods": [ ... ],
      "shared_params": { ... },
      "execution": { ... },
      "postprocess": { ... }
    }

The JSON path can also be overridden via the environment variable::

    BATCH_SS_PARAMS_JSON=/path/to/other.json python -m src.batch_runs.run_batch_super_subject

Usage
-----
From the project root::

    # Defaults (reads JSON from BASE_PARAMS_FILE)
    python -m src.batch_runs.run_batch_super_subject

    # Custom JSON
    BATCH_SS_PARAMS_JSON=/path/to/params.json python -m src.batch_runs.run_batch_super_subject

    # Verbose
    BATCH_SS_LOG_LEVEL=DEBUG python -m src.batch_runs.run_batch_super_subject
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
# Ensure package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # batch_runs/ -> src/ -> root

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
try:
    from src.utils.config import (
        BASE_CACHE_PATH,
        BASE_RESULTS_PATH,
        DB_TEST_RETEST_GEDAI_PATH,
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

# Local helper for super-subject -> subject-ids resolution (used by the
# batch to optionally pass --subject-ids to the pipeline when 'groups'
# is provided in the JSON).
from src.latent_space_extraction.super_subject_eeg import (
    resolve_super_subject_subject_ids,
    compute_channel_intersection,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOGGING_LEVEL.upper(),
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_super_subject")

logging_path = (
    Path(LOGGING_BASE_PATH + "/batch_super_subject") if LOGGING_BASE_PATH else None
)
if not logging_path:
    logger.warning("No se definio LOGGING_BASE_PATH; logs no se guardaran en archivo.")
else:
    os.makedirs(logging_path, exist_ok=True)

# ---------------------------------------------------------------------------
# Default JSON path
# ---------------------------------------------------------------------------
DEFAULT_PARAMS_JSON = Path("./src/batch_runs/supersubject_call_params.json")

# ---------------------------------------------------------------------------
# JSON loading + validation
# ---------------------------------------------------------------------------


def _load_params(json_path: Path) -> dict:
    """Load and validate the batch JSON parameters."""
    if not json_path.exists():
        logger.error("Archivo JSON de parametros no encontrado: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as fh:
        params = json.load(fh)

    # Minimal validation
    required_top_keys = [
        "super_subjects", "sessions", "tasks", "methods", "shared_params",
    ]
    for key in required_top_keys:
        if key not in params:
            logger.error("Falta la clave requerida '%s' en el JSON", key)
            sys.exit(1)

    ss_cfg = params["super_subjects"]
    if "selected" not in ss_cfg:
        logger.error("El bloque 'super_subjects' debe contener 'selected' (lista de ids).")
        sys.exit(1)
    if not isinstance(ss_cfg["selected"], list) or not ss_cfg["selected"]:
        logger.error("'super_subjects.selected' debe ser una lista no vacia.")
        sys.exit(1)

    # If 'groups' is not provided, 'subjects_per_super_subject' must be set
    if "groups" not in ss_cfg:
        if "subjects_per_super_subject" not in ss_cfg:
            logger.error(
                "Se requiere 'subjects_per_super_subject' cuando 'groups' "
                "no esta definido en 'super_subjects'."
            )
            sys.exit(1)
    else:
        # Validate that every selected id is in groups
        for sid in ss_cfg["selected"]:
            if str(sid) not in ss_cfg["groups"] and sid not in ss_cfg["groups"]:
                logger.error(
                    "super_subject_id=%s no esta en 'groups' (claves: %s)",
                    sid, list(ss_cfg["groups"].keys()),
                )
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
# Helpers (mirror of run_batch_multi_method, kept here for independence)
# ---------------------------------------------------------------------------


def _spec_label(method: dict) -> str:
    """Stage-chain label, e.g. ``hankel+dmd+top_n``."""
    s1 = method.get("stage1_embedding") or "none"
    return f"{s1}+{method['stage2_dynamics']}+{method['stage3_selection']}"


def _spec_hash(method: dict, shared: dict) -> str:
    """Short SHA-1 hash of the stage spec (mirrors the pipeline)."""
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


def _super_subject_label(super_subject_id: int) -> str:
    """BIDS-style label, e.g. ``super_subject-01``."""
    return f"super_subject-{super_subject_id:02d}"


def _resolve_subject_ids_for_job(
    super_subject_id: int, ss_cfg: dict,
) -> list[int] | None:
    """Resolve the explicit subject-ids for a super-subject, or None
    if the auto-resolution should be delegated to the pipeline.

    Returns
    -------
    list[int] | None
        * ``None`` when the JSON does not declare a ``groups`` block;
          the pipeline will auto-resolve from
          ``subjects_per_super_subject`` + ``subject_start_offset``.
        * A list of subject indices when ``groups`` is declared; the
          pipeline receives them via ``--subject-ids``.
    """
    groups = ss_cfg.get("groups")
    if groups is None:
        return None

    # JSON object keys are strings -> coerce to int when possible
    key = super_subject_id if super_subject_id in groups else str(super_subject_id)
    if key not in groups:
        raise KeyError(
            f"super_subject_id={super_subject_id} not in groups "
            f"(keys: {list(groups.keys())})"
        )
    return [int(x) for x in groups[key]]


# ===========================================================================
# JOB GENERATION
# ===========================================================================


def _generate_jobs(params: dict) -> list[dict]:
    """Generate the list of jobs from the JSON parameters.

    Each job contains everything needed to build the subprocess call:
    super-subject id, session, task, method, and the shared pipeline
    parameters.
    """
    ss_cfg = params["super_subjects"]
    selected_ids: list[int] = list(ss_cfg["selected"])
    sessions = params["sessions"]
    tasks = params["tasks"]
    methods = params["methods"]
    tw = params["time_window"]
    shared = params["shared_params"]

    jobs: list[dict] = []
    for sid in selected_ids:
        ss_label = _super_subject_label(sid)
        # Resolve subject_ids once per super_subject (so the same explicit
        # list is reused across sessions/tasks/methods).
        try:
            subject_ids = _resolve_subject_ids_for_job(sid, ss_cfg)
        except KeyError as exc:
            logger.error("%s -- este super-sujeto se omitira.", exc)
            continue

        for session in sessions:
            for task in tasks:
                for method in methods:
                    jobs.append({
                        "super_subject_id": sid,
                        "super_subject_label": ss_label,
                        "subject_ids": subject_ids,  # may be None
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


class SuperSubjectBatchRunner:
    """Orchestrates super-subject execution of the IgA pipeline."""

    CSV_FIELDS = [
        "timestamp", "super_subject", "session", "task", "method_label",
        "spec_label", "t_start", "t_end", "success", "returncode",
        "elapsed_s", "command",
    ]

    def __init__(self, params: dict) -> None:
        self.params = params
        self.pipeline_module = params.get(
            "pipeline_module",
            "src.pipelines.test_iga_from_eeg_latent_test_retest_gedai_super_subject",
        )
        self.shared = params["shared_params"]
        self.exec_cfg = params.get("execution", {})
        self.post_cfg = params.get("postprocess", {})
        self.ss_cfg = params["super_subjects"]

        # Paths
        self.db_path = Path(DB_TEST_RETEST_GEDAI_PATH)
        self.output_dir = Path(BASE_RESULTS_PATH)
        self.cache_dir = Path(BASE_CACHE_PATH)

        # Execution
        self.delay: float = self.exec_cfg.get("delay", 2.0)
        self.max_workers: int = self.exec_cfg.get(
            "max_workers", DEFAULT_BATCH_RUNS_WORKERS
        )
        self.run_postprocess: bool = self.exec_cfg.get("run_postprocess", True)
        self.ignore_cache: bool = self.shared.get("ignore_cache", False)

        # Logging level override
        log_level = self.exec_cfg.get("log_level", LOGGING_LEVEL)
        if log_level.upper() == "DEBUG":
            logger.setLevel(logging.DEBUG)

        # Images for post-processing
        self.potential_img = self.post_cfg.get("potential_img", "potential_2d.png")
        self.noncons_img = self.post_cfg.get(
            "nonconservative_img", "potential_2d_nonconservative_force.png"
        )
        self.composite_prefix = self.post_cfg.get(
            "composite_prefix", "methods_comparison"
        )

        # Checkpoint
        self.checkpoint: set[str] = self._load_checkpoint()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = logging_path / f"batch_log_super_subject_{ts}.csv"
        self._init_csv_log()

        # Generate jobs
        self.all_jobs = _generate_jobs(params)

        # ------------------------------------------------------------------
        # Pre-compute global channel intersections per (session, task)
        # ------------------------------------------------------------------
        # When any method uses cdhsa_specific_modes, the W_specific modes
        # were trained on the global channel intersection (e.g. 53
        # channels).  Different super-subjects may have different local
        # intersections (54, 55, ...), leading to Hankel matrices whose
        # row dimensionality does not match W_specific.  We compute the
        # global intersection once here and pass it to every job via
        # --channel-intersection so the pipeline can restrict the Raw
        # before building the Hankel.
        self._needs_channel_intersection = any(
            m["stage2_dynamics"] == "cdhsa_specific_modes"
            for m in params["methods"]
        )
        self._channel_intersections: dict[tuple[str, str], list[str]] = {}
        if self._needs_channel_intersection:
            self._precompute_channel_intersections(params)

        self._print_banner()

    # ------------------------------------------------------------------
    # Pre-compute global channel intersections
    # ------------------------------------------------------------------

    def _precompute_channel_intersections(self, params: dict) -> None:
        """Compute the global channel intersection for each (session, task).

        The result is stored in ``self._channel_intersections`` and
        injected into every job command via ``--channel-intersection``.
        """
        sessions = params["sessions"]
        tasks = params["tasks"]
        db_path = self.db_path

        logger.info("")
        logger.info("=" * 70)
        logger.info("  PRE-COMPUTING GLOBAL CHANNEL INTERSECTIONS")
        logger.info("=" * 70)

        for session in sessions:
            for task in tasks:
                key = (session, task)
                logger.info(
                    "  Computing intersection for %s/%s ...", session, task,
                )
                try:
                    intersection = compute_channel_intersection(
                        session,
                        task,
                        ss_cfg=self.ss_cfg,
                        db_path=db_path,
                        verbose=True,
                    )
                    self._channel_intersections[key] = intersection
                    logger.info(
                        "  Global intersection %s/%s: %d channels",
                        session, task, len(intersection),
                    )
                except Exception as exc:
                    logger.error(
                        "  Failed to compute intersection for %s/%s: %s. "
                        "CD-HSA jobs for this combo may fail.",
                        session, task, exc,
                    )

        logger.info("")
        logger.info("  Channel intersections computed for %d (session, task) combos.",
                     len(self._channel_intersections))
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        n_ss = len({j["super_subject_id"] for j in self.all_jobs})
        n_sessions = len({j["session"] for j in self.all_jobs})
        n_tasks = len({j["task"] for j in self.all_jobs})
        n_methods = len({j["method_label"] for j in self.all_jobs})

        logger.info("=" * 70)
        logger.info("  BATCH SUPER-SUBJECT -- Test-Retest Gedai (EEGLAB .set/.fdt)")
        logger.info("  [NUEVA API 3 ETAPAS -- concatenacion de sujetos]")
        logger.info("=" * 70)
        logger.info("  Pipeline    : %s", self.pipeline_module)
        logger.info("  DB path     : %s", self.db_path)
        logger.info("  Output dir  : %s", self.output_dir)
        logger.info("  Cache dir   : %s", self.cache_dir)
        logger.info("  Checkpoint  : %s", self._checkpoint_path())
        logger.info("  ---")
        logger.info(
            "  Super-subjects  : %d  -> %s",
            n_ss, sorted({j["super_subject_id"] for j in self.all_jobs}),
        )
        if "groups" in self.ss_cfg:
            logger.info("  Groups mode     : explicit (groups block in JSON)")
            for sid in self.ss_cfg["selected"]:
                ids = _resolve_subject_ids_for_job(sid, self.ss_cfg)
                logger.info(
                    "    %s -> %d subjects [%d..%d]",
                    _super_subject_label(sid), len(ids or []),
                    (ids or [0])[0], (ids or [0])[-1],
                )
        else:
            logger.info(
                "  Auto-resolve    : %d subjects per super-subject (offset %d)",
                self.ss_cfg.get("subjects_per_super_subject", 20),
                self.ss_cfg.get("subject_start_offset", 1),
            )
        logger.info("  Sessions        : %s", self.params["sessions"])
        logger.info("  Tasks           : %s", self.params["tasks"])
        logger.info("  Methods         : %s",
                    [m["label"] for m in self.params["methods"]])
        tw = self.params["time_window"]
        logger.info("  Time window     : %.1f s -> %.1f s (per subject)",
                    tw["t_start"], tw["t_end"])
        logger.info("  Latent dim      : %d", self.shared.get("latent_dim", 2))
        logger.info("  ---")
        logger.info(
            "  Total jobs  : %d (%d ss x %d sess x %d task x %d meth)",
            len(self.all_jobs), n_ss, n_sessions, n_tasks, n_methods,
        )
        logger.info("  Delay       : %.1f s", self.delay)
        logger.info("  Ignore cache: %s", self.ignore_cache)
        logger.info("  Max workers : %d (%s)",
                    self.max_workers,
                    "paralelo" if self.max_workers > 1 else "secuencial")
        logger.info("  Post-process: %s", self.run_postprocess)
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint_path(self) -> Path:
        return self.cache_dir / "batch_checkpoint_super_subject.json"

    def _load_checkpoint(self) -> set[str]:
        cp = (
            self._checkpoint_path() if hasattr(self, "cache_dir")
            else Path(BASE_CACHE_PATH) / "batch_checkpoint_super_subject.json"
        )
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
    def _checkpoint_key(super_subject_id: int, session: str, task: str,
                        method_label: str, t_start: str, t_end: str) -> str:
        return (f"ss{super_subject_id:02d}|{session}|{task}|"
                f"{method_label}|{t_start}|{t_end}")

    # ------------------------------------------------------------------
    # CSV log
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
                    "super_subject": job["super_subject_label"],
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
    # Filter out already-completed jobs
    # ------------------------------------------------------------------

    def _filter_todo(self, jobs: list[dict]) -> list[dict]:
        todo: list[dict] = []
        for job in jobs:
            key = self._checkpoint_key(
                job["super_subject_id"], job["session"], job["task"],
                job["method_label"], job["t_start"], job["t_end"],
            )
            if key in self.checkpoint:
                logger.debug(
                    "SKIP (checkpoint): %s/%s/%s [%s]",
                    job["super_subject_label"], job["session"], job["task"],
                    job["method_label"],
                )
                continue
            todo.append(job)

        if skipped := len(jobs) - len(todo):
            logger.info("Jobs ya completados (skip): %d / %d", skipped, len(jobs))
        return todo

    # ------------------------------------------------------------------
    # Output dir for a job (mirrors the pipeline's path scheme)
    # ------------------------------------------------------------------

    def _get_output_dir(self, job: dict) -> Path:
        """Return the directory where the pipeline saves results.

        Pattern:
        ``test_retest_gedai_super_subject/super_subject-{id}/{session}/
         {latent_dim}_latent_dim_{spec_label}_{spec_hash}/
         from{t_start}s_to_{t_end}s_{task}``

        Instead of computing the ``spec_hash`` locally (which would fail
        because the pipeline injects CLI defaults into the spec dict
        before hashing), we search the filesystem for the directory that
        matches the ``spec_label`` prefix and resolve the hash by
        filesystem lookup.  Mirrors the logic in
        ``run_batch_multi_method.py``.
        """
        method = job["method"]
        shared = job["shared"]
        latent_dim = shared.get("latent_dim", 2)
        spec_label = _spec_label(method)

        session_dir = (
            self.output_dir
            / "test_retest_gedai_super_subject"
            / job["super_subject_label"]
            / job["session"]
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
                # Multiple matches: resolve by t_start/t_end/task subfolder
                task_sub = f"from{job['t_start']}s_to_{job['t_end']}s_{job['task']}"
                for c in candidates:
                    if (c / task_sub).is_dir():
                        method_dir = c
                        break
                else:
                    # Fallback: use the most recently modified one
                    method_dir = max(candidates, key=lambda p: p.stat().st_mtime)
                    logger.warning(
                        "Multiples carpetas para %s, usando la mas reciente: %s",
                        spec_label, method_dir.name,
                    )
            else:
                # Not found: return estimated path so the warning message
                # in the post-processor shows what's missing.
                method_dir = session_dir / f"{prefix}????????"
        else:
            method_dir = session_dir / f"{prefix}????????"

        return method_dir / f"from{job['t_start']}s_to_{job['t_end']}s_{job['task']}"

    # ------------------------------------------------------------------
    # Build subprocess command
    # ------------------------------------------------------------------

    def _build_command(self, job: dict) -> list[str]:
        """Build the subprocess command for the super-subject pipeline."""
        method = job["method"]
        shared = job["shared"]

        cmd = [
            sys.executable,
            "-m", self.pipeline_module,
            "--super-subject", str(job["super_subject_id"]),
            "--session", job["session"],
            "--task", job["task"],
            "--t-start", job["t_start"],
            "--t-end", job["t_end"],
            "--latent-dim", str(shared.get("latent_dim", 2)),
            "--l-freq", str(shared.get("l_freq", 1.0)),
            "--h-freq", str(shared.get("h_freq", 40.0)),
            "--ica-method", shared.get("ica_method", "picard"),
        ]

        # Explicit subject-ids (when 'groups' is declared in JSON)
        if job.get("subject_ids") is not None:
            cmd.extend([
                "--subject-ids",
                json.dumps(job["subject_ids"]),
            ])
        else:
            # Auto-resolution params forwarded to the pipeline
            ss_cfg = self.ss_cfg
            cmd.extend([
                "--subjects-per-super-subject",
                str(ss_cfg.get("subjects_per_super_subject", 20)),
                "--subject-start-offset",
                str(ss_cfg.get("subject_start_offset", 1)),
            ])

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

        # --- Shared optional params ---
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

        # Diffusion maps params (only when stage2 != diffusion_maps, since
        # those are already in stage2_params otherwise)
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

        # --- Boolean flags ---
        if shared.get("ignore_cache", False) or self.ignore_cache:
            cmd.append("--ignore-cache")

        if not shared.get("verbose", True):
            cmd.append("--no-verbose")

        if shared.get("no_save_potential", False):
            cmd.append("--no-save-potential")

        # --- Channel intersection (for CD-HSA dimensionality consistency) ---
        if self._needs_channel_intersection:
            ch_key = (job["session"], job["task"])
            ch_inter = self._channel_intersections.get(ch_key)
            if ch_inter is not None:
                cmd.extend([
                    "--channel-intersection",
                    json.dumps(ch_inter),
                ])

        return cmd

    # ------------------------------------------------------------------
    # Run a single job
    # ------------------------------------------------------------------

    def _run_single_job(self, job: dict) -> tuple[str, bool]:
        key = self._checkpoint_key(
            job["super_subject_id"], job["session"], job["task"],
            job["method_label"], job["t_start"], job["t_end"],
        )

        try:
            cmd = self._build_command(job)
        except Exception as exc:
            logger.error(
                "Error construyendo comando para %s/%s/%s [%s]: %s",
                job["super_subject_label"], job["session"], job["task"],
                job["method_label"], exc,
            )
            return key, False

        logger.info(
            "RUN | %s/%s/%s | method=%s (%s) | [%s-%s] s",
            job["super_subject_label"], job["session"], job["task"],
            job["method_label"], _spec_label(job["method"]),
            job["t_start"], job["t_end"],
        )
        logger.debug("CMD: %s", " ".join(cmd))

        t0 = time.time()
        try:
            # --- FIX: desactivar buffering del subprocess para logs en tiempo real ---
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=_PROJECT_ROOT,
                env=child_env,
            )

            for line in proc.stdout:
                logger.info("  [PIPE] %s", line.rstrip())

            proc.wait()
            elapsed = time.time() - t0
            success = proc.returncode == 0

            self._write_csv_log(job, success, proc.returncode, elapsed, cmd)

            if success:
                logger.info(
                    "OK | %s/%s/%s [%s] (%.1f s)",
                    job["super_subject_label"], job["session"], job["task"],
                    job["method_label"], elapsed,
                )
            else:
                logger.error(
                    "ERROR | %s/%s/%s [%s] -- codigo %d",
                    job["super_subject_label"], job["session"], job["task"],
                    job["method_label"], proc.returncode,
                )

            return key, success

        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(
                "EXCEPTION | %s/%s/%s [%s]: %s",
                job["super_subject_label"], job["session"], job["task"],
                job["method_label"], exc,
            )
            self._write_csv_log(job, False, -1, elapsed, cmd if 'cmd' in dir() else [])
            return key, False

    # ------------------------------------------------------------------
    # Post-processing: composite matrix plots
    # ------------------------------------------------------------------

    @staticmethod
    def _embedding_key(method: dict) -> str:
        s1 = method.get("stage1_embedding")
        return s1 if s1 and s1 != "none" else "none"

    @staticmethod
    def _embedding_display_name(key: str) -> str:
        return key if key != "none" else "No embedding"

    def _organize_method_matrix(self, jobs: list[dict]) -> dict:
        """Organise jobs of a (super-subject, session, task) group into a
        matrix: columns = (stage2, stage3) combos, rows = stage1 groups."""
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
        """Generate composite matrix plots per (super-subject, session, task)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        logger.info("")
        logger.info("=" * 70)
        logger.info("  INICIANDO POST-PROCESAMIENTO: GRAFICOS COMPUESTOS (MATRIZ)")
        logger.info("=" * 70)

        # Group jobs by (super_subject_label, session, task)
        groups: dict[tuple[str, str, str], list[dict]] = {}
        for job in self.all_jobs:
            key = (job["super_subject_label"], job["session"], job["task"])
            groups.setdefault(key, []).append(job)

        total_groups = len(groups)
        generated = 0
        missing = 0
        partial = 0

        for group_idx, ((ss_label, session, task), jobs) in enumerate(groups.items(), 1):
            logger.info(
                "[%d/%d] Procesando %s/%s/%s ...",
                group_idx, total_groups, ss_label, session, task,
            )

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

                    # --- Potential row ---
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

                    # --- Non-conservative force row ---
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

                # Column titles on the first sub-row of each group
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

            # Horizontal separators between embedding groups
            if n_row_groups > 1:
                for rg_idx in range(1, n_row_groups):
                    sep_row = rg_idx * 2 - 0.5
                    fig.add_artist(plt.matplotlib.lines.Line2D(
                        [0.02, 0.98], [sep_row / n_rows, sep_row / n_rows],
                        transform=fig.transFigure,
                        color="black", linewidth=1.5, linestyle="--",
                        alpha=0.5,
                    ))

            fig.suptitle(
                f"{ss_label} | {session} | {task}  --  Methods comparison",
                fontsize=14, fontweight="bold", y=1.01,
            )

            # Save under <session>/methods_comparison
            save_dir = (
                self.output_dir
                / "test_retest_gedai_super_subject"
                / ss_label
                / session
                / "methods_comparison"
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            composite_name = (
                f"{self.composite_prefix}_{ss_label}_{session}_{task}.png"
            )
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
    # Main orchestration
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
                        job["super_subject_label"], job["session"], job["task"],
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
        description=(
            "Batch super-subject para el pipeline IgA (nueva API 3 etapas). "
            "Lee la configuracion desde un JSON."
        ),
    )
    parser.add_argument(
        "--params-json", type=str, default=None,
        help=(
            "Ruta al archivo JSON de parametros. "
            f"Default: {DEFAULT_PARAMS_JSON}"
        ),
    )
    args = parser.parse_args()

    json_path = Path(args.params_json) if args.params_json else DEFAULT_PARAMS_JSON
    if os.environ.get("BATCH_SS_PARAMS_JSON"):
        json_path = Path(os.environ["BATCH_SS_PARAMS_JSON"])

    params = _load_params(json_path)
    logger.info("Parametros cargados desde: %s", json_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner = SuperSubjectBatchRunner(params)
        return runner.run()


if __name__ == "__main__":
    sys.exit(main())

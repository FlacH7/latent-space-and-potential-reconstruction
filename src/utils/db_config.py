"""
db_config.py
============
Configuraciones centralizadas por base de datos.

Centraliza rutas, campos de metadata, labels conocidos y utilidades
para que sean reutilizables por pipelines, batch runners y postprocessing.

Uso::

    from src.utils.db_config import DB_CONFIG, resolve_result_path

    cfg = DB_CONFIG["test_retest"]
    subject_field = cfg["subject_field"]  # "subject"
    label_field = cfg["label_field"]      # "task"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CONFIGURACIÓN POR BASE DE DATOS
# ---------------------------------------------------------------------------

DB_CONFIG: dict[str, dict[str, Any]] = {
    "anphy": {
        "result_subdir": "anphy",
        "cache_subdir": "cache_eeg_anphy",
        "subject_field": "subject",
        "label_field": "stage_label",
        "known_labels": ["W", "N1", "N2", "N3", "R", "L"],
    },
    "siena": {
        "result_subdir": "siena",
        "cache_subdir": "cache_eeg_siena",
        "subject_field": "patient",
        "label_field": "period_label",
        "known_labels": ["preictal", "interictal", "postictal", "pre-ictal", "inter-ictal", "post-ictal"],
    },
    "test_retest": {
        "result_subdir": "test_retest",
        "cache_subdir": "cache_eeg_test_retest",
        "subject_field": "subject",
        "label_field": "task",
        "known_labels": ["eyesclosed", "eyesopen", "mathematic", "memory", "music"],
    },
    "test_retest_gedai": {
        "result_subdir": "test_retest_gedai",
        "cache_subdir": "cache_eeg_test_retest_gedai",
        "subject_field": "subject",
        "label_field": "task",
        "known_labels": ["eyesclosed", "eyesopen", "mathematic", "memory", "music"],
    },
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def get_db_config(db_name: str) -> dict[str, Any]:
    """Devuelve la configuración de una base de datos."""
    if db_name not in DB_CONFIG:
        raise ValueError(
            f"Base de datos '{db_name}' no reconocida. "
            f"Opciones: {', '.join(DB_CONFIG.keys())}"
        )
    return DB_CONFIG[db_name]


def resolve_result_path(db_name: str, base_results: str | Path) -> Path:
    """Resuelve la ruta de resultados para una DB."""
    cfg = get_db_config(db_name)
    return Path(base_results) / cfg["result_subdir"]


def resolve_cache_path(db_name: str, base_cache: str | Path) -> Path:
    """Resuelve la ruta de caché para una DB."""
    cfg = get_db_config(db_name)
    return Path(base_cache) / cfg["cache_subdir"]


def resolve_postprocess_path(db_name: str, base_results: str | Path) -> Path:
    """Resuelve la ruta de salida del postprocesamiento."""
    return Path(base_results) / f"postprocess_{db_name}"


def get_all_db_names() -> list[str]:
    """Devuelve la lista de nombres de DB soportadas."""
    return list(DB_CONFIG.keys())


def normalize_label(label: str, db_name: str) -> str:
    """
    Normaliza un label (ej: ``'pre-ictal'`` → ``'preictal'``).
    
    Siena tiene variantes con y sin guion; esta función unifica.
    """
    if db_name == "siena":
        return label.replace("-", "")
    return label

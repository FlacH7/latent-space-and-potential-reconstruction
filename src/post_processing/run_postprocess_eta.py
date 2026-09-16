#!/usr/bin/env python3
"""
run_postprocess_eta.py
======================
Post-procesamiento de métricas η (Helmholtz) por label con alineación
de mínimos y filtrado robusto de épocas numéricamente divergentes.

Para cada label de una DB, carga todos los ``potential_data.npz``,
calcula/recupera η, alinea los potenciales a mínimo cero, filtra
outliers numéricos vía MAD, promedia los campos sanos, y genera
una figura promedio por label.

Uso::

    python -m src.post_processing.run_postprocess_eta --db test_retest
    python -m src.post_processing.run_postprocess_eta --db anphy
    python -m src.post_processing.run_postprocess_eta --db siena

El script detecta automáticamente:
- Qué campo usar como subject/label vía ``db_config``.
- Si el residual no está en el ``.npz``, lo recalcula a partir de los
  coeficientes B-spline y el campo g.
- Interpola a una grilla común si los edges difieren entre épocas.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

# ---------------------------------------------------------------------------
# Asegurar importabilidad del paquete
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.potential_reconstruction.iga_reconstructor import compute_helmholtz_residual
from src.utils.config import BASE_RESULTS_PATH
from src.utils.db_config import (
    get_all_db_names,
    get_db_config,
    normalize_label,
    resolve_postprocess_path,
    resolve_result_path,
)
from src.utils.io import find_potential_files, load_potential

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_postprocess_eta")


# =============================================================================
# UTILIDADES DE INTERPOLACIÓN
# =============================================================================

def _interpolate_field(field, edges, target_edges):
    """
    Interpola un campo escalar o vectorial a una grilla target.
    """
    D = len(edges)
    target_shape = tuple(len(e) for e in target_edges)

    if field.shape[-D:] == target_shape:
        return np.copy(field)

    if field.ndim == D:
        if D == 1:
            f = interp1d(
                edges[0], field, kind="linear", bounds_error=False, fill_value=np.nan
            )
            return f(target_edges[0])
        pts = np.meshgrid(*target_edges, indexing="ij")
        coords = np.stack([x.ravel() for x in pts], axis=-1)
        interp = RegularGridInterpolator(
            edges, field, bounds_error=False, fill_value=np.nan
        )
        return interp(coords).reshape(target_shape)

    n_comp = field.shape[0]
    result = np.empty((n_comp,) + target_shape, dtype=float)
    if D == 1:
        for i in range(n_comp):
            f = interp1d(
                edges[0], field[i], kind="linear", bounds_error=False, fill_value=np.nan
            )
            result[i] = f(target_edges[0])
    else:
        pts = np.meshgrid(*target_edges, indexing="ij")
        coords = np.stack([x.ravel() for x in pts], axis=-1)
        for i in range(n_comp):
            interp = RegularGridInterpolator(
                edges, field[i], bounds_error=False, fill_value=np.nan
            )
            result[i] = interp(coords).reshape(target_shape)
    return result


# =============================================================================
# ALINEACIÓN Y FILTRADO ROBUSTO
# =============================================================================

def align_potential_to_min(potential, density_mask):
    """
    Resta el mínimo local (dentro de la máscara) para anclar el pozo en 0.
    """
    valid = potential[density_mask]
    if len(valid) == 0:
        return potential
    return potential - np.nanmin(valid)


def _mad_outlier_mask(values, threshold=5.0):
    """
    Máscara de outliers basada en MAD (Median Absolute Deviation).
    Más robusto que z-score frente a colas pesadas.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.ones(len(values), dtype=bool)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad < 1e-12:
        return np.ones(len(values), dtype=bool)
    # Escalar MAD a desviación estándar (~1.4826)
    lower = med - threshold * 1.4826 * mad
    upper = med + threshold * 1.4826 * mad
    return (values >= lower) & (values <= upper)


def filter_outlier_epochs(label_data, mad_threshold=5.0, eta_max=2.0,
                          min_valid_fraction=0.2):
    """
    Filtra épocas numéricamente divergentes antes de promediar.

    Criterios (todos deben pasar):
    1. η_global finito y <= eta_max.
    2. Fracción de celdas válidas >= min_valid_fraction.
    3. Rango del potencial (ptp) dentro de MAD robusto del label.
    4. Máximo de |g| dentro de MAD robusto del label.

    Returns
    -------
    list[dict]  -- épocas sanas
    dict        -- reporte de descartes por criterio
    """
    n_total = len(label_data)
    if n_total == 0:
        return [], {}

    # --- Métricas brutas por época ---
    ranges = []
    g_maxs = []
    valid_fracs = []
    etas = []

    for entry in label_data:
        pot = entry["potential"]
        mask = entry["density_mask"]
        g = entry["g_field"]
        eta = entry.get("eta")

        valid = mask & ~np.isnan(pot)
        valid_fracs.append(np.mean(valid) if valid.size else 0.0)

        if np.any(valid):
            ranges.append(np.nanmax(pot[valid]) - np.nanmin(pot[valid]))
        else:
            ranges.append(np.nan)

        g_norm = np.sqrt(np.nansum(g ** 2, axis=0))
        g_maxs.append(np.nanmax(g_norm) if np.any(~np.isnan(g_norm)) else np.nan)
        etas.append(eta if eta is not None and np.isfinite(eta) else np.nan)

    ranges = np.array(ranges)
    g_maxs = np.array(g_maxs)
    valid_fracs = np.array(valid_fracs)
    etas = np.array(etas)

    # --- Máscaras individuales ---
    mask_eta = np.isfinite(etas) & (etas <= eta_max) & (etas >= 0.0)
    mask_valid = valid_fracs >= min_valid_fraction
    mask_range = _mad_outlier_mask(ranges, threshold=mad_threshold)
    mask_gmax = _mad_outlier_mask(g_maxs, threshold=mad_threshold)

    combined = mask_eta & mask_valid & mask_range & mask_gmax

    report = {
        "n_total": int(n_total),
        "n_kept": int(np.sum(combined)),
        "n_discarded_eta": int(np.sum(~mask_eta)),
        "n_discarded_valid": int(np.sum(~mask_valid)),
        "n_discarded_range": int(np.sum(~mask_range)),
        "n_discarded_gmax": int(np.sum(~mask_gmax)),
        "median_range": float(np.nanmedian(ranges)),
        "mad_range": float(np.median(np.abs(ranges - np.nanmedian(ranges)))),
        "median_gmax": float(np.nanmedian(g_maxs)),
        "median_eta": float(np.nanmedian(etas[mask_eta])),
    }

    kept = [entry for entry, ok in zip(label_data, combined) if ok]
    return kept, report


# =============================================================================
# RECOLECCIÓN POR LABEL
# =============================================================================

def collect_eta_by_label(root_dir,
                         subject_field,
                         label_field,
                         db_name=None,
                         latent_dim=None,
                         scoring_method=None):
    """
    Recorre ``potential_data.npz`` bajo *root_dir* y agrupa por label.
    """
    files = find_potential_files(root_dir)

    # --- FILTRO POR CONFIGURACIÓN DE CARPETAS ---
    if latent_dim is not None:
        files = [f for f in files if f"{latent_dim}_latent_dim" in str(f)]
    if scoring_method is not None:
        files = [f for f in files if f"{scoring_method}" in str(f)]
    by_label = defaultdict(list)

    for f in files:
        try:
            data = load_potential(f)
            meta = data.get("metadata", {})

            label = meta.get(label_field)
            if label is None:
                for fallback in (
                    "stage_label", "period_label", "task", "label", "stage"
                ):
                    label = meta.get(fallback)
                    if label is not None:
                        break
            if label is None:
                label = "unknown"

            if db_name:
                label = normalize_label(label, db_name)

            entry = {
                "potential": data["potential"],
                "residual": data.get("residual"),
                "eta": data.get("eta"),
                "g_field": data["g_field"],
                "density_mask": data["density_mask"],
                "density": data["density"],
                "edges": data["edges"],
                "coefficients": data["coefficients"],
                "bspline_space": data.get("bspline_space"),
                "metadata": meta,
                "filepath": str(f),
            }
            by_label[label].append(entry)
        except Exception as exc:
            logger.warning("Error cargando %s: %s", f, exc)

    return by_label


# =============================================================================
# PROMEDIO POR LABEL (ALINEADO + FILTRADO)
# =============================================================================

def compute_label_averages(label_data, ref_edges=None, mad_threshold=5.0,
                           eta_max=2.0, min_valid_fraction=0.2):
    """
    Promedia campos sobre épocas sanas de un label, con alineación de mínimos.

    Returns
    -------
    dict | None
    """
    # 1. Filtrar outliers numéricos
    kept, report = filter_outlier_epochs(
        label_data,
        mad_threshold=mad_threshold,
        eta_max=eta_max,
        min_valid_fraction=min_valid_fraction,
    )
    n = len(kept)
    if n == 0:
        logger.warning("Todas las épocas fueron descartadas como outliers.")
        return None

    logger.info(
        "  Kept %d/%d epochs (discarded: η=%d, valid_frac=%d, range=%d, gmax=%d)",
        report["n_kept"], report["n_total"],
        report["n_discarded_eta"], report["n_discarded_valid"],
        report["n_discarded_range"], report["n_discarded_gmax"],
    )

    # 2. Elegir grilla de referencia (la más común entre las sanas)
    if ref_edges is None:
        shapes = [tuple(len(e) for e in d["edges"]) for d in kept]
        from collections import Counter
        shape_counts = Counter(shapes)
        most_common = shape_counts.most_common(1)[0][0]  # ya es un tuple
        for d in kept:
            if tuple(len(e) for e in d["edges"]) == most_common:
                ref_edges = d["edges"]
                break

    grid_shape = tuple(len(e) for e in ref_edges)
    D = len(ref_edges)

    pot_sum = np.zeros(grid_shape, dtype=float)
    g_norm_sum = np.zeros(grid_shape, dtype=float)
    res_norm_sum = np.zeros(grid_shape, dtype=float)
    count = np.zeros(grid_shape, dtype=int)
    res_count = np.zeros(grid_shape, dtype=int)

    etas = []

    for entry in kept:
        edges = entry["edges"]
        mask = entry["density_mask"]

        # --- Alineación de mínimos ---
        pot_raw = entry["potential"]
        pot_aligned = align_potential_to_min(pot_raw, mask)

        # Interpolar a grilla común
        pot = _interpolate_field(pot_aligned, edges, ref_edges)
        mask_interp = (
            _interpolate_field(mask.astype(float), edges, ref_edges) > 0.5
        )
        g = _interpolate_field(entry["g_field"], edges, ref_edges)
        g_norm = np.sqrt(np.nansum(g ** 2, axis=0))

        # --- Residual / η ---
        residual = entry["residual"]
        eta = entry["eta"]

        if residual is None and entry["bspline_space"] is not None:
            try:
                _, residual, eta = compute_helmholtz_residual(
                    entry["g_field"],
                    entry["coefficients"],
                    entry["bspline_space"],
                    entry["density_mask"],
                    entry["density"],
                )
            except Exception as exc:
                logger.debug(
                    "Fallo recálculo residual en %s: %s", entry["filepath"], exc
                )

        if eta is not None and np.isfinite(eta):
            etas.append(float(eta))

        # Acumular potencial y |g|
        valid = mask_interp & ~np.isnan(pot) & ~np.isnan(g_norm)
        pot_sum[valid] += pot[valid]
        g_norm_sum[valid] += g_norm[valid]
        count[valid] += 1

        # Acumular residual
        if residual is not None:
            res = _interpolate_field(residual, edges, ref_edges)
            res_norm = np.sqrt(np.nansum(res ** 2, axis=0))
            valid_res = mask_interp & ~np.isnan(res_norm)
            res_norm_sum[valid_res] += res_norm[valid_res]
            res_count[valid_res] += 1

    # --- Promedios ---
    avg = {}
    valid = count > 0

    avg["potential"] = np.full(grid_shape, np.nan)
    avg["potential"][valid] = pot_sum[valid] / count[valid]

    avg["g_norm"] = np.full(grid_shape, np.nan)
    avg["g_norm"][valid] = g_norm_sum[valid] / count[valid]

    avg["residual_norm"] = np.full(grid_shape, np.nan)
    valid_res = res_count > 0
    avg["residual_norm"][valid_res] = res_norm_sum[valid_res] / res_count[valid_res]

    avg["eta_local"] = np.full(grid_shape, np.nan)
    valid_eta = valid & valid_res & (avg["g_norm"] > 1e-12)
    avg["eta_local"][valid_eta] = (
        avg["residual_norm"][valid_eta] / avg["g_norm"][valid_eta]
    )

    avg["count"] = count
    avg["res_count"] = res_count
    avg["eta_global"] = np.array(etas) if etas else np.array([])
    avg["n_epochs"] = n
    avg["edges"] = ref_edges
    avg["D"] = D
    avg["filter_report"] = report

    return avg


# =============================================================================
# PLOTEO
# =============================================================================

def _plot_2d_label_average(label, avg_data, output_dir):
    """Figura 2×2 para D=2: potencial, |g|, ||r||, η_local."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges = avg_data["edges"]
    x, y = edges
    X, Y = np.meshgrid(x, y, indexing="ij")

    eta_vals = avg_data["eta_global"]
    n = avg_data["n_epochs"]
    rep = avg_data["filter_report"]
    eta_str = (
        f"{eta_vals.mean():.4f}±{eta_vals.std():.4f}"
        if len(eta_vals) > 0
        else "N/A"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Label: {label} | n={n}/{rep['n_total']} epochs kept | "
        f"η_global={eta_str}",
        fontsize=13,
    )

    # --- Panel 1: Potencial promedio (alineado a 0) ---
    ax = axes[0, 0]
    U = avg_data["potential"]
    if np.any(~np.isnan(U)):
        vmin, vmax = np.nanmin(U), np.nanmax(U)
        levels = np.linspace(vmin, vmax, 20)
        cnt = ax.contourf(
            X, Y, np.nan_to_num(U, nan=np.nanmean(U)), levels=levels, cmap="viridis"
        )
        plt.colorbar(cnt, ax=ax, label="U (min-aligned)")
    ax.set_title("Mean Potential (min=0)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")

    # --- Panel 2: |g| promedio ---
    ax = axes[0, 1]
    G = avg_data["g_norm"]
    if np.any(~np.isnan(G)):
        vmin, vmax = np.nanmin(G), np.nanmax(G)
        levels = np.linspace(vmin, vmax, 20)
        cnt = ax.contourf(
            X, Y, np.nan_to_num(G, nan=0), levels=levels, cmap="plasma"
        )
        plt.colorbar(cnt, ax=ax, label="|g|")
    ax.set_title("Mean |g|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")

    # --- Panel 3: ||r|| promedio ---
    ax = axes[1, 0]
    R = avg_data["residual_norm"]
    if np.any(~np.isnan(R)):
        vmin, vmax = np.nanmin(R), np.nanmax(R)
        levels = np.linspace(vmin, vmax, 20)
        cnt = ax.contourf(
            X, Y, np.nan_to_num(R, nan=0), levels=levels, cmap="magma"
        )
        plt.colorbar(cnt, ax=ax, label="||r||")
    ax.set_title("Mean ||r||")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")

    # --- Panel 4: η_local promedio ---
    ax = axes[1, 1]
    Eta = avg_data["eta_local"]
    if np.any(~np.isnan(Eta)):
        vmin, vmax = np.nanmin(Eta), np.nanmax(Eta)
        levels = np.linspace(vmin, vmax, 20)
        cnt = ax.contourf(
            X, Y, np.nan_to_num(Eta, nan=0), levels=levels, cmap="coolwarm"
        )
        plt.colorbar(cnt, ax=ax, label="η_local")
    ax.set_title("Mean Local η = ||r|| / |g|")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = output_dir / f"eta_label_average_{label}.png"
    fig.savefig(str(fname), dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", fname)


def _plot_1d_label_average(label, avg_data, output_dir):
    """Figura 2×2 para D=1."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = avg_data["edges"][0]
    eta_vals = avg_data["eta_global"]
    n = avg_data["n_epochs"]
    rep = avg_data["filter_report"]
    eta_str = (
        f"{eta_vals.mean():.4f}±{eta_vals.std():.4f}"
        if len(eta_vals) > 0
        else "N/A"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Label: {label} | n={n}/{rep['n_total']} epochs kept | "
        f"η_global={eta_str}",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(x, avg_data["potential"], "b-", label="Mean Potential")
    ax.set_title("Mean Potential (min=0)")
    ax.set_xlabel("x")
    ax.set_ylabel("U")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, avg_data["g_norm"], "g-", label="Mean |g|")
    ax.set_title("Mean |g|")
    ax.set_xlabel("x")
    ax.set_ylabel("|g|")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, avg_data["residual_norm"], "r-", label="Mean ||r||")
    ax.set_title("Mean ||r||")
    ax.set_xlabel("x")
    ax.set_ylabel("||r||")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    eta_loc = avg_data["eta_local"]
    ax.plot(x, eta_loc, "m-", label="Local η")
    ax.set_title("Mean Local η")
    ax.set_xlabel("x")
    ax.set_ylabel("||r|| / |g|")
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fname = output_dir / f"eta_label_average_{label}.png"
    fig.savefig(str(fname), dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", fname)


def plot_label_average(label, avg_data, output_dir):
    """Dispatcher según D."""
    D = avg_data["D"]
    if D == 2:
        _plot_2d_label_average(label, avg_data, output_dir)
    elif D == 1:
        _plot_1d_label_average(label, avg_data, output_dir)
    else:
        logger.warning(
            "D=%d no soportado para plots promedio; omitiendo label '%s'", D, label
        )


# =============================================================================
# ORQUESTACIÓN PRINCIPAL
# =============================================================================

def run_eta_postprocessing(db_name,
                           root_dir=None,
                           output_dir=None,
                           mad_threshold=5.0,
                           eta_max=2.0,
                           min_valid_fraction=0.2,
                           latent_dim=None,
                           scoring_method=None):
    """
    Ejecuta el post-procesamiento completo de η para una base de datos.
    """
    cfg = get_db_config(db_name)

    if root_dir is None:
        root_dir = resolve_result_path(db_name, BASE_RESULTS_PATH)
    else:
        root_dir = Path(root_dir)

    if output_dir is None:
        output_dir = resolve_postprocess_path(db_name, BASE_RESULTS_PATH) / "eta_analysis"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  ETA POST-PROCESSING: %s", db_name.upper())
    logger.info("=" * 60)
    logger.info("  Root dir : %s", root_dir.absolute())
    logger.info("  Out dir  : %s", output_dir.absolute())
    logger.info("  Subject  : %s", cfg["subject_field"])
    logger.info("  Label    : %s", cfg["label_field"])
    logger.info("  MAD thr  : %.1f", mad_threshold)
    logger.info("  η_max    : %.2f", eta_max)
    logger.info("=" * 60)

    if not root_dir.exists():
        logger.error("Root dir no existe: %s", root_dir)
        return

    # 1. Colectar por label
    by_label = collect_eta_by_label(
        root_dir,
        cfg["subject_field"],
        cfg["label_field"],
        db_name=db_name,
        latent_dim=latent_dim,
        scoring_method=scoring_method
    )

    if not by_label:
        logger.warning("No se encontraron potenciales para procesar.")
        return

    # 2. Procesar cada label
    summary = {}
    for label in sorted(by_label.keys()):
        n_eps = len(by_label[label])
        logger.info("Procesando label '%s' (%d epochs)...", label, n_eps)
        avg = compute_label_averages(
            by_label[label],
            mad_threshold=mad_threshold,
            eta_max=eta_max,
            min_valid_fraction=min_valid_fraction,
        )
        if avg is None:
            logger.warning("  Label '%s' sin épocas válidas después de filtrar.", label)
            continue

        plot_label_average(label, avg, output_dir)

        eta_vals = avg["eta_global"]
        rep = avg["filter_report"]
        summary[label] = {
            "n_epochs_total": int(rep["n_total"]),
            "n_epochs_kept": int(rep["n_kept"]),
            "n_discarded": {
                "eta": int(rep["n_discarded_eta"]),
                "valid_fraction": int(rep["n_discarded_valid"]),
                "range_outlier": int(rep["n_discarded_range"]),
                "gmax_outlier": int(rep["n_discarded_gmax"]),
            },
            "eta_global_mean": float(eta_vals.mean()) if len(eta_vals) else None,
            "eta_global_std": float(eta_vals.std()) if len(eta_vals) else None,
            "eta_global_min": float(eta_vals.min()) if len(eta_vals) else None,
            "eta_global_max": float(eta_vals.max()) if len(eta_vals) else None,
            "grid_shape": [int(s) for s in avg["potential"].shape],
            "cells_with_data": int(np.sum(avg["count"] > 0)),
            "cells_with_residual": int(np.sum(avg["res_count"] > 0)),
        }

    # 3. Guardar resumen JSON
    summary_file = output_dir / "eta_summary.json"
    with open(summary_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("Resumen JSON: %s", summary_file)
    logger.info("Output dir : %s", output_dir)
    logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-procesamiento de métricas η por label",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m src.post_processing.run_postprocess_eta --db test_retest
  python -m src.post_processing.run_postprocess_eta --db anphy
  python -m src.post_processing.run_postprocess_eta --db siena
  python -m src.post_processing.run_postprocess_eta --db test_retest \
      --root-dir ./results --out-dir ./postprocess_eta
        """,
    )
    parser.add_argument(
        "--db",
        type=str,
        required=True,
        choices=get_all_db_names(),
        help="Base de datos a procesar",
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Directorio raíz con resultados (override automático)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directorio de salida (override automático)",
    )
    parser.add_argument(
        "--mad-threshold",
        type=float,
        default=5.0,
        help="Umbral MAD para filtrar explotes numéricos (default: 5.0)",
    )
    parser.add_argument(
        "--eta-max",
        type=float,
        default=2.0,
        help="η máximo aceptable por época (default: 2.0)",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.05,
        help="Fracción mínima de celdas válidas por época (default: 0.2)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int, 
        default=None,
        help="Filtrar por latent_dim"
    )
    
    parser.add_argument(
        "--scoring-method", 
        type=str, 
        default=None,
        help="Filtrar por scoring_method"
    )
    
    args = parser.parse_args()

    run_eta_postprocessing(
        db_name=args.db,
        root_dir=args.root_dir,
        output_dir=args.out_dir,
        mad_threshold=args.mad_threshold,
        eta_max=args.eta_max,
        min_valid_fraction=args.min_valid_fraction,
        latent_dim=args.latent_dim,
        scoring_method=args.scoring_method
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

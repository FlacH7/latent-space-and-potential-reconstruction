#!/usr/bin/env python3
"""
run_postprocessing.py
=====================
Post-procesamiento genérico de potenciales IgA para cualquier base de datos.

Recorre los archivos ``potential_data.npz`` generados por los pipelines,
calcula métricas de asimetría por basin, y produce visualizaciones
agrupadas por sujeto y estado/label.

Uso::

    # Post-procesar resultados de test-retest
    python -m src.post_processing.run_postprocessing --db test_retest

    # Post-procesar resultados de ANPHY
    python -m src.post_processing.run_postprocessing --db anphy

    # Post-procesar resultados de Siena
    python -m src.post_processing.run_postprocessing --db siena

    # Especificar rutas manualmente
    python -m src.post_processing.run_postprocessing --db test_retest \\
        --root-dir ./results/test_retest --out-dir ./postprocess_test_retest

El script detecta automáticamente:
- Qué campo del metadata usar como subject (``subject`` vs ``patient``)
- Qué campo del metadata usar como label/estado (``stage_label`` vs
  ``period_label`` vs ``task``)
- Todos los labels presentes en los datos
- Genera paleta de colores automáticamente
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ---------------------------------------------------------------------------
# Asegurar importabilidad del paquete
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

for _p in (_PROJECT_ROOT, _PROJECT_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.utils.config import BASE_RESULTS_PATH
from src.utils.db_config import (
    DB_CONFIG,
    get_db_config,
    resolve_result_path,
    resolve_postprocess_path,
)
from src.utils.io import load_potential, find_potential_files
from src.post_processing.postprocess_potentials import (
    detect_basins_gradient_following,
    compute_basin_asymmetry,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_postprocessing")

# ===========================================================================
# PALETA DE COLORES AUTOMATICA
# ===========================================================================

def _generate_palette(labels: list[str]) -> dict[str, str]:
    """Genera una paleta de colores distintivos para una lista de labels."""
    if not labels:
        return {}

    tab10 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    if len(labels) <= len(tab10):
        colors = tab10[:len(labels)]
    else:
        cmap = plt.colormaps["tab20"]
        colors = [mcolors.to_hex(cmap(i / len(labels))) for i in range(len(labels))]

    return {label: color for label, color in zip(labels, colors)}


# ===========================================================================
# RECOLECCION DE DATOS
# ===========================================================================

def collect_asymmetry_data(
    root_dir: str | Path,
    subject_field: str,
    label_field: str,
    known_labels: list[str] | None = None,
    subject_filter: str | None = None,
    latent_dim: int | None = None,
    scoring_method: str | None = None,
) -> dict:
    """
    Recolecta métricas de asimetría para todos los potenciales bajo root_dir.

    Parameters
    ----------
    root_dir : str | Path
        Directorio raíz con los resultados del pipeline.
    subject_field : str
        Nombre del campo en metadata que identifica al sujeto.
    label_field : str
        Nombre del campo en metadata que identifica el estado/label.
    known_labels : list[str] | None
        Labels conocidos para validación.
    subject_filter : str | None
        Si se proporciona, solo procesa archivos de ese sujeto.
    latent_dim : int | None
    scoring_method : str | None
    Returns
    -------
    dict : {(subject, label): [lista de listas de métricas por epoch]}
    """
    root_dir = Path(root_dir)
    files = find_potential_files(root_dir)

    # --- FILTRO POR CONFIGURACIÓN DE CARPETAS ---
    if latent_dim is not None:
        files = [f for f in files if f"{latent_dim}_latent_dim" in str(f)]
    if scoring_method is not None:
        files = [f for f in files if f"{scoring_method}" in str(f)]

    if subject_filter:
        files = [f for f in files if subject_filter in str(f)]

    results: dict[tuple[str, str], list[list[dict]]] = {}
    n_ok = 0
    n_err = 0

    for f in files:
        try:
            data = load_potential(f)
            meta = data.get("metadata", {})

            # Leer subject y label con fallback
            subj = meta.get(subject_field)
            if subj is None:
                subj = meta.get("subject", meta.get("patient", "unknown"))

            label = meta.get(label_field)
            # Fallbacks si el campo principal no existe
            if label is None:
                for fallback in ("stage_label", "period_label", "task", "label", "stage"):
                    label = meta.get(fallback)
                    if label is not None:
                        break
            if label is None:
                label = "unknown"

            key = (subj, label)

            potential = data["potential"]
            density = data["density"]
            density_mask = data["density_mask"]
            edges = data["edges"]

            basin_result = detect_basins_gradient_following(
                potential, density_mask, edges
            )

            epoch_metrics = []
            for b_id in range(1, basin_result["n_basins"] + 1):
                metrics = compute_basin_asymmetry(
                    potential, density, density_mask,
                    basin_result["basin_labels"], b_id, edges
                )
                metrics["basin_id"] = b_id
                metrics["epoch_idx"] = meta.get("epoch_idx", "?")
                metrics["t_start"] = meta.get("t_start", 0)
                metrics["t_end"] = meta.get("t_end", 0)
                metrics["filepath"] = str(f)
                epoch_metrics.append(metrics)

            results.setdefault(key, []).append(epoch_metrics)
            n_ok += 1

        except Exception as e:
            logger.warning("Error procesando %s: %s", f, e)
            n_err += 1

    logger.info("Archivos procesados: %d OK, %d errores", n_ok, n_err)
    logger.info("Combinaciones (subject, label) únicas: %d", len(results))
    return results


# ===========================================================================
# VISUALIZACIONES
# ===========================================================================

def plot_subject_label_asymmetries(
    results: dict,
    subject: str,
    output_dir: Path,
    palette: dict[str, str],
) -> None:
    """Genera figuras de asimetría para un sujeto, una por label."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_data: dict[str, list[list[dict]]] = {}
    for (subj, label), epochs_list in results.items():
        if subj == subject:
            labels_data.setdefault(label, []).extend(epochs_list)

    if not labels_data:
        logger.info("No hay datos para %s", subject)
        return

    for label, epochs_list in sorted(labels_data.items()):
        if not epochs_list:
            continue

        epochs_list = sorted(
            epochs_list,
            key=lambda eps: eps[0]["t_start"] if eps and eps[0] else 0
        )
        n_epochs = len(epochs_list)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Subject: {subject} | Label: {label} | n={n_epochs} epochs", fontsize=14)

        color = palette.get(label, "#333333")

        eccs, anis, angles = [], [], []
        displacements, radii, epoch_labels = [], [], []

        for i, epoch_metrics in enumerate(epochs_list):
            if not epoch_metrics:
                continue
            m = epoch_metrics[0]
            eccs.append(m["eccentricity"])
            anis.append(m["anisotropy"])
            angles.append(m.get("asymmetry_angle", np.nan))
            displacements.append(m["displacement"])
            radii.append(m["effective_radius"])
            epoch_labels.append(f"E{i}")

        if not eccs:
            plt.close(fig)
            continue

        eccs = np.array(eccs)
        anis = np.array(anis)
        angles = np.array(angles)
        displacements = np.array(displacements)
        radii = np.array(radii)

        # Panel 1: Eccentricidad
        ax = axes[0, 0]
        ax.bar(range(len(eccs)), eccs, color=color, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(epoch_labels)))
        ax.set_xticklabels(epoch_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Eccentricity")
        ax.set_title("Eccentricity\n(displacement / radius)")
        ax.axhline(np.mean(eccs), color="red", linestyle="--", label=f"mean={np.mean(eccs):.3f}")
        ax.legend(fontsize=8)

        # Panel 2: Anisotropía
        ax = axes[0, 1]
        ax.bar(range(len(anis)), anis, color=color, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(epoch_labels)))
        ax.set_xticklabels(epoch_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Anisotropy")
        ax.set_title("Anisotropy\n(sqrt(major/minor inertia))")
        ax.axhline(np.mean(anis), color="red", linestyle="--", label=f"mean={np.mean(anis):.3f}")
        ax.legend(fontsize=8)

        # Panel 3: Scatter
        ax = axes[0, 2]
        ax.scatter(eccs, anis, c=color, s=100, alpha=0.7, edgecolors="black")
        for i, lab in enumerate(epoch_labels):
            ax.annotate(lab, (eccs[i], anis[i]), fontsize=7, ha="center", va="bottom")
        ax.set_xlabel("Eccentricity")
        ax.set_ylabel("Anisotropy")
        ax.set_title("Eccentricity vs Anisotropy")

        # Panel 4: Dirección de asimetría
        ax = axes[1, 0]
        valid_angles = angles[~np.isnan(angles)]
        if len(valid_angles) > 0:
            ax.hist(valid_angles, bins=12, range=(-180, 180), color=color, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Count")
            ax.set_title("Asymmetry Direction")
            ax.set_xlim(-180, 180)
            ax.axvline(np.mean(valid_angles), color="red", linestyle="--", label="mean")
            ax.legend(fontsize=8)
        else:
            ax.set_title("Asymmetry Direction\n(N/A for D>2)")
            ax.axis("off")

        # Panel 5: Displacement
        ax = axes[1, 1]
        ax.bar(range(len(displacements)), displacements, color=color, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(epoch_labels)))
        ax.set_xticklabels(epoch_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Displacement")
        ax.set_title("Min-to-Centroid Displacement")
        ax.axhline(np.mean(displacements), color="red", linestyle="--",
                   label=f"mean={np.mean(displacements):.3f}")
        ax.legend(fontsize=8)

        # Panel 6: Resumen
        ax = axes[1, 2]
        ax.axis("off")
        summary_text = (
            f"n_epochs: {n_epochs}\n"
            f"Eccentricity: {np.mean(eccs):.3f} ± {np.std(eccs):.3f}\n"
            f"Anisotropy:   {np.mean(anis):.3f} ± {np.std(anis):.3f}\n"
            f"Displacement: {np.mean(displacements):.3f} ± {np.std(displacements):.3f}\n"
            f"Eff. Radius:  {np.mean(radii):.3f} ± {np.std(radii):.3f}\n"
        )
        if len(valid_angles) > 0:
            try:
                from scipy.stats import circmean, circstd
                mean_angle = circmean(np.radians(valid_angles))
                std_angle = circstd(np.radians(valid_angles))
                summary_text += f"Mean Angle:   {np.degrees(mean_angle):.1f} ± {np.degrees(std_angle):.1f} deg\n"
            except ImportError:
                pass
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment="center", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = output_dir / f"asymmetry_{subject}_{label}.png"
        fig.savefig(str(fname), dpi=150)
        plt.close("all")
        plt.close(fig)
        logger.info("Saved: %s", fname)


def plot_cross_subject_scatter(
    results: dict,
    output_dir: Path,
    palette: dict[str, str],
    labels_of_interest: list[str] | None = None,
) -> None:
    """
    Para cada métrica y cada label: scatter de todas las observaciones
    por sujeto + línea de media.  Eje Y unificado por métrica (min → P95
    global); outliers se grafican en P95 pero la media usa valores reales.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if labels_of_interest is None:
        labels_of_interest = sorted(set(lbl for _, lbl in results.keys()))

    metrics = ["eccentricity", "anisotropy", "displacement", "effective_radius"]
    metric_titles = {
        "eccentricity": "Eccentricity",
        "anisotropy": "Anisotropy",
        "displacement": "Displacement",
        "effective_radius": "Effective Radius",
    }

    # --- 1. Recolectar todo ---
    all_data: dict[str, dict[str, dict[str, list[float]]]] = {
        label: {} for label in labels_of_interest
    }
    for label in labels_of_interest:
        all_data[label] = {m: {} for m in metrics}

    for (subj, lbl), epochs_list in results.items():
        if lbl not in labels_of_interest:
            continue
        for epoch_metrics in epochs_list:
            if not epoch_metrics:
                continue
            m0 = epoch_metrics[0]
            for metric in metrics:
                if metric in m0:
                    v = m0[metric]
                    if np.isfinite(v):
                        all_data[lbl][metric].setdefault(subj, []).append(v)

    # --- 2. Percentil 95 global por métrica ---
    p95 = {}
    gmin = {}
    for metric in metrics:
        vals = []
        for label in labels_of_interest:
            for subj_vals in all_data[label][metric].values():
                vals.extend(subj_vals)
        arr = np.array(vals)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            p95[metric] = 1.0
            gmin[metric] = 0.0
        else:
            p95[metric] = float(np.percentile(arr, 95))
            gmin[metric] = float(np.min(arr))

    # --- 3. Un gráfico por (métrica, label) ---
    for metric in metrics:
        for label in labels_of_interest:
            subj_data = all_data[label][metric]
            if not subj_data:
                continue

            subjects_sorted = sorted(subj_data.keys())
            color = palette.get(label, "#333333")
            y_max = p95[metric]
            y_min = gmin[metric]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(
                f"{metric_titles[metric]} | Label: {label} | n={sum(len(v) for v in subj_data.values())} epochs",
                fontsize=14,
            )

            mean_xs = []
            mean_ys = []
            for i, subj in enumerate(subjects_sorted):
                vals = np.array(subj_data[subj])
                real_mean = float(np.mean(vals))

                # Jitter en X para separar puntos
                pos = i + np.random.normal(0, 0.06, size=len(vals))

                # Clipping visual: graficar en P95, media con valor real
                vals_clipped = np.clip(vals, None, y_max)
                mean_clipped = min(real_mean, y_max)

                ax.scatter(pos, vals_clipped, alpha=0.5, color=color, s=30, zorder=3)
                # Segmento de media por sujeto
                ax.hlines(
                    mean_clipped, i - 0.3, i + 0.3,
                    colors="red", linewidths=2, zorder=4,
                )
                # Acumular para curva interpolada
                mean_xs.append(i)
                mean_ys.append(mean_clipped)
                # Anotar valor real de la media si fue recortada
                if real_mean > y_max:
                    ax.annotate(
                        f"μ={real_mean:.2f}",
                        xy=(i, y_max), fontsize=7, ha="center", va="bottom",
                        color="darkred", fontweight="bold",
                    )

            # Curva interpolada que atraviesa las medias de todos los sujetos
            ax.plot(
                mean_xs, mean_ys, color="red", linewidth=1.5,
                linestyle="-", alpha=0.4, zorder=3,
                label="Mean trend",
            )

            ax.set_xticks(range(len(subjects_sorted)))
            ax.set_xticklabels(subjects_sorted, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel(metric_titles[metric])
            ax.set_ylim(y_min * 0.95, y_max * 1.05)
            ax.grid(axis="y", alpha=0.3)
            ax.axhline(y_max, color="gray", linestyle=":", alpha=0.5,
                       label=f"P95={y_max:.3f}")
            ax.legend(fontsize=8, loc="upper left")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fname = output_dir / f"scatter_{metric}_{label}.png"
            fig.savefig(str(fname), dpi=150)
            plt.close("all")
            logger.info("Saved scatter: %s", fname)

    # --- 4. Gráfico de superposición: todos los labels en uno ---
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle(
            f"{metric_titles[metric]} | All Labels Overlay | All Subjects",
            fontsize=14,
        )

        y_max = p95[metric]
        y_min = gmin[metric]

        for label in labels_of_interest:
            subj_data = all_data[label][metric]
            if not subj_data:
                continue
            color = palette.get(label, "#333333")
            subjects_sorted = sorted(subj_data.keys())

            for i, subj in enumerate(subjects_sorted):
                vals = np.array(subj_data[subj])
                pos = i + np.random.normal(0, 0.06, size=len(vals))
                vals_clipped = np.clip(vals, None, y_max)
                ax.scatter(pos, vals_clipped, alpha=0.4, color=color, s=25, zorder=3)

        # Etiquetas de sujetos (todos los que aparecen en cualquier label)
        all_subjects = sorted(set(
            subj
            for label in labels_of_interest
            for subj in all_data[label][metric].keys()
        ))
        ax.set_xticks(range(len(all_subjects)))
        ax.set_xticklabels(all_subjects, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(metric_titles[metric])
        ax.set_ylim(y_min * 0.95, y_max * 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y_max, color="gray", linestyle=":", alpha=0.5,
                   label=f"P95={y_max:.3f}")

        # Leyenda con colores de labels
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=palette.get(lbl, "#333"),
                   markersize=8, label=lbl)
            for lbl in labels_of_interest
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = output_dir / f"scatter_{metric}_all_labels_overlay.png"
        fig.savefig(str(fname), dpi=150)
        plt.close("all")
        logger.info("Saved overlay: %s", fname)


def _make_log_bins(vals: np.ndarray, n_bins: int) -> np.ndarray | None:
    """Crea bins log-espaciados (base 10) para valores estrictamente positivos."""
    pos = vals[vals > 0]
    if len(pos) == 0:
        return None
    log_min = np.log10(pos.min())
    log_max = np.log10(pos.max())
    if not np.isfinite(log_min) or not np.isfinite(log_max):
        return None
    if log_max - log_min < 0.5:
        log_max = log_min + 0.5
    return np.logspace(log_min, log_max, num=n_bins + 1)


def _trim_log_xlim(counts: np.ndarray, bins: np.ndarray,
                     pad_ratio: float = 1.3) -> tuple[float, float]:
    """
    Recorta el eje X logarítmico para que solo abarque los bins con conteo > 0,
    eliminando el espacio vacío al inicio y al final.

    Parameters
    ----------
    counts : np.ndarray
        Conteo por bin (len == len(bins) - 1).
    bins : np.ndarray
        Bordes de los bins (log-espaciados).
    pad_ratio : float
        Padding multiplicativo alrededor del rango de datos (ej. 1.3 = 30% más).

    Returns
    -------
    (xmin, xmax) : tuple[float, float]
    """
    nonzero_idx = np.where(counts > 0)[0]
    if len(nonzero_idx) == 0:
        return float(bins[0]), float(bins[-1])

    first = nonzero_idx[0]
    last = nonzero_idx[-1]

    # Padding logarítmico: multiplicar/dividir por pad_ratio
    xmin = bins[first] / pad_ratio
    xmax = bins[min(last + 1, len(bins) - 1)] * pad_ratio
    return float(xmin), float(xmax)


def plot_label_histograms(
    results: dict,
    output_dir: Path,
    palette: dict[str, str],
    labels_of_interest: list[str] | None = None,
    n_bins: int = 20,
) -> None:
    """
    Genera histogramas agregados por label con **escala logarítmica en X**
    (base 10).  Los bins son log-espaciados **por label** (no globales) y
    el eje X se recorta automáticamente para eliminar espacio vacío.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if labels_of_interest is None:
        labels_of_interest = sorted(set(lbl for _, lbl in results.keys()))

    metrics = ["eccentricity", "anisotropy", "displacement", "effective_radius"]

    # Pasada 1: recolectar datos por label
    all_data = {label: {m: [] for m in metrics} for label in labels_of_interest}

    for (subj, lbl), epochs_list in results.items():
        if lbl not in labels_of_interest:
            continue
        for epoch_metrics in epochs_list:
            if not epoch_metrics:
                continue
            m0 = epoch_metrics[0]
            for metric in metrics:
                if metric in m0:
                    v = m0[metric]
                    if np.isfinite(v):
                        all_data[lbl][metric].append(v)

    # Pasada 2: dibujar — bins independientes por label, recorte auto de X
    for label in labels_of_interest:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"Histogramas de Frecuencia (log X) | Label: {label} | Todos los sujetos",
            fontsize=14,
        )
        color = palette.get(label, "#333333")

        for ax, metric in zip(axes.flat, metrics):
            vals = np.array(all_data[label][metric])
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                ax.set_title(f"{metric.capitalize()} (sin datos)")
                ax.axis("off")
                continue

            vals_pos = vals[vals > 0]
            if len(vals_pos) == 0:
                ax.set_title(f"{metric.capitalize()} (sin datos > 0)")
                ax.axis("off")
                continue

            # Bins log-espaciados SOLO para este label+metric
            bins_log = _make_log_bins(vals_pos, n_bins)
            if bins_log is None:
                ax.set_title(f"{metric.capitalize()} (sin datos > 0)")
                ax.axis("off")
                continue

            # Conteos y recorte de eje X
            counts, _ = np.histogram(vals_pos, bins=bins_log)
            x_min, x_max = _trim_log_xlim(counts, bins_log, pad_ratio=1.3)
            y_max = float(np.max(counts)) * 1.15 if len(counts) > 0 else 1.0

            ax.hist(vals_pos, bins=bins_log, color=color, alpha=0.7, edgecolor="black")
            ax.set_xscale("log", base=10)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0.0, y_max)
            ax.set_xlabel(f"{metric.capitalize()} (log₁₀)")
            ax.set_ylabel("Frecuencia")
            ax.set_title(f"{metric.capitalize()} (n={len(vals_pos)} epochs)")
            ax.axvline(
                np.mean(vals_pos), color="red", linestyle="--",
                label=f"μ={np.mean(vals_pos):.3f}",
            )
            ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = output_dir / f"histogram_all_subjects_{label}.png"
        fig.savefig(str(fname), dpi=150)
        plt.close("all")
        logger.info("Saved histogram: %s", fname)


# ===========================================================================
# ORQUESTACION PRINCIPAL
# ===========================================================================

def run_postprocessing(
    db_name: str,
    root_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    subject_filter: str | None = None,
    latent_dim: int | None = None,
    scoring_method: str | None = None,
) -> None:
    """
    Ejecuta el post-procesamiento completo para una base de datos.

    Parameters
    ----------
    db_name : str
        Nombre de la base de datos: ``anphy``, ``siena``, ``test_retest``.
    root_dir : str | Path | None
        Directorio con resultados del pipeline. Si None, resuelve
        automáticamente vía ``db_config``.
    output_dir : str | Path | None
        Directorio de salida. Si None, resuelve automáticamente.
    subject_filter : str | None
        Filtrar por sujeto específico.
    """
    cfg = get_db_config(db_name)

    # Resolver rutas
    if root_dir is None:
        root_dir = resolve_result_path(db_name, BASE_RESULTS_PATH)
    else:
        root_dir = Path(root_dir)

    if output_dir is None:
        output_dir = resolve_postprocess_path(db_name, BASE_RESULTS_PATH)
    else:
        output_dir = Path(output_dir)

    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  POST-PROCESSING: %s", db_name.upper())
    logger.info("=" * 60)
    logger.info("  Root dir : %s", root_dir.absolute())
    logger.info("  Out dir  : %s", output_dir.absolute())
    logger.info("  Subject  : %s", cfg["subject_field"])
    logger.info("  Label    : %s", cfg["label_field"])
    if subject_filter:
        logger.info("  Filter   : %s", subject_filter)
    logger.info("=" * 60)

    if not root_dir.exists():
        logger.error("Root dir no existe: %s", root_dir)
        return

    # 1. Recolectar métricas
    logger.info("Recolectando métricas de asimetría...")
    results = collect_asymmetry_data(
        root_dir,
        subject_field=cfg["subject_field"],
        label_field=cfg["label_field"],
        known_labels=cfg.get("known_labels"),
        subject_filter=subject_filter,
        latent_dim=latent_dim,
        scoring_method=scoring_method
    )

    if not results:
        logger.warning("No se encontraron datos de potenciales para procesar.")
        return

    # 2. Detectar labels y generar paleta
    all_labels = sorted(set(lbl for _, lbl in results.keys()))
    logger.info("Labels detectados: %s", all_labels)
    palette = _generate_palette(all_labels)

    # 3. Figuras por sujeto y label
    subjects = sorted(set(s for s, _ in results.keys()))
    logger.info("Sujetos detectados: %d", len(subjects))
    for subj in subjects:
        logger.info("Procesando sujeto: %s", subj)
        plot_subject_label_asymmetries(results, subj, output_dir / subj, palette)

    # 4. Scatter por sujeto + superposición de labels
    logger.info("Generando scatter plots cross-subject...")
    plot_cross_subject_scatter(results, output_dir, palette, all_labels)

    # 5. Histogramas
    logger.info("Generando histogramas agregados...")
    plot_label_histograms(results, output_dir, palette, all_labels)

    # 6. Resumen JSON
    summary = {}
    for (subj, label), epochs_list in results.items():
        key = f"{subj}_{label}"
        eccs = []
        anis = []
        for epoch_metrics in epochs_list:
            if epoch_metrics:
                eccs.append(epoch_metrics[0]["eccentricity"])
                anis.append(epoch_metrics[0]["anisotropy"])
        summary[key] = {
            "subject": subj,
            "label": label,
            "n_epochs": len(epochs_list),
            "eccentricity_mean": float(np.mean(eccs)) if eccs else None,
            "eccentricity_std": float(np.std(eccs)) if eccs else None,
            "anisotropy_mean": float(np.mean(anis)) if anis else None,
            "anisotropy_std": float(np.std(anis)) if anis else None,
        }

    summary_file = output_dir / "asymmetry_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Resumen JSON: %s", summary_file)
    logger.info("Output dir: %s", output_dir)
    logger.info("=" * 60)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    from src.utils.db_config import get_all_db_names

    parser = argparse.ArgumentParser(
        description="Post-procesamiento genérico de potenciales IgA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m src.post_processing.run_postprocessing --db test_retest
  python -m src.post_processing.run_postprocessing --db anphy
  python -m src.post_processing.run_postprocessing --db siena
  python -m src.post_processing.run_postprocessing --db test_retest --subject sub-01
        """,
    )
    parser.add_argument(
        "--db", type=str, required=True,
        choices=get_all_db_names(),
        help="Base de datos a procesar",
    )
    parser.add_argument(
        "--root-dir", type=str, default=None,
        help="Directorio raíz con resultados (override automático)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directorio de salida (override automático)",
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="Filtrar por sujeto específico",
    )
    
    parser.add_argument("--latent-dim", type=int, default=None,
                    help="Filtrar por latent_dim (ej: 2)")
    
    parser.add_argument("--scoring-method", type=str, default=None,
                    help="Filtrar por scoring_method (ej: hankel_dmd)")
    
    args = parser.parse_args()

    run_postprocessing(
        db_name=args.db,
        root_dir=args.root_dir,
        output_dir=args.out_dir,
        subject_filter=args.subject,
        latent_dim=args.latent_dim,
        scoring_method=args.scoring_method,
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

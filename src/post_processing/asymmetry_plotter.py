"""
asymmetry_plotter.py
====================
Genera visualizaciones de asimetria de potenciales agrupadas por
sujeto y estado de sueno.

Usado por run_batch_iga_anphy.py en la fase de post-procesamiento.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import load_potential, find_potential_files
from src.post_processing.postprocess_potentials import (
    detect_basins_gradient_following,
    compute_basin_asymmetry,
)


def collect_asymmetry_data(root_dir: str | Path, subject: str | None = None) -> dict:
    """
    Recolecta metricas de asimetria para todos los potenciales bajo root_dir.

    Returns
    -------
    dict: {(subject, stage): [lista de dicts con metricas por epoch]}
    """
    root_dir = Path(root_dir)
    files = find_potential_files(root_dir)

    if subject:
        files = [f for f in files if subject in str(f)]

    results: dict[tuple[str, str], list[list[dict]]] = {}

    for f in files:
        try:
            data = load_potential(f)
            meta = data.get("metadata", {})
            subj = meta.get("subject", "unknown")
            stage = meta.get("stage_label", "unknown")
            key = (subj, stage)

            potential = data["potential"]
            density = data["density"]
            density_mask = data["density_mask"]
            edges = data["edges"]

            # Detectar basins
            basin_result = detect_basins_gradient_following(
                potential, density_mask, edges
            )

            # Calcular metricas para cada basin
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
        except Exception as e:
            print(f"  [ASYM-PLOT] Error procesando {f}: {e}")

    return results


def plot_subject_stage_asymmetries(
    results: dict,
    subject: str,
    output_dir: Path,
):
    """
    Genera figuras de asimetria para un sujeto, una por estado de sueno.
    Cada figura muestra todos los epochs de ese estado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar por estado
    stages: dict[str, list[list[dict]]] = {}
    for (subj, stage), epochs_list in results.items():
        if subj == subject:
            stages.setdefault(stage, []).extend(epochs_list)

    if not stages:
        print(f"  [ASYM-PLOT] No hay datos para {subject}")
        return

    # Paleta de estados de sueno
    stage_colors = {
        "W": "#1f77b4",
        "N1": "#ff7f0e",
        "N2": "#2ca02c",
        "N3": "#d62728",
        "R": "#9467bd",
        "L": "#8c564b",
    }

    # Para cada estado, generar una figura
    for stage, epochs_list in sorted(stages.items()):
        if not epochs_list:
            continue
        
        epochs_list = sorted(epochs_list, key=lambda eps: eps[0]['t_start'] if eps and eps[0] else 0)
        n_epochs = len(epochs_list)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Subject: {subject} | Stage: {stage} | n={n_epochs} epochs", fontsize=14)

        color = stage_colors.get(stage, "#333333")

        # Extraer metricas de la primera base de atraccion de cada epoch
        # (se asume una unica base como indica el usuario)
        eccs = []
        anis = []
        angles = []
        displacements = []
        radii = []
        epoch_labels = []

        for i, epoch_metrics in enumerate(epochs_list):
            if not epoch_metrics:
                continue
            m = epoch_metrics[0]  # primera base
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

        # Panel 1: Eccentricidad por epoch
        ax = axes[0, 0]
        bars = ax.bar(range(len(eccs)), eccs, color=color, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(epoch_labels)))
        ax.set_xticklabels(epoch_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Eccentricity")
        ax.set_title("Eccentricity\n(displacement / radius)")
        ax.axhline(np.mean(eccs), color="red", linestyle="--", label=f"mean={np.mean(eccs):.3f}")
        ax.legend(fontsize=8)

        # Panel 2: Anisotropia por epoch
        ax = axes[0, 1]
        ax.bar(range(len(anis)), anis, color=color, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(epoch_labels)))
        ax.set_xticklabels(epoch_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Anisotropy")
        ax.set_title("Anisotropy\n(sqrt(major/minor inertia eigenvalue))")
        ax.axhline(np.mean(anis), color="red", linestyle="--", label=f"mean={np.mean(anis):.3f}")
        ax.legend(fontsize=8)

        # Panel 3: Scatter Eccentricity vs Anisotropy
        ax = axes[0, 2]
        ax.scatter(eccs, anis, c=color, s=100, alpha=0.7, edgecolors="black")
        for i, lab in enumerate(epoch_labels):
            ax.annotate(lab, (eccs[i], anis[i]), fontsize=7, ha="center", va="bottom")
        ax.set_xlabel("Eccentricity")
        ax.set_ylabel("Anisotropy")
        ax.set_title("Eccentricity vs Anisotropy")

        # Panel 4: Direccion de asimetria (histograma de angulos)
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

        # Panel 5: Displacement por epoch
        ax = axes[1, 1]
        ax.bar(range(len(displacements)), displacements, color=color, alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(epoch_labels)))
        ax.set_xticklabels(epoch_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Displacement")
        ax.set_title("Min-to-Centroid Displacement")
        ax.axhline(np.mean(displacements), color="red", linestyle="--",
                   label=f"mean={np.mean(displacements):.3f}")
        ax.legend(fontsize=8)

        # Panel 6: Resumen estadistico
        ax = axes[1, 2]
        ax.axis("off")
        summary_text = (
            f"n_epochs: {n_epochs}\n"
            f"Eccentricity: {np.mean(eccs):.3f} +/- {np.std(eccs):.3f}\n"
            f"Anisotropy:   {np.mean(anis):.3f} +/- {np.std(anis):.3f}\n"
            f"Displacement: {np.mean(displacements):.3f} +/- {np.std(displacements):.3f}\n"
            f"Eff. Radius:  {np.mean(radii):.3f} +/- {np.std(radii):.3f}\n"
        )
        if len(valid_angles) > 0:
            try:
                from scipy.stats import circmean, circstd
                mean_angle = circmean(np.radians(valid_angles))
                std_angle = circstd(np.radians(valid_angles))
                summary_text += f"Mean Angle:   {np.degrees(mean_angle):.1f} +/- {np.degrees(std_angle):.1f} deg\n"
            except ImportError:
                pass
        ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment="center", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = output_dir / f"asymmetry_{subject}_{stage}.png"
        fname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(fname), dpi=150)
        plt.close("all")
        plt.close(fig)
        print(f"  [ASYM-PLOT] Saved: {fname}")


def plot_cross_subject_comparison(
    results: dict,
    output_dir: Path,
    stages_of_interest: list[str] | None = None,
):
    """
    Genera figura comparativa entre sujetos para cada estado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stages_of_interest is None:
        stages_of_interest = ["N2", "N3", "R", "W"]

    stage_colors = {
        "W": "#1f77b4", "N1": "#ff7f0e", "N2": "#2ca02c",
        "N3": "#d62728", "R": "#9467bd", "L": "#8c564b",
    }

    for stage in stages_of_interest:
        # Recolectar datos por sujeto para este estado
        subject_data: dict[str, dict[str, list]] = {}

        for (subj, stg), epochs_list in results.items():
            if stg != stage:
                continue
            eccs, anis = [], []
            for epoch_metrics in epochs_list:
                if epoch_metrics:
                    eccs.append(epoch_metrics[0]["eccentricity"])
                    anis.append(epoch_metrics[0]["anisotropy"])
            if eccs:
                subject_data[subj] = {"ecc": eccs, "anis": anis}

        if not subject_data:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Cross-Subject Comparison | Stage: {stage}", fontsize=14)

        subjects_sorted = sorted(subject_data.keys())
        color = stage_colors.get(stage, "#333333")

        # Eccentricidad
        ax = axes[0]
        positions = []
        all_vals = []
        for i, subj in enumerate(subjects_sorted):
            vals = subject_data[subj]["ecc"]
            pos = i + np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(pos, vals, alpha=0.6, color=color, s=50, zorder=3)
            ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.3),
                       medianprops=dict(color="red", linewidth=2))
            positions.append(i)
        ax.set_xticks(positions)
        ax.set_xticklabels(subjects_sorted, rotation=45, ha="right")
        ax.set_ylabel("Eccentricity")
        ax.set_title("Eccentricity by Subject")
        ax.grid(axis="y", alpha=0.3)

        # Anisotropia
        ax = axes[1]
        positions = []
        for i, subj in enumerate(subjects_sorted):
            vals = subject_data[subj]["anis"]
            pos = i + np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(pos, vals, alpha=0.6, color=color, s=50, zorder=3)
            ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.3),
                       medianprops=dict(color="red", linewidth=2))
            positions.append(i)
        ax.set_xticks(positions)
        ax.set_xticklabels(subjects_sorted, rotation=45, ha="right")
        ax.set_ylabel("Anisotropy")
        ax.set_title("Anisotropy by Subject")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = output_dir / f"cross_subject_{stage}.png"
        fname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(fname), dpi=150)
        plt.close("all")
        plt.close(fig)
        print(f"  [ASYM-PLOT] Saved cross-subject: {fname}")

def plot_stage_histograms(
    results: dict,
    output_dir: Path,
    stages_of_interest: list[str] | None = None,
    n_bins: int = 20,
    log_metrics: list[str] | None = None,
    auto_log_threshold: float = 50.0,
    log_base: float = 10.0,
):
    """
    Genera histogramas agregados por estado con rangos, límites y bins
    UNIFICADOS para cada métrica entre todas las figuras (estados).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stages_of_interest is None:
        stages_of_interest = ["N2", "N3", "R", "W"]

    stage_colors = {
        "W": "#1f77b4", "N1": "#ff7f0e", "N2": "#2ca02c",
        "N3": "#d62728", "R": "#9467bd", "L": "#8c564b",
    }

    metrics = ["eccentricity", "anisotropy", "displacement", "effective_radius"]

    # ============================================================
    # PASADA 1: Recolectar TODOS los datos y calcular rangos globales
    # ============================================================
    all_data = {stage: {m: [] for m in metrics} for stage in stages_of_interest}

    for (subj, stg), epochs_list in results.items():
        if stg not in stages_of_interest:
            continue
        for epoch_metrics in epochs_list:
            if not epoch_metrics:
                continue
            m0 = epoch_metrics[0]
            for metric in metrics:
                if metric in m0:
                    v = m0[metric]
                    if np.isfinite(v):
                        all_data[stg][metric].append(v)

    # Determinar qué métricas usan escala log
    log_flags = {}
    for metric in metrics:
        all_vals = []
        for stage in stages_of_interest:
            all_vals.extend(all_data[stage][metric])
        arr = np.array(all_vals)
        if len(arr) == 0 or np.min(arr) <= 0:
            log_flags[metric] = False
            continue
        if log_metrics is not None:
            log_flags[metric] = metric in log_metrics
        else:
            log_flags[metric] = np.max(arr) / np.min(arr) > auto_log_threshold

    # Calcular bins y límites globales por métrica
    global_bins = {}
    global_xlims = {}
    global_ylims = {}

    for metric in metrics:
        all_vals = []
        for stage in stages_of_interest:
            all_vals.extend(all_data[stage][metric])
        arr = np.array(all_vals)
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0:
            global_bins[metric] = None
            global_xlims[metric] = (0.0, 1.0)
            global_ylims[metric] = (0.0, 1.0)
            continue

        if log_flags[metric]:
            log_min = np.log(np.min(arr)) / np.log(log_base)
            log_max = np.log(np.max(arr)) / np.log(log_base)

            if not np.isfinite(log_min) or not np.isfinite(log_max):
                log_flags[metric] = False
            elif log_max - log_min < 0.01:
                log_max = log_min + 0.5
            else:
                edges = np.logspace(
                    log_min, log_max, num=n_bins + 1, base=log_base
                )
                global_bins[metric] = edges
                x_left = log_base ** (log_min - 0.05 * (log_max - log_min))
                x_right = log_base ** (log_max + 0.05 * (log_max - log_min))
                global_xlims[metric] = (x_left, x_right)

        if not log_flags[metric]:
            x_min, x_max = float(np.min(arr)), float(np.max(arr))
            if x_min == x_max:
                x_min -= 0.5
                x_max += 0.5
            padding = (x_max - x_min) * 0.05
            edges = np.linspace(x_min - padding, x_max + padding, num=n_bins + 1)
            global_bins[metric] = edges
            global_xlims[metric] = (float(x_min - padding), float(x_max + padding))

        counts, _ = np.histogram(arr, bins=global_bins[metric])
        y_max = float(np.max(counts)) if len(counts) > 0 else 1.0
        global_ylims[metric] = (0.0, y_max * 1.15)

        x_left, x_right = global_xlims[metric]
        y_bottom, y_top = global_ylims[metric]
        if not all(np.isfinite([x_left, x_right, y_bottom, y_top])):
            global_xlims[metric] = (0.0, 1.0)
            global_ylims[metric] = (0.0, 1.0)

    # ============================================================
    # PASADA 2: Dibujar las figuras con los parámetros unificados
    # ============================================================
    for stage in stages_of_interest:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(
            f"Histogramas de Frecuencia | Stage: {stage} | Todos los sujetos",
            fontsize=14,
        )

        color = stage_colors.get(stage, "#333333")

        for ax, metric in zip(axes.flat, metrics):
            vals = np.array(all_data[stage][metric])
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0 or global_bins[metric] is None:
                ax.set_title(f"{metric.capitalize()} (sin datos)")
                ax.axis("off")
                continue

            ax.hist(
                vals,
                bins=global_bins[metric],
                color=color,
                alpha=0.7,
                edgecolor="black",
            )

            if log_flags[metric]:
                ax.set_xscale("log", base=log_base)
                ax.set_xlabel(f"{metric.capitalize()} (log base {log_base})")

                # --- INICIO: Sistema de ticks mejorado ---
                from matplotlib.ticker import LogLocator, LogFormatter

                # Major ticks: potencias de la base (1, 10, 100...)
                ax.xaxis.set_major_locator(
                    LogLocator(base=log_base, numticks=15)
                )
                ax.xaxis.set_major_formatter(
                    LogFormatter(base=log_base, labelOnlyBase=False)
                )

                # Minor ticks: subdivisiones entre potencias
                if log_base == 10:
                    subs = np.arange(2, 10)
                elif log_base == 2:
                    subs = []
                elif log_base > 2 and abs(log_base - round(log_base)) < 0.01:
                    n_subs = min(int(round(log_base)) - 2, 8)
                    if n_subs > 0:
                        subs = np.linspace(2, int(round(log_base)) - 1, n_subs)
                        subs = np.unique(subs.astype(int))
                    else:
                        subs = []
                else:
                    subs = 'auto'

                if len(subs) > 0 or subs == 'auto':
                    ax.xaxis.set_minor_locator(
                        LogLocator(base=log_base, subs=subs, numticks=15)
                    )
                    # Mostrar etiquetas de minor ticks cuando hay pocos major ticks
                    ax.xaxis.set_minor_formatter(
                        LogFormatter(
                            base=log_base,
                            labelOnlyBase=False,
                            minor_thresholds=(4, 0.4)
                        )
                    )
                # --- FIN: Sistema de ticks mejorado ---

            else:
                ax.set_xlabel(metric.capitalize())

            ax.set_ylabel("Frecuencia")
            ax.set_title(f"{metric.capitalize()} (n={len(vals)} epochs)")
            ax.set_xlim(global_xlims[metric])
            # ax.set_ylim(global_ylims[metric])

            ax.axvline(
                np.mean(vals),
                color="red",
                linestyle="--",
                label=f"μ={np.mean(vals):.3f}",
            )
            ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fname = output_dir / f"histogram_all_subjects_{stage}.png"
        fig.savefig(str(fname), dpi=150)
        plt.close("all")
        print(f"  [ASYM-PLOT] Saved histogram: {fname}")
        
        
def run_full_postprocessing(root_dir: str | Path, output_dir: str | Path, subject: str | None = None):
    """Ejecuta el post-procesamiento completo."""
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  POST-PROCESSING: Asymmetry Analysis")
    print("=" * 60)

    # 1. Recolectar metricas
    print("\n  Collecting asymmetry metrics...")
    results = collect_asymmetry_data(root_dir, subject=subject)

    if not results:
        print("  [WARN] No potential data found for post-processing.")
        return

    # 2. Generar figuras por sujeto y estado
    subjects = sorted(set(s for s, _ in results.keys()))
    for subj in subjects:
        print(f"\n  Processing subject: {subj}")
        plot_subject_stage_asymmetries(results, subj, output_dir / subj)

    # 3. Comparacion cross-subject
    print("\n  Generating cross-subject comparisons...")
    plot_cross_subject_comparison(results, output_dir)
    
    # 4. Histogramas agregados por estado (todos los sujetos)
    print("\n  Generando histogramas agregados por estado...")
    plot_stage_histograms(results, output_dir, log_metrics=["anisotropy"])

    # 4. Guardar resumen JSON
    summary = {}
    for (subj, stage), epochs_list in results.items():
        key = f"{subj}_{stage}"
        eccs = []
        anis = []
        for epoch_metrics in epochs_list:
            if epoch_metrics:
                eccs.append(epoch_metrics[0]["eccentricity"])
                anis.append(epoch_metrics[0]["anisotropy"])
        summary[key] = {
            "subject": subj,
            "stage": stage,
            "n_epochs": len(epochs_list),
            "eccentricity_mean": float(np.mean(eccs)) if eccs else None,
            "eccentricity_std": float(np.std(eccs)) if eccs else None,
            "anisotropy_mean": float(np.mean(anis)) if anis else None,
            "anisotropy_std": float(np.std(anis)) if anis else None,
        }

    summary_file = output_dir / "asymmetry_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_file}")
    print(f"  Output dir: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Asymmetry plotter for IgA potentials")
    parser.add_argument("--root-dir", type=str, required=True, help="Directorio raiz con resultados")
    parser.add_argument("--output-dir", type=str, default="./asymmetry_plots", help="Directorio de salida")
    parser.add_argument("--subject", type=str, default=None, help="Filtrar por sujeto")
    args = parser.parse_args()

    run_full_postprocessing(args.root_dir, args.output_dir, args.subject)

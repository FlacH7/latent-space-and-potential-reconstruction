"""
postprocess_potentials.py
=========================
Script de post-procesamiento para analizar potenciales reconstruidos.

Funcionalidades:
  1. Detectar bases de atraccion (basins) mediante watershed sobre el potencial.
  2. Cuantificar la asimetria de cada basin via:
     - Excentricidad: desplazamiento minimo-centroide normalizado
     - Anisotropia: ratio de autovalores de momentos de inercia
     - Direccion de asimetria: angulo del vector minimo->centroide
  3. Generar reportes y visualizaciones por estado de sueno.

Uso:
    # Analizar un solo fichero
    python postprocess_potentials.py --file path/to/potential_data.npz

    # Analizar todo un arbol de resultados (por sujeto y estado)
    python postprocess_potentials.py --root-dir ./batch_results_anphy --output-dir ./postprocess_results

    # Solo por sujeto especifico
    python postprocess_potentials.py --root-dir ./batch_results_anphy --subject EPCTL01
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from src.utils.io import (
    load_potential,
    find_potential_files,
    group_potentials_by_subject_and_stage,
)


# =============================================================================
# Deteccion de basins (bases de atraccion)
# =============================================================================

def detect_basins(potential: np.ndarray, density_mask: np.ndarray) -> dict:
    """
    Detecta bases de atraccion mediante watershed invertido sobre el potencial.

    Parameters
    ----------
    potential : np.ndarray shape (n1, n2) o (n1, n2, n3)
        Potencial reconstruido. Se asume que U esta anclado a 0 en el minimo.
    density_mask : np.ndarray bool
        Mascara de densidad valida.

    Returns
    -------
    result : dict
        {
            'n_basins': int,
            'basin_labels': np.ndarray (misma shape que potential),
            'minima_coords': list[tuple],  # coordenadas de minimos
            'minima_values': list[float],
            'minima_grid_positions': list[tuple[float, ...]],  # posiciones en coords reales
        }
    """
    # Asegurar que solo trabajamos dentro de la mascara
    U = np.where(density_mask, potential, np.inf)

    # Encontrar minimos locales (8-vecindad en 2D, 26-vecindad en 3D)
    D = potential.ndim
    if D == 2:
        footprint = np.ones((3, 3))
    elif D == 3:
        footprint = np.ones((3, 3, 3))
    else:
        raise NotImplementedError(f"detect_basins solo soporta D=2 o D=3, got D={D}")

    # Minimos locales: U[x] < U[vecinos]
    local_min = ndimage.minimum_filter(U, footprint=footprint, mode="constant", cval=np.inf)
    minima_mask = (U == local_min) & density_mask

    # Eliminar minimos degenerados (con valores inf)
    minima_mask &= np.isfinite(U)

    # Etiquetar minimos conectados
    labeled_minima, n_minima = ndimage.label(minima_mask)

    if n_minima == 0:
        return {
            'n_basins': 0,
            'basin_labels': np.zeros_like(potential, dtype=int),
            'minima_coords': [],
            'minima_values': [],
            'minima_grid_positions': [],
        }

    # Comprimir minimos conectados: tomar el punto de menor U en cada componente
    minima_coords = []
    minima_values = []
    for i in range(1, n_minima + 1):
        coords_i = np.argwhere(labeled_minima == i)
        vals_i = potential[tuple(coords_i.T)]
        best_idx = np.argmin(vals_i)
        minima_coords.append(tuple(coords_i[best_idx]))
        minima_values.append(float(vals_i[best_idx]))

    # Watershed desde los minimos
    # Invertimos el potencial para que los minimos sean "picos"
    U_inv = np.where(density_mask, -potential, -np.inf)
    # Normalizar para evitar problemas numericos
    U_finite = U_inv[np.isfinite(U_inv)]
    if len(U_finite) > 0:
        U_inv = np.where(density_mask, U_inv - U_finite.min(), U_inv)

    # Crear marcadores para watershed
    markers = np.zeros_like(potential, dtype=int)
    for i, coord in enumerate(minima_coords, start=1):
        markers[coord] = i

    # Aplicar watershed sobre el potencial invertido
    try:
        from scipy.ndimage import watershed_ift
        U_scaled = np.clip(U_inv * 1e6, 0, 2**31 - 1).astype(np.int32)
        basin_labels = watershed_ift(U_scaled, markers)
    except ImportError:
        # Fallback: usar metodo de gradiente si watershed_ift no esta disponible
        basin_labels = _watershed_gradient_fallback(U_inv, markers, density_mask, D)

    # Solo conservar regiones dentro de la mascara de densidad
    basin_labels = np.where(density_mask, basin_labels, 0)

    # Convertir coordenadas de grid a posiciones reales
    minima_grid_positions = list(minima_coords)

    return {
        'n_basins': len(minima_coords),
        'basin_labels': basin_labels,
        'minima_coords': minima_coords,
        'minima_values': minima_values,
        'minima_grid_positions': minima_grid_positions,
    }


def _watershed_gradient_fallback(U_inv, markers, density_mask, D):
    """Fallback para watershed usando seguimiento de gradiente."""
    basin_labels = np.zeros_like(markers)
    coords_valid = np.argwhere(density_mask & (markers == 0))

    # Precomputar gradiente
    grad = np.gradient(np.where(np.isfinite(U_inv), U_inv, np.nan))

    for pt in coords_valid:
        pt = tuple(pt)
        current = pt
        visited = set()
        while current not in visited:
            visited.add(current)
            if markers[current] > 0:
                basin_labels[pt] = markers[current]
                break
            # Mover en direccion del gradiente (hacia arriba en U_inv = hacia abajo en U)
            best_next = None
            best_dot = -np.inf
            for offset in np.ndindex(*([3] * D)):
                offset = tuple(o - 1 for o in offset)
                if all(o == 0 for o in offset):
                    continue
                neighbor = tuple(c + o for c, o in zip(current, offset))
                if any(n < 0 or n >= s for n, s in zip(neighbor, basin_labels.shape)):
                    continue
                if not density_mask[neighbor]:
                    continue
                direction = np.array(offset, dtype=float)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                grad_at = np.array([g[current] for g in grad])
                dot = np.dot(direction, grad_at)
                if dot > best_dot:
                    best_dot = dot
                    best_next = neighbor
            if best_next is None or best_next == current:
                break
            current = best_next

    # Tambien asignar los puntos con marcadores directamente
    basin_labels[markers > 0] = markers[markers > 0]
    return basin_labels


def detect_basins_gradient_following(
    potential: np.ndarray,
    density_mask: np.ndarray,
    edges: list[np.ndarray],
) -> dict:
    """
    Detecta basins siguiendo el gradiente descendiente desde cada punto.
    Metodo mas robusto que watershed para potenciales suaves.

    Returns
    -------
    dict con mismo formato que detect_basins.
    """
    D = potential.ndim
    U = np.where(density_mask, potential, np.inf)

    # Encontrar minimos locales
    if D == 2:
        footprint = np.ones((3, 3))
    elif D == 3:
        footprint = np.ones((3, 3, 3))
    else:
        raise NotImplementedError(f"Solo D=2 o D=3, got D={D}")

    local_min = ndimage.minimum_filter(U, footprint=footprint, mode="constant", cval=np.inf)
    minima_mask = (U == local_min) & density_mask & np.isfinite(U)
    labeled_minima, n_minima = ndimage.label(minima_mask)

    if n_minima == 0:
        return {
            'n_basins': 0,
            'basin_labels': np.zeros_like(potential, dtype=int),
            'minima_coords': [],
            'minima_values': [],
            'minima_grid_positions': [],
        }

    # Comprimir minimos conectados
    minima_coords = []
    minima_values = []
    for i in range(1, n_minima + 1):
        coords_i = np.argwhere(labeled_minima == i)
        vals_i = potential[tuple(coords_i.T)]
        best_idx = np.argmin(vals_i)
        minima_coords.append(tuple(coords_i[best_idx]))
        minima_values.append(float(vals_i[best_idx]))

    # Asignar cada punto valido a su minimo siguiendo -gradiente
    basin_labels = np.zeros_like(potential, dtype=int)
    coords_valid = np.argwhere(density_mask)

    # Precomputar gradiente
    grad = np.gradient(np.where(np.isfinite(U), U, np.nan))

    for pt in coords_valid:
        pt = tuple(pt)
        current = pt
        visited = set()
        while current not in visited:
            visited.add(current)
            # Si llegamos a un minimo etiquetado, asignar
            if labeled_minima[current] > 0:
                basin_labels[pt] = labeled_minima[current]
                break
            # Mover en direccion opuesta al gradiente (hacia abajo)
            best_next = None
            best_dot = -np.inf
            # Revisar vecinos
            for offset in np.ndindex(*([3] * D)):
                offset = tuple(o - 1 for o in offset)
                if all(o == 0 for o in offset):
                    continue
                neighbor = tuple(c + o for c, o in zip(current, offset))
                # Check bounds
                if any(n < 0 or n >= s for n, s in zip(neighbor, potential.shape)):
                    continue
                if not density_mask[neighbor]:
                    continue
                # Direccion de descenso mas pronunciado
                direction = np.array(offset, dtype=float)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                grad_at = np.array([g[current] for g in grad])
                # Queremos ir contra el gradiente (hacia abajo)
                dot = -np.dot(direction, grad_at)
                if dot > best_dot:
                    best_dot = dot
                    best_next = neighbor
            if best_next is None or best_next == current:
                # No hay a donde ir
                break
            current = best_next

    # Re-mapear etiquetas a 1..n_minima
    basin_labels_remapped = np.zeros_like(basin_labels)
    for new_id, old_coord in enumerate(minima_coords, start=1):
        old_label = labeled_minima[old_coord]
        basin_labels_remapped[basin_labels == old_label] = new_id

    return {
        'n_basins': len(minima_coords),
        'basin_labels': basin_labels_remapped,
        'minima_coords': minima_coords,
        'minima_values': minima_values,
        'minima_grid_positions': list(minima_coords),
    }


# =============================================================================
# Metricas de asimetria
# =============================================================================

def compute_basin_asymmetry(
    potential: np.ndarray,
    density: np.ndarray,
    density_mask: np.ndarray,
    basin_labels: np.ndarray,
    basin_id: int,
    edges: list[np.ndarray],
) -> dict[str, Any]:
    """
    Calcula metricas de asimetria para una base de atraccion.

    Parameters
    ----------
    potential, density, density_mask : np.ndarray
        Datos del potencial.
    basin_labels : np.ndarray int
        Etiqueta de basin para cada punto.
    basin_id : int
        ID del basin a analizar.
    edges : list[np.ndarray]
        Coordenadas de los ejes.

    Returns
    -------
    metrics : dict
        {
            'centroid': tuple[float, ...],      # centro de masa de la cuenca
            'minimum_pos': tuple[float, ...],   # posicion del minimo
            'displacement': float,               # distancia minimo->centroide
            'effective_radius': float,           # radio efectivo de la cuenca
            'eccentricity': float,               # excentricidad (0=circular, ->1=asimetrico)
            'anisotropy': float,                 # ratio mayor/menor momento de inercia
            'asymmetry_angle': float,            # direccion de asimetria en grados
            'covariance_matrix': np.ndarray,     # matriz de covarianza espacial
            'eigenvalues': tuple[float, float],  # autovalores ordenados
            'eigenvectors': np.ndarray,          # autovectores
            'area_pixels': int,                  # tamano de la cuenca en pixeles
        }
    """
    D = potential.ndim

    # Mascara de este basin
    basin_mask = (basin_labels == basin_id) & density_mask
    if not np.any(basin_mask):
        return {
            'centroid': tuple([np.nan] * D),
            'minimum_pos': tuple([np.nan] * D),
            'displacement': np.nan,
            'effective_radius': np.nan,
            'eccentricity': np.nan,
            'anisotropy': np.nan,
            'asymmetry_angle': np.nan,
            'covariance_matrix': np.full((D, D), np.nan),
            'eigenvalues': tuple([np.nan] * D),
            'eigenvectors': np.full((D, D), np.nan),
            'area_pixels': 0,
        }

    # Construir meshgrid de coordenadas reales
    grid_coords = np.meshgrid(*edges, indexing="ij")
    coords_flat = np.stack([g[basin_mask] for g in grid_coords], axis=-1)

    # Centro de masa (ponderado por densidad)
    weights = density[basin_mask]
    weights = np.maximum(weights, 0)  # asegurar no negativo
    wsum = weights.sum()
    if wsum > 0:
        centroid = (coords_flat.T @ weights) / wsum
    else:
        centroid = coords_flat.mean(axis=0)

    # Encontrar el minimo del potencial dentro del basin
    U_in_basin = np.where(basin_mask, potential, np.inf)
    min_idx = np.unravel_index(np.argmin(U_in_basin), potential.shape)
    minimum_pos = np.array([edges[d][min_idx[d]] for d in range(D)])

    # Vector de desplazamiento
    displacement_vec = centroid - minimum_pos
    displacement = float(np.linalg.norm(displacement_vec))

    # Radio efectivo: sqrt(area/pi) en 2D, (3V/4pi)^(1/3) en 3D
    area_pixels = int(np.sum(basin_mask))
    if D == 2:
        # Aproximar area en coords reales
        dx = np.mean([np.diff(e).mean() for e in edges])
        area_real = area_pixels * dx**2
        effective_radius = np.sqrt(area_real / np.pi)
    else:
        dx = np.mean([np.diff(e).mean() for e in edges])
        vol_real = area_pixels * dx**D
        effective_radius = ((3 * vol_real) / (4 * np.pi)) ** (1 / 3)

    # Excentricidad: desplazamiento normalizado por radio
    eccentricity = displacement / (effective_radius + 1e-10)

    # Momentos de inercia (covarianza espacial ponderada)
    if wsum > 0:
        centered = coords_flat - centroid
        cov = (centered.T @ (centered * weights[:, None])) / wsum
    else:
        centered = coords_flat - centroid
        cov = np.cov(centered.T)

    # Autovalores y autovectores
    if D == 2:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Ordenar de mayor a menor
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        min_ev = max(eigenvalues[-1], 1e-6 * eigenvalues[0])
        anisotropy = float(np.clip(np.sqrt(eigenvalues[0] / min_ev), 1.0, 100.0))
        # Direccion de asimetria
        asymmetry_angle = float(np.degrees(np.arctan2(displacement_vec[1], displacement_vec[0])))
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        min_ev = max(eigenvalues[-1], 1e-6 * eigenvalues[0])
        anisotropy = float(np.clip(np.sqrt(eigenvalues[0] / min_ev), 1.0, 100.0))
        asymmetry_angle = np.nan

    return {
        'centroid': tuple(float(c) for c in centroid),
        'minimum_pos': tuple(float(m) for m in minimum_pos),
        'displacement': displacement,
        'effective_radius': float(effective_radius),
        'eccentricity': eccentricity,
        'anisotropy': anisotropy,
        'asymmetry_angle': asymmetry_angle,
        'covariance_matrix': cov,
        'eigenvalues': tuple(float(e) for e in eigenvalues),
        'eigenvectors': eigenvectors,
        'area_pixels': area_pixels,
    }


# =============================================================================
# Analisis completo de un potencial
# =============================================================================

def analyze_potential(filepath: str | Path, plot: bool = True, out_dir: str | Path | None = None) -> dict:
    """
    Analisis completo: detecta basins y calcula asimetrias.

    Returns
    -------
    dict con resultados del analisis.
    """
    filepath = Path(filepath)
    data = load_potential(filepath)

    potential = data["potential"]
    density = data["density"]
    density_mask = data["density_mask"]
    edges = data["edges"]
    metadata = data.get("metadata", {})
    config = data.get("config", {})

    D = potential.ndim

    # Detectar basins
    basin_result = detect_basins_gradient_following(potential, density_mask, edges)
    n_basins = basin_result['n_basins']
    basin_labels = basin_result['basin_labels']

    # Calcular asimetria para cada basin
    basin_metrics = []
    for b_id in range(1, n_basins + 1):
        metrics = compute_basin_asymmetry(
            potential, density, density_mask, basin_labels, b_id, edges
        )
        metrics['basin_id'] = b_id
        basin_metrics.append(metrics)

    # Resultado global
    result = {
        'filepath': str(filepath),
        'metadata': metadata,
        'config': config,
        'n_basins': n_basins,
        'potential_shape': potential.shape,
        'basin_metrics': basin_metrics,
    }

    # Visualizacion
    if plot and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_asymmetry_analysis(result, potential, density_mask, basin_labels, edges, out_dir)

    return result


def _plot_asymmetry_analysis(
    result: dict,
    potential: np.ndarray,
    density_mask: np.ndarray,
    basin_labels: np.ndarray,
    edges: list[np.ndarray],
    out_dir: Path,
):
    """Genera figuras de diagnostico del analisis de asimetria."""
    D = potential.ndim
    if D != 2:
        return  # Solo 2D por ahora

    meta = result['metadata']
    subject = meta.get('subject', 'unknown')
    stage = meta.get('stage_label', 'unknown')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Potencial con basins coloreados
    ax = axes[0]
    X, Y = np.meshgrid(edges[0], edges[1], indexing="ij")
    U_plot = np.where(density_mask, potential, np.nan)
    im = ax.contourf(X, Y, U_plot, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, label="U")
    # Contornos de basins
    for b in result['basin_metrics']:
        cx, cy = b['centroid']
        mx, my = b['minimum_pos']
        ax.plot(cx, cy, "r*", markersize=15, label="Centroid")
        ax.plot(mx, my, "w+", markersize=15, mew=2, label="Minimum")
        # Vector de asimetria
        ax.annotate("", xy=(cx, cy), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        # Elipse de covarianza
        _plot_covariance_ellipse(ax, b['centroid'], b['covariance_matrix'],
                                  b['eigenvectors'], b['eigenvalues'],
                                  n_std=2, edgecolor="yellow", facecolor="none", lw=2)
    ax.set_title(f"Potential + Basins\n{subject} | {stage}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Panel 2: Labels de basins
    ax = axes[1]
    labels_plot = np.where(density_mask, basin_labels, np.nan)
    im2 = ax.imshow(labels_plot.T, origin="lower", extent=[edges[0][0], edges[0][-1],
                                                            edges[1][0], edges[1][-1]],
                    cmap="tab10", alpha=0.8, interpolation="nearest")
    plt.colorbar(im2, ax=ax, label="Basin ID")
    ax.set_title(f"Detected Basins: {result['n_basins']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Panel 3: Metricas de asimetria como barras
    ax = axes[2]
    metrics = result['basin_metrics']
    if metrics:
        ids = [f"B{m['basin_id']}" for m in metrics]
        eccs = [m['eccentricity'] for m in metrics]
        anis = [m['anisotropy'] for m in metrics]
        x = np.arange(len(ids))
        w = 0.35
        ax.bar(x - w/2, eccs, w, label="Eccentricity", color="steelblue")
        ax.bar(x + w/2, anis, w, label="Anisotropy", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(ids)
        ax.set_ylabel("Value")
        ax.set_title("Asymmetry Metrics")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5)
        # Anotar angulos
        for i, m in enumerate(metrics):
            angle = m.get('asymmetry_angle', np.nan)
            if not np.isnan(angle):
                ax.annotate(f"{angle:.1f} deg", xy=(i, max(eccs[i], anis[i])),
                           ha="center", va="bottom", fontsize=8, color="darkred")

    plt.tight_layout()
    fname = out_dir / f"asymmetry_{subject}_{stage}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [POST] Asymmetry plot saved to: {fname}")


def _plot_covariance_ellipse(ax, center, cov, eigenvectors, eigenvalues,
                              n_std=2, **kwargs):
    """Dibuja una elipse de covarianza 2D."""
    if len(eigenvalues) < 2:
        return
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(eigenvalues[0])
    height = 2 * n_std * np.sqrt(eigenvalues[1])
    from matplotlib.patches import Ellipse
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle,
                      **kwargs)
    ax.add_patch(ellipse)


# =============================================================================
# Agregacion por sujeto/estado
# =============================================================================

def analyze_batch(
    root_dir: str | Path,
    output_dir: str | Path,
    subject_filter: str | None = None,
):
    """
    Analiza todos los potenciales en root_dir y genera reportes agregados.
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar por (subject, stage)
    grouped = group_potentials_by_subject_and_stage(root_dir)

    if subject_filter:
        grouped = {k: v for k, v in grouped.items() if k[0] == subject_filter}

    all_results = []

    for (subject, stage), files in sorted(grouped.items()):
        print(f"\n[POST] Subject: {subject} | Stage: {stage} | Files: {len(files)}")
        stage_results = []
        for f in files:
            try:
                res = analyze_potential(f, plot=True, out_dir=output_dir / subject)
                stage_results.append(res)
                all_results.append(res)
                bm = res['basin_metrics']
                if bm:
                    print(f"  {f.name}: {res['n_basins']} basins, "
                          f"ecc={bm[0]['eccentricity']:.3f}, "
                          f"aniso={bm[0]['anisotropy']:.3f}")
            except Exception as e:
                print(f"  ERROR procesando {f}: {e}")

        # Guardar resumen por stage
        summary_file = output_dir / subject / f"summary_{subject}_{stage}.json"
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            'subject': subject,
            'stage': stage,
            'n_epochs': len(stage_results),
            'n_basins_list': [r['n_basins'] for r in stage_results],
            'eccentricity_mean': float(np.mean([r['basin_metrics'][0]['eccentricity']
                                                for r in stage_results if r['basin_metrics']])) if stage_results else None,
            'anisotropy_mean': float(np.mean([r['basin_metrics'][0]['anisotropy']
                                              for r in stage_results if r['basin_metrics']])) if stage_results else None,
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    # Reporte global
    print(f"\n[POST] Analisis completo. Resultados en: {output_dir}")
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-procesamiento de potenciales IgA")
    parser.add_argument("--file", type=str, help="Analizar un solo fichero .npz")
    parser.add_argument("--root-dir", type=str, help="Directorio raiz con resultados")
    parser.add_argument("--output-dir", type=str, default="./postprocess_results")
    parser.add_argument("--subject", type=str, default=None, help="Filtrar por sujeto")
    args = parser.parse_args()

    if args.file:
        res = analyze_potential(args.file, plot=True, out_dir=Path(args.output_dir))
        print(f"\nBasins detectados: {res['n_basins']}")
        for bm in res['basin_metrics']:
            print(f"  Basin {bm['basin_id']}:")
            print(f"    Centroid: ({bm['centroid'][0]:.2f}, {bm['centroid'][1]:.2f})")
            print(f"    Minimum:  ({bm['minimum_pos'][0]:.2f}, {bm['minimum_pos'][1]:.2f})")
            print(f"    Eccentricity: {bm['eccentricity']:.4f}")
            print(f"    Anisotropy:   {bm['anisotropy']:.4f}")
            print(f"    Angle:        {bm['asymmetry_angle']:.1f} deg")
    elif args.root_dir:
        analyze_batch(args.root_dir, args.output_dir, args.subject)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
postprocess_potentials_analytical.py
=====================================
Version analitica del post-procesamiento de potenciales IgA.

Aprovecha la representacion B-spline U^h(x) = sum_j c_j N_j(x) para calcular
analiticamente todos los valores de interes:

  1. Puntos criticos exactos via Newton-Raphson con evaluacion exacta de
     gradiente y Hessiano via B-splines en puntos arbitrarios.
  2. Centroides y momentos de inercia por cuadratura Gauss-Legendre exacta
     (2 puntos/dim son suficientes para B-splines cuadraticos).
  3. Metricas de asimetria con precision sub-celda.
  4. Clasificacion exacta de puntos criticos (minimo/maximo/silla).

Uso:
    python postprocess_potentials_analytical.py --file path/to/potential_data.npz
    python postprocess_potentials_analytical.py --root-dir ./batch_results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import load_potential, find_potential_files


# =============================================================================
# Evaluacion exacta de U, grad U, Hess U en puntos arbitrarios via B-splines
# =============================================================================

def eval_U_and_derivatives(
    coeffs: np.ndarray,
    bspline_space: Any,
    point: np.ndarray,
) -> dict:
    """
    Evalua U, gradiente y Hessiano en un punto arbitrario usando B-splines.
    Utiliza los metodos internos del BSplineTensorSpace.
    
    Returns dict con: 'U', 'grad_U' (array D), 'hess_U' (DxD)
    """
    D = bspline_space.D
    point = np.asarray(point, dtype=float)
    c = np.asarray(coeffs).reshape(bspline_space.n_bases)

    # Para cada dimension: encontrar span y evaluar bases + derivadas
    basis_vals = []       # N_d(x_d) para cada dimension
    basis_derivs = []     # N'_d(x_d)
    basis_second = []     # N''_d(x_d)

    for d in range(D):
        t = bspline_space.knots[d]
        p = bspline_space.degree
        x = point[d]
        n_bases = len(t) - p - 1

        span = _find_span(t, x, p)

        # Evaluar bases y derivadas hasta orden 2
        ders = _eval_basis_ders(t, span, x, p, n=2)

        # Armar arrays completos (sparse)
        N = np.zeros(n_bases)
        dN = np.zeros(n_bases)
        ddN = np.zeros(n_bases)
        for j_local, (v0, v1, v2) in ders.items():
            j_global = span - p + j_local
            if 0 <= j_global < n_bases:
                N[j_global] = v0
                dN[j_global] = v1
                ddN[j_global] = v2

        basis_vals.append(N)
        basis_derivs.append(dN)
        basis_second.append(ddN)

    # Contraer con coeficientes para U
    U_val = _contract_tensor(c, basis_vals)

    # Gradiente: contraccion con derivada en una direccion y bases en las demas
    grad = np.zeros(D)
    for d in range(D):
        factors = [basis_vals[d2] if d2 != d else basis_derivs[d2]
                   for d2 in range(D)]
        grad[d] = _contract_tensor(c, factors)

    # Hessiano
    hess = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            if i == j:
                # Segunda derivada pura
                factors = [basis_vals[d2] if d2 != i else basis_second[d2]
                           for d2 in range(D)]
            else:
                # Derivada mixta
                factors = [basis_vals[d2] if d2 not in (i, j)
                           else (basis_derivs[d2] if d2 == i else basis_derivs[d2])
                           for d2 in range(D)]
                # Correccion: cada dimension que difiere usa su derivada
                factors = [basis_vals[d2] for d2 in range(D)]
                factors[i] = basis_derivs[i]
                factors[j] = basis_derivs[j]
            hess[i, j] = _contract_tensor(c, factors)

    return {'U': U_val, 'grad_U': grad, 'hess_U': hess}


def _contract_tensor(coeffs_tensor, factor_list):
    """Contraccion tensorial: suma_{indices} c_{i1,i2,...} * f1[i1] * f2[i2] * ..."""
    result = coeffs_tensor
    for d in range(len(factor_list)):
        # En cada paso, contraer el eje 0 de result con factor_list[d]
        result = np.tensordot(result, factor_list[d], axes=(0, 0))
    return float(result)


def _find_span(t, x, p):
    """Find span: t[span] <= x < t[span+1]. (Piegl & Tiller, Alg. A2.1)"""
    n = len(t) - p - 1
    if x >= t[n]:
        return n - 1
    if x <= t[p]:
        return p
    low, high = p, n
    span = (low + high) // 2
    while x < t[span] or x >= t[span + 1]:
        if x < t[span]:
            high = span
        else:
            low = span
        span = (low + high) // 2
    return span


def _eval_basis_ders(t, span, x, p, n=2):
    """
    Evalua bases B-spline y derivadas hasta orden n en punto x.
    Retorna dict {j_local: (val, dval, ddval)}.
    
    Implementacion basada en Piegl & Tiller, Algoritmo A2.3.
    """
    ndu = np.zeros((p + 1, p + 1), dtype=float)
    ndu[0, 0] = 1.0
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)

    for j in range(1, p + 1):
        left[j] = x - t[span + 1 - j]
        right[j] = t[span + j] - x
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if denom == 0:
                temp = 0.0
            else:
                temp = ndu[r, j - 1] / denom
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    # Extraer valores y derivadas
    # ders[k, r] = k-esima derivada de la r-esima base local
    ders = np.zeros((n + 1, p + 1), dtype=float)
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    if n >= 1:
        a = np.zeros((2, p + 1), dtype=float)
        for r in range(p + 1):
            s1, s2 = 0, 1
            a[0, 0] = 1.0
            for k in range(1, n + 1):
                d = 0.0
                rk = r - k
                pk = p - k
                if r >= k:
                    denom = t[span + r + 1] - t[span + r + 1 - k]
                    a[s2, 0] = a[s1, 0] / denom if denom != 0 else 0.0
                    d = a[s2, 0] * ndu[rk, pk]
                if r >= k + 1:
                    for j_idx in range(1, k):
                        denom = t[span + r + j_idx + 1] - t[span + r + j_idx + 1 - k]
                        a[s2, j_idx] = (a[s1, j_idx] - a[s1, j_idx - 1]) / denom if denom != 0 else 0.0
                        d += a[s2, j_idx] * ndu[rk + j_idx, pk]
                if r <= p - 1:
                    denom = t[span + r + k + 1] - t[span + r + 1]
                    a[s2, k] = -a[s1, k - 1] / denom if denom != 0 else 0.0
                    d += a[s2, k] * ndu[r, pk]
                ders[k, r] = d
                j_idx = s1
                s1 = s2
                s2 = j_idx
                if p <= k:
                    for j_idx2 in range(r + 1, p + 1):
                        a[s1, j_idx2] = 0.0
                        a[s2, j_idx2] = 0.0

        # Multiplicar por factorial
        r = p
        for k in range(1, n + 1):
            for j in range(p + 1):
                ders[k, j] *= r
            r *= (p - k)

    return {j: (ders[0, j], ders[1, j] if n >= 1 else 0.0,
                ders[2, j] if n >= 2 else 0.0)
            for j in range(p + 1)}


# =============================================================================
# Newton-Raphson para puntos criticos exactos
# =============================================================================

def find_critical_points_analytical(
    coeffs: np.ndarray,
    bspline_space: Any,
    edges: list[np.ndarray],
    density_mask: np.ndarray | None = None,
    tol: float = 1e-12,
    max_iter: int = 30,
) -> list[dict]:
    """
    Encuentra puntos criticos exactos usando Newton-Raphson con evaluacion
    exacta de gradiente y Hessiano via B-splines.
    
    Inicializa desde cada celda valida de la grilla.
    """
    D = bspline_space.D
    c = np.asarray(coeffs).reshape(bspline_space.n_bases)
    grid_shape = tuple(len(e) for e in edges)

    candidates = []

    for idx in np.ndindex(grid_shape):
        if density_mask is not None and not density_mask[idx]:
            continue

        # Punto inicial: centro de celda
        x0 = np.array([edges[d][idx[d]] for d in range(D)], dtype=float)

        # Verificar que |grad| no sea demasiado grande (posible cero)
        val0 = eval_U_and_derivatives(c, bspline_space, x0)
        if np.linalg.norm(val0['grad_U']) > 10.0:
            continue

        candidates.append({'x0': x0, 'idx': idx, 'grad0_norm': np.linalg.norm(val0['grad_U'])})

    # Ordenar por norma de gradiente ascendente (mas prometedores primero)
    candidates.sort(key=lambda c: c['grad0_norm'])

    # Newton-Raphson desde cada candidato
    all_points = []
    tol_dist = min(np.min(np.diff(e)) for e in edges if len(e) > 1) * 0.05

    for cand in candidates[:min(len(candidates), 500)]:  # limitar candidatos
        x = cand['x0'].copy()
        converged = False

        for it in range(max_iter):
            val = eval_U_and_derivatives(c, bspline_space, x)
            grad = val['grad_U']
            hess = val['hess_U']
            gnorm = np.linalg.norm(grad)

            if gnorm < tol:
                converged = True
                break

            try:
                delta = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                delta = -np.linalg.lstsq(hess, grad, rcond=None)[0]

            # Backtracking line search
            alpha = 1.0
            for _ in range(8):
                x_new = x + alpha * delta
                # Verificar que sigue en region valida
                in_bounds = True
                if density_mask is not None:
                    idx_new = tuple(
                        max(0, min(int(np.searchsorted(edges[d], x_new[d])),
                                   len(edges[d]) - 1))
                        for d in range(D)
                    )
                    if not density_mask[idx_new]:
                        in_bounds = False
                if in_bounds:
                    val_new = eval_U_and_derivatives(c, bspline_space, x_new)
                    if np.linalg.norm(val_new['grad_U']) < gnorm:
                        x = x_new
                        break
                alpha *= 0.5
            else:
                x = x + 0.1 * delta

        if converged:
            # Re-evaluar en punto convergido
            val = eval_U_and_derivatives(c, bspline_space, x)
            hess = val['hess_U']
            eigvals = np.linalg.eigvalsh(hess)

            if all(ev > 1e-10 for ev in eigvals):
                ptype = "minimum"
            elif all(ev < -1e-10 for ev in eigvals):
                ptype = "maximum"
            elif any(ev > 1e-10 for ev in eigvals) and any(ev < -1e-10 for ev in eigvals):
                ptype = "saddle"
            else:
                ptype = "degenerate"

            all_points.append({
                'position': x.copy(),
                'U_value': val['U'],
                'grad_norm': np.linalg.norm(val['grad_U']),
                'hessian': hess,
                'eigenvalues': eigvals,
                'point_type': ptype,
                'iterations': it + 1,
            })

    # Eliminar duplicados
    unique = []
    for pt in all_points:
        is_dup = False
        for existing in unique:
            if np.linalg.norm(pt['position'] - existing['position']) < tol_dist:
                is_dup = True
                break
        if not is_dup:
            unique.append(pt)

    return unique


# =============================================================================
# Cuadratura Gaussiana exacta para centroides y momentos
# =============================================================================

def gauss_quadrature_basin(
    coeffs: np.ndarray,
    bspline_space: Any,
    edges: list[np.ndarray],
    basin_mask: np.ndarray,
    density: np.ndarray | None = None,
    n_gauss: int = 3,
) -> dict:
    """
    Integra sobre la cuenca usando cuadratura Gauss-Legendre exacta.
    Para B-splines de grado p=2, n_gauss=2 es suficiente para U y momentos
    primeros; n_gauss=3 para momentos segundos exactos.
    """
    D = bspline_space.D
    c = np.asarray(coeffs).reshape(bspline_space.n_bases)
    xi, wi = np.polynomial.legendre.leggauss(n_gauss)

    M0 = 0.0   # masa total
    M1 = np.zeros(D)  # primer momento (centroide * masa)
    M2 = np.zeros((D, D))  # segundo momento
    vol = 0.0

    grid_shape = tuple(len(e) for e in edges)

    for idx in np.ndindex(grid_shape):
        if not basin_mask[idx]:
            continue

        # Limites de la celda
        bounds = []
        for d in range(D):
            e = edges[d]
            i = idx[d]
            left = e[0] - 0.5*(e[1]-e[0]) if i == 0 else 0.5*(e[i-1]+e[i])
            right = e[-1] + 0.5*(e[-1]-e[-2]) if i == len(e)-1 else 0.5*(e[i]+e[i+1])
            bounds.append((left, right))

        cell_vol = np.prod([b - a for a, b in bounds])
        vol += cell_vol

        # Mapear puntos de Gauss a la celda
        pts_1d = []
        wts_1d = []
        for d in range(D):
            a, b = bounds[d]
            pts_1d.append(0.5*(b-a)*xi + 0.5*(b+a))
            wts_1d.append(0.5*(b-a)*wi)

        # Iterar sobre producto cartesiano
        from itertools import product
        for multi_idx in product(range(n_gauss), repeat=D):
            x = np.array([pts_1d[d][multi_idx[d]] for d in range(D)])
            w = np.prod([wts_1d[d][multi_idx[d]] for d in range(D)])

            # Evaluar U en el punto de Gauss
            val = eval_U_and_derivatives(c, bspline_space, x)
            rho = density[idx] if density is not None else 1.0

            weight = w * max(rho, 0)
            M0 += weight
            M1 += weight * x
            M2 += weight * np.outer(x, x)

    if M0 > 0:
        centroid = M1 / M0
        cov = M2 / M0 - np.outer(centroid, centroid)
    else:
        centroid = np.zeros(D)
        cov = np.zeros((D, D))

    return {
        'centroid': centroid,
        'covariance': cov,
        'total_mass': M0,
        'volume': vol,
    }


# =============================================================================
# Analisis completo
# =============================================================================

def analyze_potential_analytical(
    filepath: str | Path,
    n_gauss: int = 3,
    plot: bool = True,
    out_dir: str | Path | None = None,
) -> dict:
    """Analisis completo analitico de un potencial IgA."""
    filepath = Path(filepath)
    data = load_potential(filepath)

    potential = data["potential"]
    coeffs = data["coefficients"]
    density = data["density"]
    density_mask = data["density_mask"]
    edges = data["edges"]
    bspline_space = data.get("bspline_space")
    metadata = data.get("metadata", {})

    D = potential.ndim
    subject = metadata.get('subject', '?')
    stage = metadata.get('stage_label', '?')
    print(f"  [ANALYTICAL] {subject} | {stage} | shape={potential.shape} | D={D}")

    # -----------------------------------------------------------------
    # 1. Puntos criticos exactos via Newton-Raphson + B-splines
    # -----------------------------------------------------------------
    critical_points = []
    minima = []

    if bspline_space is not None and D <= 3:
        print(f"  [ANALYTICAL] Finding critical points (Newton-Raphson + exact B-spline eval)...")
        critical_points = find_critical_points_analytical(
            coeffs, bspline_space, edges, density_mask
        )
        minima = [cp for cp in critical_points if cp['point_type'] == 'minimum']
        others = [cp for cp in critical_points if cp['point_type'] != 'minimum']
        print(f"  [ANALYTICAL] {len(minima)} minima, {len(others)} other critical points")
        for cp in critical_points:
            print(f"    {cp['point_type']:12s} at ({cp['position'][0]:+.4f}, {cp['position'][1]:+.4f}) "
                  f"U={cp['U_value']:.4f} |grad|={cp['grad_norm']:.2e} iters={cp['iterations']}")
    else:
        # Fallback: usar metodo numerico
        from postprocess_potentials import detect_basins_gradient_following
        br = detect_basins_gradient_following(potential, density_mask, edges)
        for i, coord in enumerate(br['minima_coords']):
            pos = np.array([edges[d][coord[d]] for d in range(D)])
            minima.append({
                'position': pos,
                'U_value': potential[tuple(coord)],
                'point_type': 'minimum',
                'iterations': 0,
            })
        print(f"  [ANALYTICAL] Fallback: {len(minima)} minima (grid-based)")

    # -----------------------------------------------------------------
    # 2. Centroides exactos por cuadratura Gaussiana
    # -----------------------------------------------------------------
    basin_metrics = []

    for b_id, min_pt in enumerate(minima, start=1):
        min_pos = min_pt['position']

        # Mascara del basin
        if len(minima) == 1:
            basin_mask = density_mask.copy()
        else:
            from postprocess_potentials import detect_basins_gradient_following
            br = detect_basins_gradient_following(potential, density_mask, edges)
            basin_mask = (br['basin_labels'] == b_id) & density_mask

        # Integracion exacta
        if bspline_space is not None:
            print(f"  [ANALYTICAL] Basin {b_id}: Gauss-Legendre quadrature (n={n_gauss})...")
            int_res = gauss_quadrature_basin(
                coeffs, bspline_space, edges, basin_mask, density, n_gauss
            )
            centroid = int_res['centroid']
            cov = int_res['covariance']
            vol = int_res['volume']
            area_px = int(np.sum(basin_mask))

            eigvals, eigvecs = np.linalg.eigh(cov)
            idx = eigvals.argsort()[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
        else:
            # Fallback
            from postprocess_potentials import compute_basin_asymmetry, detect_basins_gradient_following
            br = detect_basins_gradient_following(potential, density_mask, edges)
            fb = compute_basin_asymmetry(potential, density, density_mask, br['basin_labels'], b_id, edges)
            centroid = np.array(fb['centroid'])
            eigvals = np.array(fb['eigenvalues'])
            eigvecs = fb['eigenvectors']
            vol = fb.get('effective_radius', 1.0)**2 * np.pi
            area_px = fb['area_pixels']

        # Metricas
        disp_vec = centroid - min_pos
        displacement = float(np.linalg.norm(disp_vec))
        eff_radius = np.sqrt(vol / np.pi) if D == 2 else ((3*vol)/(4*np.pi))**(1/3)
        eccentricity = displacement / (eff_radius + 1e-10)

        if D == 2:
            min_ev = max(eigvals[-1], 1e-6 * eigvals[0]) if eigvals[0] > 0 else 1e-10
            anisotropy = float(np.clip(np.sqrt(eigvals[0] / min_ev), 1.0, 100.0))
            angle = float(np.degrees(np.arctan2(disp_vec[1], disp_vec[0])))
        else:
            min_ev = max(eigvals[-1], 1e-6 * eigvals[0]) if eigvals[0] > 0 else 1e-10
            anisotropy = float(np.clip(np.sqrt(eigvals[0] / min_ev), 1.0, 100.0))
            angle = np.nan

        basin_metrics.append({
            'basin_id': b_id,
            'centroid': tuple(float(v) for v in centroid),
            'minimum_pos': tuple(float(v) for v in min_pos),
            'minimum_U': float(min_pt['U_value']),
            'displacement': displacement,
            'effective_radius': float(eff_radius),
            'eccentricity': eccentricity,
            'anisotropy': anisotropy,
            'asymmetry_angle': angle,
            'eigenvalues': tuple(float(v) for v in eigvals),
            'eigenvectors': eigvecs,
            'area_pixels': area_px,
            'newton_iterations': min_pt.get('iterations', 0),
        })

    result = {
        'filepath': str(filepath),
        'metadata': metadata,
        'n_basins': len(minima),
        'potential_shape': potential.shape,
        'critical_points': critical_points,
        'basin_metrics': basin_metrics,
        'analytical': True,
    }

    if plot and out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _plot_analytical(result, potential, density_mask, edges, out_dir)

    return result


def _plot_analytical(result, potential, density_mask, edges, out_dir):
    """Figura de diagnostico con layout mejorado via gridspec."""
    D = potential.ndim
    if D != 2:
        return

    meta = result['metadata']
    subj, stage = meta.get('subject', '?'), meta.get('stage_label', '?')

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35,
                          left=0.06, right=0.94, top=0.92, bottom=0.06)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]

    fig.suptitle(f"ANALYTICAL | {subj} | {stage} | {result['n_basins']} basin(s)", fontsize=16)

    colors = {'minimum': 'green', 'maximum': 'red', 'saddle': 'blue', 'degenerate': 'gray'}

    # --- Panel 1: Potencial + minimos exactos + centroides ---
    ax = axes[0][0]
    X, Y = np.meshgrid(edges[0], edges[1], indexing="ij")
    U_plot = np.where(density_mask, potential, np.nan)
    im = ax.contourf(X, Y, U_plot, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, label="U", shrink=0.75)
    for b in result['basin_metrics']:
        cx, cy = b['centroid']
        mx, my = b['minimum_pos']
        ax.plot(cx, cy, "r*", markersize=15, label="Centroid (Gauss)")
        ax.plot(mx, my, "w+", markersize=15, mew=2, label="Min (Newton)")
        ax.annotate("", xy=(cx, cy), xytext=(mx, my),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        _plot_ellipse(ax, b['centroid'], b['eigenvectors'], b['eigenvalues'],
                      edgecolor="yellow", facecolor="none", lw=2)
    ax.set_title("Potential + Exact Minima & Centroids", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # --- Panel 2: Puntos criticos clasificados ---
    ax = axes[0][1]
    plotted_types = set()
    for cp in result.get('critical_points', []):
        pos = cp['position']
        c = colors.get(cp['point_type'], 'black')
        label = cp['point_type'] if cp['point_type'] not in plotted_types else None
        ax.scatter(pos[0], pos[1], c=c, s=120, edgecolors='black', zorder=5, label=label)
        plotted_types.add(cp['point_type'])
    ax.set_title(f"Critical Points: {len(result.get('critical_points', []))}", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if plotted_types:
        ax.legend(fontsize=8, loc='upper right')

    # --- Panel 3: Asimetria ---
    ax = axes[0][2]
    m = result['basin_metrics']
    if m:
        ids = [f"B{mm['basin_id']}" for mm in m]
        eccs = [mm['eccentricity'] for mm in m]
        anis = [mm['anisotropy'] for mm in m]
        x = np.arange(len(ids))
        w = 0.35
        ax.bar(x - w/2, eccs, w, label="Eccentricity", color="steelblue", edgecolor='black')
        ax.bar(x + w/2, anis, w, label="Anisotropy", color="coral", edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(ids)
        ax.set_ylabel("Value")
        ax.set_title("Asymmetry Metrics (Analytical)", fontsize=11)
        ax.legend(fontsize=9)
        ax.axhline(0, color='black', linewidth=0.5)

    # --- Panel 4: Tipos de puntos criticos ---
    ax = axes[1][0]
    types = [cp['point_type'] for cp in result.get('critical_points', [])]
    if types:
        from collections import Counter
        cnt = Counter(types)
        ax.bar(cnt.keys(), cnt.values(), color=[colors.get(t, 'gray') for t in cnt.keys()],
               edgecolor='black')
        ax.set_ylabel("Count")
        ax.set_title("Critical Point Classification", fontsize=11)
    else:
        ax.text(0.5, 0.5, "No critical points found\n(fallback mode)",
                ha='center', va='center', transform=ax.transAxes, fontsize=10,
                style='italic', color='gray')
        ax.set_title("Critical Points", fontsize=11)

    # --- Panel 5: Eigenvalues del Hessiano ---
    ax = axes[1][1]
    for b in result['basin_metrics']:
        evs = list(b['eigenvalues'])
        xs = list(range(1, len(evs) + 1))
        ax.scatter(xs, evs, s=100, label=f"B{b['basin_id']}", edgecolors='black', zorder=5)
        ax.plot(xs, evs, 'k--', alpha=0.3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([r'$\lambda_{\max}$', r'$\lambda_{\min}$'])
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Curvature at Minima (Hessian)", fontsize=11)
    ax.set_yscale("log")
    ax.axhline(1e-10, color='red', linestyle='--', alpha=0.5, label='threshold')
    ax.legend(fontsize=8)

    # --- Panel 6: Resumen textual ---
    ax = axes[1][2]
    ax.axis("off")
    summary = f"Method: ANALYTICAL (Newton + Gauss Quad)\n"
    summary += f"Critical points: {len(result.get('critical_points', []))}\n"
    summary += f"Basins: {result['n_basins']}\n\n"
    for b in result['basin_metrics']:
        summary += (
            f"Basin {b['basin_id']}:\n"
            f"  Min:    ({b['minimum_pos'][0]:+.4f}, {b['minimum_pos'][1]:+.4f})\n"
            f"  U(min): {b['minimum_U']:.4f}\n"
            f"  Centr:  ({b['centroid'][0]:+.4f}, {b['centroid'][1]:+.4f})\n"
            f"  Ecc:    {b['eccentricity']:.4f}\n"
            f"  Aniso:  {b['anisotropy']:.4f}\n"
            f"  Angle:  {b['asymmetry_angle']:.1f} deg\n"
            f"  Newton: {b['newton_iterations']} iters\n\n"
        )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fname = out_dir / f"analytical_{subj}_{stage}.png"
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fname), dpi=150, bbox_inches='tight')
    plt.close("all")
    print(f"  [ANALYTICAL] Plot saved: {fname}")


def _plot_ellipse(ax, center, eigvecs, eigvals, n_std=2, **kwargs):
    from matplotlib.patches import Ellipse
    if len(eigvals) < 2:
        return
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    w = 2 * n_std * np.sqrt(max(eigvals[0], 0))
    h = 2 * n_std * np.sqrt(max(eigvals[1], 0))
    ax.add_patch(Ellipse(xy=center, width=w, height=h, angle=angle, **kwargs))


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Post-procesamiento ANALITICO de potenciales IgA")
    parser.add_argument("--file", type=str, help="Analizar un fichero .npz")
    parser.add_argument("--root-dir", type=str, help="Directorio raiz con resultados")
    parser.add_argument("--output-dir", type=str, default="./postprocess_analytical")
    parser.add_argument("--gauss-points", type=int, default=3)
    args = parser.parse_args()

    if args.file:
        res = analyze_potential_analytical(
            args.file, n_gauss=args.gauss_points, plot=True, out_dir=args.output_dir
        )
        print(f"\\n=== RESULTADOS ANALITICOS ===")
        print(f"Basins: {res['n_basins']}")
        for bm in res['basin_metrics']:
            print(f"\\n  Basin {bm['basin_id']}:")
            print(f"    Minimo exacto:  ({bm['minimum_pos'][0]:+.6f}, {bm['minimum_pos'][1]:+.6f})")
            print(f"    U(min):         {bm['minimum_U']:.6f}")
            print(f"    Centroid Gauss: ({bm['centroid'][0]:+.6f}, {bm['centroid'][1]:+.6f})")
            print(f"    Displacement:   {bm['displacement']:.6f}")
            print(f"    Eccentricity:   {bm['eccentricity']:.6f}")
            print(f"    Anisotropy:     {bm['anisotropy']:.6f}")
            print(f"    Newton iters:   {bm['newton_iterations']}")
    elif args.root_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = find_potential_files(args.root_dir)
        print(f"Encontrados {len(files)} potenciales")
        for f in files:
            try:
                analyze_potential_analytical(f, n_gauss=3, plot=True, out_dir=out_dir)
            except Exception as e:
                print(f"ERROR en {f}: {e}")
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

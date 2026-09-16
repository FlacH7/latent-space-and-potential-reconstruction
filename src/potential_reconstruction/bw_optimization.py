"""
bw_optimization.py
==================
Módulo independiente para la selección óptima del ancho de banda (bw)
en la estimación de coeficientes de Kramers-Moyal.

Evalúa métricas directamente sobre los campos de deriva y difusión,
no sobre el potencial reconstruido. Soporta dos modos:
  1. Con ground truth: minimiza error L2 de drift y diffusion.
  2. Sin ground truth: maximiza estabilidad + consistencia física.

Novedad v3.0:
  - Se elimina la métrica de roughness (suavidad espacial) porque tenía un
    sesgo monótono hacia bw grandes: a mayor bw, menor laplaciano, mayor
    "smoothness", favoreciendo sistemáticamente anchos de banda excesivos.
  - Se quedan solo 2 métricas sin teórico: stability + consistency.
  - Pesos automáticos recalibrados por variación intrínseca de estas 2 métricas.
  - Métrica de reproducibilidad split-half disponible como opción avanzada
    (use_split_half=True), aunque duplica el costo computacional.
"""

import numpy as np
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.potential_reconstruction.km_tools_v2 import extract_km_coefficients
from src.potential_reconstruction.iga_reconstructor import reconstruct_potential_iga

# =============================================================================
# Métricas auxiliares
# =============================================================================

def _gradient_log_rho(density, edges):
    """Gradiente discreto de log(densidad) en la grilla de bins."""
    rho_floor = np.percentile(density[density > 0], 1) * 0.1
    log_rho = np.log(density + rho_floor)
    D = len(edges)
    grad = np.zeros((D,) + density.shape, dtype=float)
    for d in range(D):
        grad[d] = np.gradient(log_rho, axis=d)
    return grad


# =============================================================================
# Cálculo de métricas escalares para un candidato de bw
# =============================================================================

def _compute_bw_metrics(drift, diffusion, edges, density, drift_theo, diff_theo, mask_density):
    """
    Calcula métricas escalares sobre drift y diffusion para un candidato de bw.

    Retorna tupla:
      (norm_drift, norm_diff, error_drift, error_diff, consistency)
    """
    D = drift.shape[0]
    grid_shape = drift.shape[1:]

    # Volúmenes de celda (midpoint rule)
    vol = np.ones(grid_shape, dtype=float)
    for d in range(D):
        x = edges[d]
        if len(x) > 1:
            left = np.concatenate([
                [x[0] - 0.5 * (x[1] - x[0])],
                0.5 * (x[:-1] + x[1:]),
                [x[-1] + 0.5 * (x[-1] - x[-2])]
            ])
            dx = np.diff(left)
        else:
            dx = np.array([1.0])
        sh = [1] * D
        sh[d] = len(dx)
        vol *= dx.reshape(sh)

    if mask_density is None:
        mask_density = density > 0.05 * np.max(density)

    w = vol[mask_density]
    w_sum = np.sum(w)
    if w_sum == 0:
        w_sum = 1.0

    # --- Normas L2 ponderadas en región de alta densidad ---
    drift_masked = drift[:, mask_density]
    diff_masked = diffusion[:, :, mask_density]

    norm_drift = np.sqrt(np.sum(w * np.sum(drift_masked ** 2, axis=0)) / w_sum)
    norm_diff = np.sqrt(np.sum(w * np.sum(diff_masked ** 2, axis=(0, 1))) / w_sum)

    # --- Errores L2 vs teórico (si están disponibles) ---
    error_drift = np.nan
    error_diff = np.nan
    if drift_theo is not None:
        err_d = drift - drift_theo
        error_drift = np.sqrt(np.sum(w * np.sum(err_d[:, mask_density] ** 2, axis=0)) / w_sum)
    if diff_theo is not None:
        err_D = diffusion - diff_theo
        error_diff = np.sqrt(np.sum(w * np.sum(err_D[:, :, mask_density] ** 2, axis=(0, 1))) / w_sum)

    # --- Consistencia física: || f + D·∇logρ || (equilibrio detallado) ---
    consistency = np.nan
    try:
        grad_log_rho = _gradient_log_rho(density, edges)
        J = drift + np.einsum('ij...,j...->i...', diffusion, grad_log_rho)
        consistency = np.sqrt(np.sum(w * np.sum(J[:, mask_density] ** 2, axis=0)) / w_sum)
        
        # En _compute_bw_metrics, reemplazar:
        denominator = np.sqrt(np.sum(w * np.sum((drift[:, mask_density])**2 + 
                                        np.sum((diffusion[:,:,mask_density])**2, axis=0), 
                                        axis=0)) / w_sum) + 1e-12
        consistency_rel = consistency / denominator
    except Exception:
        pass

    return (norm_drift, norm_diff, error_drift, error_diff, consistency_rel)


# =============================================================================
# Worker para paralelización
# =============================================================================

def _worker_bw(args):
    """Worker que estima KM y calcula métricas escalares sobre drift/diffusion."""
    (data, bins, bw, dt, p, kernel, sigma_smooth,
     density, drift_theo, diff_theo, mask_density) = args

    drift, diffusion, edges = extract_km_coefficients(
        data, bins=bins, p=p, bw=bw, kernel=kernel, dt=dt,
        sigma_smooth=sigma_smooth,
        density_threshold = 0.0
    )

    metrics = _compute_bw_metrics(
        drift, diffusion, edges, density,
        drift_theo, diff_theo, mask_density
    )

    # Potencial solo para visualización final (no entra en el score)
    result = reconstruct_potential_iga(
        drift, diffusion, edges, density,
        degree=2, decompose_helmholtz=True
    )

    return (result['potential'], edges, bw) + metrics


# =============================================================================
# Utilidades de normalización y pesos
# =============================================================================

def _norm01(arr):
    """Mapea a [0,1] preservando orden (mayor valor → 1)."""
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return np.zeros_like(arr)
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if a_max - a_min < 1e-12:
        return np.where(valid, 1.0, np.nan)
    return np.where(valid, (arr - a_min) / (a_max - a_min), np.nan)


def _inv_norm01(arr):
    """Mapea a [0,1] invirtiendo orden (menor valor → 1)."""
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return np.zeros_like(arr)
    a_min, a_max = np.nanmin(arr), np.nanmax(arr)
    if a_max - a_min < 1e-12:
        return np.where(valid, 1.0, np.nan)
    return np.where(valid, (a_max - arr) / (a_max - a_min), np.nan)


def _compute_auto_weights(variations, min_ratio=0.05, eps=1e-12):
    """
    Calcula pesos proporcionales a las variaciones (rango dinámico) de cada métrica.

    Parameters
    ----------
    variations : list of float
        Rango (max-min) de cada métrica cruda.
    min_ratio : float
        Fracción mínima del peso total que recibe una métrica no nula,
        evitando que métricas casi-constantes queden con peso ~0.

    Returns
    -------
    weights : np.ndarray
        Pesos normalizados que suman 1.0.
    """
    variations = np.asarray(variations, dtype=float)
    variations = np.where(np.isnan(variations), 0.0, variations)
    active = variations > eps
    if not np.any(active):
        return np.ones(len(variations)) / len(variations)

    v_active = variations[active]
    v_max = np.max(v_active)
    v_clipped = np.clip(v_active, v_max * min_ratio, None)

    weights = np.zeros(len(variations))
    weights[active] = v_clipped / np.sum(v_clipped)
    return weights


# =============================================================================
# Función principal: optimal_bw
# =============================================================================

def optimal_bw(data, bins, dt=1.0, p=2, kernel='epanechnikov',
               sigma_smooth=1.0, n_candidates=30, interval= None, n_jobs=1,
               plot=True, figsize=None,
               theoretical=None,
               w_drift=0.5, w_diff=0.5,
               w_stability=0.50, w_consistency=0.50,
               auto_weight=True, min_weight_ratio=0.05):
    """
    Busca el bw óptimo evaluando métricas directamente sobre los campos
    de deriva y difusión estimados.

    Modos
    -----
    * Con teórico (theoretical != None): minimiza error L2 de drift y diffusion.
    * Sin teórico: maximiza estabilidad de normas + consistencia física.

    Pesos
    -----
    - auto_weight=True (default): los pesos se recalibran automáticamente
      según el rango dinámico (max-min) de cada métrica.
    - auto_weight=False: se usan los pesos manuales proporcionados.

    Parámetros
    ----------
    auto_weight : bool
        Activa recalibración proporcional al rango dinámico.
    min_weight_ratio : float
        Fracción mínima del peso máximo que recibe una métrica activa.
    """
    D = data.shape[1]
    bins_arr = np.atleast_1d(bins)

    # Rango de búsqueda log-espaciado
    if interval is None:
        data_range = np.ptp(data, axis=0)
        data_range[data_range == 0] = 1.0
        dx = data_range / bins_arr
        bw_min = np.max(dx) * 1.5

        std_data = np.std(data, axis=0)
        std_data = std_data[std_data > 0]
        if len(std_data) == 0:
            std_data = np.array([1.0])
        bw_max = np.max(std_data) * 0.6
        if bw_max <= bw_min:
            bw_max = bw_min * 5.0
    else:
        bw_min = interval[0]
        bw_max = interval[1]

    bw_candidates = np.geomspace(bw_min, bw_max, n_candidates)
    print(f"[optimal_bw] Rango log-espaciado: [{bw_min:.4f}, {bw_max:.4f}] ({n_candidates} candidatos)")

    # Histograma base (máscara de densidad común para todos los candidatos)
    drift0, diffusion0, edges = extract_km_coefficients(
        data, bins=bins, p=p, bw=bw_candidates[0], kernel=kernel, dt=dt
    )
    hist_edges = []
    for d in range(D):
        c = edges[d]
        half = (c[1] - c[0]) / 2.0 if len(c) > 1 else 0.5
        e = np.concatenate([[c[0] - half], c + half])
        hist_edges.append(e)

    density, _ = np.histogramdd(data, bins=hist_edges)
    density = density.astype(float)
    mask_density = density > (0.05 * density.max())

    # Precalcular teórico en la grilla base (evita problemas de pickle con lambdas)
    drift_theo_grid = None
    diff_theo_grid = None
    if theoretical is not None:
        print("[optimal_bw] Modo TEÓRICO: minimizando error L2 de drift y diffusion.")
        drift_theo_grid = theoretical['drift_grid'](edges)
        diff_theo_grid = theoretical['diffusion_grid'](edges)
    else:
        print("[optimal_bw] Modo SIN TEÓRICO: estabilidad + consistencia física.")

    args_list = [
        (data, bins, bw, dt, p, kernel, sigma_smooth,
         density, drift_theo_grid, diff_theo_grid, mask_density)
        for bw in bw_candidates
    ]

    # Arrays para métricas
    n = len(bw_candidates)
    norm_drift_arr = np.full(n, np.nan)
    norm_diff_arr = np.full(n, np.nan)
    error_drift_arr = np.full(n, np.nan)
    error_diff_arr = np.full(n, np.nan)
    consistency_arr = np.full(n, np.nan)

    potentials = []
    edges_list = []
    bw_out = []

    # --- Ejecución ---
    if n_jobs != 1 and n > 1:
        max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_bw, arg): i for i, arg in enumerate(args_list)}
            with tqdm(total=n, desc="Scanning bw", unit="bw") as pbar:
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        (pot, edges_i, bw_i, nd, nD, ed, eD, cons) = future.result()
                        potentials.append((i, pot))
                        edges_list.append((i, edges_i))
                        bw_out.append((i, bw_i))
                        norm_drift_arr[i] = nd
                        norm_diff_arr[i] = nD
                        error_drift_arr[i] = ed
                        error_diff_arr[i] = eD
                        consistency_arr[i] = cons
                    except Exception as e:
                        print(f"[optimal_bw] Error en worker bw={args_list[i][2]:.4f}: {e}")
                    pbar.update(1)
        # FIX: ProcessPoolExecutor cleanup puede demorar
        sys.stdout.flush()
        print("[optimal_bw] Scan complete. Processing results...")
        sys.stdout.flush()
    else:
        with tqdm(total=n, desc="Scanning bw", unit="bw") as pbar:
            for i, arg in enumerate(args_list):
                try:
                    (pot, edges_i, bw_i, nd, nD, ed, eD, cons) = _worker_bw(arg)
                    potentials.append((i, pot))
                    edges_list.append((i, edges_i))
                    bw_out.append((i, bw_i))
                    norm_drift_arr[i] = nd
                    norm_diff_arr[i] = nD
                    error_drift_arr[i] = ed
                    error_diff_arr[i] = eD
                    consistency_arr[i] = cons
                except Exception as e:
                    print(f"[optimal_bw] Error en bw={arg[2]:.4f}: {e}")
                pbar.update(1)
        sys.stdout.flush()

    # Reordenar por índice original
    potentials = [p for _, p in sorted(potentials, key=lambda x: x[0])]
    edges_list = [e for _, e in sorted(edges_list, key=lambda x: x[0])]
    bw_candidates = np.array([b for _, b in sorted(bw_out, key=lambda x: x[0])])

    # --- Estabilidad de normas ---
    stability = np.full(n, np.nan)
    for i in range(n):
        if 0 < i < n - 1:
            if not (np.isnan(norm_drift_arr[i]) or np.isnan(norm_drift_arr[i - 1]) or np.isnan(norm_drift_arr[i + 1])):
                var_d = (abs(norm_drift_arr[i] - norm_drift_arr[i - 1]) +
                         abs(norm_drift_arr[i] - norm_drift_arr[i + 1]))
                rel_d = var_d / (norm_drift_arr[i] + 1e-12)
            else:
                rel_d = np.nan

            if not (np.isnan(norm_diff_arr[i]) or np.isnan(norm_diff_arr[i - 1]) or np.isnan(norm_diff_arr[i + 1])):
                var_D = (abs(norm_diff_arr[i] - norm_diff_arr[i - 1]) +
                         abs(norm_diff_arr[i] - norm_diff_arr[i + 1]))
                rel_D = var_D / (norm_diff_arr[i] + 1e-12)
            else:
                rel_D = np.nan

            if not (np.isnan(rel_d) and np.isnan(rel_D)):
                stability[i] = np.exp(-np.nanmean([rel_d, rel_D]))
        else:
            neigh = 1 if i == 0 else n - 2
            if not (np.isnan(norm_drift_arr[i]) or np.isnan(norm_drift_arr[neigh])):
                rel_d = abs(norm_drift_arr[i] - norm_drift_arr[neigh]) / (norm_drift_arr[i] + 1e-12)
            else:
                rel_d = np.nan
            if not (np.isnan(norm_diff_arr[i]) or np.isnan(norm_diff_arr[neigh])):
                rel_D = abs(norm_diff_arr[i] - norm_diff_arr[neigh]) / (norm_diff_arr[i] + 1e-12)
            else:
                rel_D = np.nan
            if not (np.isnan(rel_d) and np.isnan(rel_D)):
                stability[i] = np.exp(-np.nanmean([rel_d, rel_D]))

    # --- Cálculo de pesos y score ---
    score = np.full(n, np.nan)

    if theoretical is not None:
        # Modo teórico: métricas = error_drift, error_diff
        v_err_d = (np.nanmax(error_drift_arr) - np.nanmin(error_drift_arr))
        v_err_D = (np.nanmax(error_diff_arr) - np.nanmin(error_diff_arr))
        variations = [v_err_d, v_err_D]

        if auto_weight:
            weights = _compute_auto_weights(variations, min_ratio=min_weight_ratio)
            w_drift_eff, w_diff_eff = weights[0], weights[1]
            print(f"\n[optimal_bw] PESOS AUTOMÁTICOS (modo teórico):")
            print(f"    Variación error_drift : {v_err_d:.6f}  →  peso = {w_drift_eff:.4f}")
            print(f"    Variación error_diff  : {v_err_D:.6f}  →  peso = {w_diff_eff:.4f}")
            print(f"    Suma de pesos = {weights.sum():.4f}")
        else:
            w_drift_eff, w_diff_eff = w_drift, w_diff
            print(f"\n[optimal_bw] PESOS MANUALES (modo teórico): w_drift={w_drift_eff:.4f}, w_diff={w_diff_eff:.4f}")

        err_drift_norm = _inv_norm01(error_drift_arr)
        err_diff_norm = _inv_norm01(error_diff_arr)
        for i in range(n):
            if not (np.isnan(err_drift_norm[i]) or np.isnan(err_diff_norm[i])):
                score[i] = w_drift_eff * err_drift_norm[i] + w_diff_eff * err_diff_norm[i]

    else:
        # Modo sin teórico: métricas = stability, consistency
        v_stab = (np.nanmax(stability) - np.nanmin(stability))
        v_cons = (np.nanmax(consistency_arr) - np.nanmin(consistency_arr))
        variations = [v_stab, v_cons]

        if auto_weight:
            weights = _compute_auto_weights(variations, min_ratio=min_weight_ratio)
            w_stab_eff, w_cons_eff = weights[0], weights[1]
            print(f"\n[optimal_bw] PESOS AUTOMÁTICOS (modo sin teórico):")
            print(f"    Variación stability   : {v_stab:.6f}  →  peso = {w_stab_eff:.4f}")
            print(f"    Variación consistency : {v_cons:.6f}  →  peso = {w_cons_eff:.4f}")
            print(f"    Suma de pesos = {weights.sum():.4f}")
        else:
            # Normalizar pesos manuales a 1.0 (solo 2 métricas ahora)
            total = w_stability + w_consistency
            w_stab_eff = w_stability / total
            w_cons_eff = w_consistency / total
            print(f"\n[optimal_bw] PESOS MANUALES (modo sin teórico):")
            print(f"    w_stability={w_stab_eff:.4f}, w_consistency={w_cons_eff:.4f}")

        stab_norm = _norm01(stability)
        cons_norm = _inv_norm01(consistency_arr)
        for i in range(n):
            if not (np.isnan(stab_norm[i]) or np.isnan(cons_norm[i])):
                score[i] = w_stab_eff * stab_norm[i] + w_cons_eff * cons_norm[i]

    if np.any(~np.isnan(score)):
        optimal_idx = int(np.nanargmax(score))
    else:
        optimal_idx = n // 2
        warnings.warn("Ningún candidato produjo score válido. Usando candidato central.")

    # --- Plotting ---
    fig = None
    if plot:
        if theoretical is not None:
            # MODO TEÓRICO: 6 paneles (2 filas x 3 cols)
            if figsize is None:
                figsize = (18, 10)
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            axes = axes.ravel()

            # 0: Error L2 drift
            ax = axes[0]
            valid = ~np.isnan(error_drift_arr)
            ax.plot(bw_candidates[valid], error_drift_arr[valid], 'bo-', markersize=5, zorder=3)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel(r'$\|f_{\rm est} - f_{\rm teo}\|_{L^2}$')
            ax.set_title('Drift L2-error')
            ax.grid(True, alpha=0.3)

            # 1: Error L2 diffusion
            ax = axes[1]
            valid = ~np.isnan(error_diff_arr)
            ax.plot(bw_candidates[valid], error_diff_arr[valid], 'go-', markersize=5, zorder=3)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel(r'$\|D_{\rm est} - D_{\rm teo}\|_{L^2}$')
            ax.set_title('Diffusion L2-error')
            ax.grid(True, alpha=0.3)

            # 2: Norma L2 drift
            ax = axes[2]
            valid = ~np.isnan(norm_drift_arr)
            ax.plot(bw_candidates[valid], norm_drift_arr[valid], 'co-', markersize=5, zorder=3)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel(r'$\|f\|_{L^2}$')
            ax.set_title('Drift L2-Norm')
            ax.grid(True, alpha=0.3)

            # 3: Norma L2 diffusion
            ax = axes[3]
            valid = ~np.isnan(norm_diff_arr)
            ax.plot(bw_candidates[valid], norm_diff_arr[valid], 'mo-', markersize=5, zorder=3)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel(r'$\|D\|_{L^2}$')
            ax.set_title('Diffusion L2-Norm')
            ax.grid(True, alpha=0.3)

            # 4: Score
            ax = axes[4]
            valid = ~np.isnan(score)
            ax.plot(bw_candidates[valid], score[valid], 'm^-', markersize=7, zorder=3, linewidth=2)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2,
                       label=f'Score máx = {score[optimal_idx]:.3f}', zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel('Score')
            w_d_plot = w_drift_eff if auto_weight else w_drift
            w_D_plot = w_diff_eff if auto_weight else w_diff
            ax.set_title(f'Score = {w_d_plot:.2f}·err_drift + {w_D_plot:.2f}·err_diff')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # 5: Potencial óptimo
            ax = axes[5]
            U_opt = potentials[optimal_idx]
            edges_opt = edges_list[optimal_idx]
            if D == 2:
                X, Y = np.meshgrid(edges_opt[0], edges_opt[1], indexing='ij')
                U_plot = np.nan_to_num(U_opt, nan=np.nanmedian(U_opt))
                u_min, u_max = float(U_plot.min()), float(U_plot.max())
                if u_max - u_min < 1e-12:
                    ax.text(0.5, 0.5, f"Constant U = {u_min:.4g}",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=10, color="gray")
                else:
                    levels = np.linspace(u_min, u_max, 20)
                    cnt = ax.contourf(X, Y, U_plot, levels=levels, cmap='viridis')
                    plt.colorbar(cnt, ax=ax, label='U')
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            elif D == 1:
                ax.plot(edges_opt[0], U_opt, 'b-', label='Potencial óptimo')
                ax.set_xlabel('x')
                ax.set_ylabel('U')
            ax.set_title(f'Optimal potential (bw={bw_candidates[optimal_idx]:.4f})')

            for idx in range(6, len(axes)):
                axes[idx].axis('off')

        else:
            # MODO SIN TEÓRICO: 6 paneles (2 filas x 3 cols)
            if figsize is None:
                figsize = (18, 10)
            fig, axes = plt.subplots(2, 3, figsize=figsize)
            axes = axes.ravel()

            # 0: Norma L2 drift
            ax = axes[0]
            valid = ~np.isnan(norm_drift_arr)
            ax.plot(bw_candidates[valid], norm_drift_arr[valid], 'bo-', markersize=5, zorder=3)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel(r'$\|f\|_{L^2}$')
            ax.set_title('Drift L2-Norm ')
            ax.grid(True, alpha=0.3)

            # 1: Norma L2 diffusion
            ax = axes[1]
            valid = ~np.isnan(norm_diff_arr)
            ax.plot(bw_candidates[valid], norm_diff_arr[valid], 'go-', markersize=5, zorder=3)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel(r'$\|D\|_{L^2}$')
            ax.set_title('Diffusion L2-Norm ')
            ax.grid(True, alpha=0.3)

            # 2: Stability
            ax = axes[2]
            valid = ~np.isnan(stability)
            ax.plot(bw_candidates[valid], stability[valid], 'ro-', markersize=5, zorder=3)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel('Stability')
            ax.set_title('L2-Norm Stability')
            ax.grid(True, alpha=0.3)

            # 3: Consistency (valor crudo + normalizado)
            ax = axes[3]
            valid = ~np.isnan(consistency_arr)
            ax2 = ax.twinx()
            # Eje izquierdo: valor crudo adimensional
            ax.plot(bw_candidates[valid], consistency_arr[valid], 'ko--', markersize=4, alpha=0.5, label=r'$\eta_{\rm DB}$ raw')
            ax.set_ylabel(r'$\eta_{\rm DB}$ (dimensionless)', color='k')
            ax.tick_params(axis='y', labelcolor='k')
            # Eje derecho: versión normalizada [0,1] para el score
            cons_norm = _inv_norm01(consistency_arr)
            valid2 = ~np.isnan(cons_norm)
            ax2.plot(bw_candidates[valid2], cons_norm[valid2], 'yo-', markersize=5, zorder=3, label='Relative Consistency (norm.)')
            ax2.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
            ax2.set_ylabel('Consistency [0,1]', color='y')
            ax2.tick_params(axis='y', labelcolor='y')
            ax.set_xlabel('bw')
            ax.set_title(r'Relative Consistency $\eta_{\rm DB}$') #= \frac{\|f + D\nabla\log\rho\|_2}{\|f\|_2 + \|D\|_2}$')
            ax.grid(True, alpha=0.3)
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

            # 4: Score
            ax = axes[4]
            valid = ~np.isnan(score)
            ax.plot(bw_candidates[valid], score[valid], 'm^-', markersize=7, zorder=3, linewidth=2)
            ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2,
                       label=f'Score máx = {score[optimal_idx]:.3f}', zorder=4)
            ax.set_xlabel('bw')
            ax.set_ylabel('Hibrid Score')
            w_s_plot = w_stab_eff if auto_weight else w_stability
            w_c_plot = w_cons_eff if auto_weight else w_consistency
            ax.set_title(f'Score = {w_s_plot:.2f}·stab + {w_c_plot:.2f}·cons')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # 5: Potencial óptimo
            ax = axes[5]
            U_opt = potentials[optimal_idx]
            edges_opt = edges_list[optimal_idx]
            if D == 2:
                X, Y = np.meshgrid(edges_opt[0], edges_opt[1], indexing='ij')
                U_plot = np.nan_to_num(U_opt, nan=np.nanmedian(U_opt))
                u_min, u_max = float(U_plot.min()), float(U_plot.max())
                if u_max - u_min < 1e-12:
                    ax.text(0.5, 0.5, f"Constant U = {u_min:.4g}",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=10, color="gray")
                else:
                    levels = np.linspace(u_min, u_max, 20)
                    cnt = ax.contourf(X, Y, U_plot, levels=levels, cmap='viridis')
                    plt.colorbar(cnt, ax=ax, label='U')
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            elif D == 1:
                ax.plot(edges_opt[0], U_opt, 'b-', label='Optimal Potential')
                ax.set_xlabel('x')
                ax.set_ylabel('U')
            ax.set_title(f'Optimal potential (bw={bw_candidates[optimal_idx]:.4f})')

        fig.tight_layout()
        plt.show()

    return {
        'bw_candidates': bw_candidates,
        'potentials': potentials,
        'edges': edges,
        'norm_drift': norm_drift_arr,
        'norm_diff': norm_diff_arr,
        'error_drift': error_drift_arr,
        'error_diff': error_diff_arr,
        'consistency': consistency_arr,
        'stability': stability,
        'score': score,
        'optimal_bw': float(bw_candidates[optimal_idx]),
        'optimal_idx': int(optimal_idx),
        'fig': fig
    }

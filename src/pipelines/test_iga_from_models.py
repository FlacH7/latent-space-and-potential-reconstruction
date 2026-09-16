"""
test_iga_pipeline.py
====================
Pipeline de validación para km_tools_v2.py + iga_reconstructor.py.

Este script demuestra la reconstrucción de potencial via Galerkin-B-spline
con inversión robusta SVD y descomposición de Helmholtz explícita.

Nuevas métricas reportadas:
  - eta (η): métrica de no-equilibrio (0 = equilibrio perfecto)
  - rank_D: rango numérico efectivo de D en cada celda
  - condition_D: número de condición efectivo de D
  - is_equilibrium: True si η < 0.1
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.potential_reconstruction.km_tools_v2 import (
    simulate_data,
    extract_km_coefficients,
    plot_km_components,
    reconstruct_potential,
    reconstruct_potential_1D,
    plot_potential,
    plot_potential_2d,
    plot_nonconservative_force_2d,
    plot_potential_combined_2d,
    # optimal_bw
)
from src.potential_reconstruction.bw_optimization import optimal_bw
from pathlib import Path
from src.latent_space_extraction.data_analysis_tools import outliers_cleaning, augment_samples

from src.plotters.trajectory_plots import plot_latent_trajectory

# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
from src.utils.config import (
    CONFIGS,
    DT,
    SEED,
    TIME,
    BASE_RESULTS_PATH
)

# ---------------------------------------------------------------------------
# MODELS CONFIG
# ---------------------------------------------------------------------------
configs = CONFIGS
dt = DT
seed = SEED
T = TIME

# ================================================================
# ELEGIR MODELO
# ================================================================
models = ['single_well', 'asymmetric_double_well', 'ou', 'double_well', 'multi_stable', 'triple_well_3d', 'ring_attractor', 'stochastic_oscillator']
model_name = models[6]  

config = configs[model_name]
D = config['D']

out_dir = Path(BASE_RESULTS_PATH + f"/{model_name}_d{D}")
out_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print(f"PIPELINE IGA: {model_name.upper()}  |  D={D}  |  T={T}  |  dt={dt}")
print(f"Reconstrucción: Galerkin-B-spline (degree={config['degree']})")
print("="*70)

# ================================================================
# 1. SIMULAR DATOS
# ================================================================
result = simulate_data(config, cache_dir='sim_cache')
data = result['data']
print(data.shape)
theoretical = result['theoretical']
model_info = result['model_info']

print(f"\n[1] Datos simulados: {data.shape} (N={data.shape[0]}, D={D})")
print(f"    Modelo: {model_info['name']}")

plot_latent_trajectory(
            data,
            out_dir=out_dir,
            method_name=f"simulated_{model_info['name']}_D{D}",
        )

#==================================================================
# 1.5 Limpiar outliers y plotear datos simulados
#==================================================================
# Optional: plot the latent time series
fig_ts, axes = plt.subplots(D, 1, figsize=(14, 2.5 * D), squeeze=False)
for d in range(D):
    ax = axes[d, 0]
    ax.plot(data[:, d], lw=0.5)
    ax.set_title(f"Latent dimension {d}")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
plt.tight_layout()
fig_ts.savefig(out_dir / "latent_timeseries.png", dpi=150)
plt.close(fig_ts)
print(f"  Saved latent_timeseries.png")

data = outliers_cleaning(data, method='iqr', threshold=4)

# Optional: plot the latent time series
fig_ts, axes = plt.subplots(D, 1, figsize=(14, 2.5 * D), squeeze=False)
for d in range(D):
    ax = axes[d, 0]
    ax.plot(data[:, d], lw=0.5)
    ax.set_title(f"Latent dimension {d} (after outlier cleaning)")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
plt.tight_layout()
fig_ts.savefig(out_dir / "latent_timeseries_cleaned.png", dpi=150)
plt.close(fig_ts)
print(f"  Saved latent_timeseries_cleaned.png")

### Resampleo de datos
# data = augment_samples(data, n_target=data.shape[0]*5, method='block_bootstrap')

# ================================================================
# 2. BW ÓPTIMO (usa IgA internamente)
# ================================================================
# teorico = theoretical
teorico = None
# if not (D==3 and model_name == 'triple_well_3d'):
#     result_bw = optimal_bw(
#         data=data,
#         bins=config['bins'],
#         dt=dt,
#         p=2,
#         kernel='epanechnikov',
#         # kernel = 'gaussian',
#         theoretical = teorico,
#         sigma_smooth=1.0,
#         n_candidates=20,
#         n_jobs=-1 if D+config['degree'] <= 4 else 1,  # Paralelizar solo para D=1 o D=2 (evitar overhead en D>=3)
#         # n_jobs = 3,
#         plot=True,
#         auto_weight = True,
#         # interval = [0.05, 0.5]
#     )
    
#     # Save BW plot
#     if "fig" in result_bw:
#         result_bw["fig"].savefig(out_dir / "bw_optimisation.png", dpi=150)
#         plt.close(result_bw["fig"])
#     bw_opt = result_bw['optimal_bw']
#     if teorico is not None:
#         print(f"\n[2] bw óptimo (modo teórico): {bw_opt:.4f}")
#     else:
#         print(f"\n[2] bw óptimo (modo sin teórico): {bw_opt:.4f}")
#     print(f"    Error drift: {result_bw['error_drift'][result_bw['optimal_idx']]:.4f}, "
#         f"Error diff: {result_bw['error_diff'][result_bw['optimal_idx']]:.4f}")

# ================================================================
# 3. ESTIMAR COEFICIENTES KM
# ================================================================
if D==3 and model_name == 'triple_well_3d':
    bandwidth = 0.3544
else:
    # bandwidth = bw_opt
    bandwidth = 0.3
    
drift, diffusion, edges = extract_km_coefficients(
    data,
    bins=config['bins'],
    p=2,
    bw=bandwidth,
    # bw = 0.25,
    kernel='epanechnikov',
    # kernel = 'gaussian',
    dt=dt,
    sigma_smooth = 0.5,
    density_threshold = 0.005
)

print(f"\n[3] Coeficientes estimados:")
print(f"    Drift shape: {drift.shape}")
print(f"    Diffusion shape: {diffusion.shape}")
print(f"    Bins: {[len(e) for e in edges]}")

# ================================================================
# 4. DENSIDAD EMPÍRICA (requerida por IgA)
# ================================================================
hist_edges = []
for d in range(D):
    c = edges[d]
    if len(c) > 1:
        half = (c[1] - c[0]) / 2.0
    else:
        half = 0.5
    e = np.concatenate([[c[0] - half], c + half])
    hist_edges.append(e)

density, _ = np.histogramdd(data, bins=hist_edges)
density = density.astype(float)
print(f"\n[4] Densidad empírica shape: {density.shape}")

# ================================================================
# 5. PLOT D^1 y D^2 CON TEÓRICOS
# ================================================================
figs_km = plot_km_components(
    drift, diffusion, edges,
    drift_components=config['drift_components'],
    diff_components=config['diff_components'],
    fixed_coords=None,
    theoretical=theoretical,
    figsize=(12, 8)
)
for key, fname in [("drift", "km_components_drift.png"),
                   ("diffusion", "km_components_diffusion.png")]:
    fig_km = figs_km.get(key)
    if fig_km is None:
        continue
    fig_km.suptitle("...", fontsize=14)
    fig_km.tight_layout(rect=[0, 0, 1, 0.95])
    fig_km.savefig(out_dir / fname, dpi=150)
    plt.close(fig_km)
print(f"  Saved km components figures: km_components_drift.png, km_components_diffusion.png")

# ================================================================
# 6. RECONSTRUIR POTENCIAL 1D (cortes via IgA)
# ================================================================
U_rec_1d = reconstruct_potential_1D(
    drift, diffusion, edges,
    density=density,
    degree=config['degree']
)

fig_pot_1d = plot_potential(
    U_rec_1d, edges,
    theoretical=theoretical,
    component_labels=[f'x{j+1}' for j in range(D)],
    figsize=(5*D, 4)
)
plt.suptitle(f"{model_name.upper()} — Potencial 1D reconstruido vs teórico (IgA)")
plt.tight_layout()
fig_pot_1d.savefig(out_dir / "potential_1d.png", dpi=150)
plt.close(fig_pot_1d)
print(f"  Saved potential_1d.png")

# ================================================================
# 7. RECONSTRUIR POTENCIAL MULTIDIMENSIONAL (IgA completo)
# ================================================================
result_iga = reconstruct_potential(
    drift, diffusion, edges,
    density=density,
    method='iga',
    return_full=True,
    degree=config['degree'],
    decompose_helmholtz=True,
    compute_stream_function=True,  # NUEVO: ψ, g_recon = ∇U + ∇⊥ψ y v = f + D·∇U
    density_threshold=0.01,
    rtol=1e-6,
    atol=1e-12
)

U_rec = result_iga['potential']
print(f"\n[7] Potencial reconstruido shape: {U_rec.shape}")
print(f"    Método: Galerkin-B-spline (degree={config['degree']})")

# Métricas de calidad robusta
rank_D = result_iga['rank_D']
condition_D = result_iga['condition_D']
frac_low_rank = np.mean(rank_D < D) if np.any(~np.isnan(rank_D)) else 0.0
print(f"    Fracción celdas con rank_D < {D}: {frac_low_rank:.2%}")
print(f"    Condición D (mediana): {np.nanmedian(condition_D):.2e}")
print(f"    Condición D (max): {np.nanmax(condition_D):.2e}")

# Métricas de Helmholtz
if 'eta' in result_iga:
    eta = result_iga['eta']
    is_eq = result_iga['is_equilibrium']
    print(f"    Métrica no-equilibrio η: {eta:.4f}")
    print(f"    ¿Equilibrio detallado? {is_eq} (umbral η < 0.1)")

# Métricas vs teórico (solo para validación)
if not np.all(np.isnan(U_rec)):
    U_theo_grid = theoretical['potential_grid'](edges)
    valid = ~np.isnan(U_rec)
    if np.any(valid):
        U_rec_norm = (U_rec[valid] - np.nanmean(U_rec)) / np.nanstd(U_rec[valid])
        U_theo_norm = (U_theo_grid[valid] - np.nanmean(U_theo_grid)) / np.nanstd(U_theo_grid)
        rmse_norm = np.sqrt(np.mean((U_rec_norm - U_theo_norm)**2))
        from scipy.stats import pearsonr
        corr, _ = pearsonr(U_rec[valid], U_theo_grid[valid])
        print(f"    Correlación forma (Pearson): {corr:.4f}")
        print(f"    RMSE normalizado: {rmse_norm:.4f}")

# ================================================================
# 8. PLOT POTENCIAL 2D / SLICES
# ================================================================
if D == 2:
    U_theo_grid = theoretical['potential_grid'](edges)
    fig_pot_2d = plot_potential_2d(
        U_rec, edges,
        U_theoretical=U_theo_grid,
        title_est=f"{model_name.upper()} — Reconstructed",
        title_theo=f"{model_name.upper()} — Theoretical",
        figsize=(12, 5),
        unify_colorbar=False,
        align_minima=False,
        align_to_zero=False,
        crop_to_valid = True,
        show_streamlines=False,
        stream_function=result_iga.get('stream_function'),        # NUEVO
        reconstructed_field=result_iga.get('reconstructed_field'),# NUEVO
        reconstructed_drift=result_iga.get('reconstructed_drift'),  # NUEVO
        clip_percentile = 99.9,
    )
    if fig_pot_2d is None:
            fig_pot_2d = plt.gcf()
    fig_pot_2d.savefig(out_dir / "potential_2d.png", dpi=150)
    plt.close(fig_pot_2d)
    print("  Saved potential_2d.png")

    # Plot adicional: residuo de Helmholtz ||r||
    if 'residual' in result_iga:
        residual = result_iga['residual']
        r_norm = np.sqrt(np.nansum(residual**2, axis=0))
        x_c, y_c = edges
        X, Y = np.meshgrid(x_c, y_c, indexing='ij')
        fig_res = plt.figure(figsize=(6, 5))
        plt.contourf(X, Y, np.nan_to_num(r_norm, nan=0), levels=20, cmap='magma')
        plt.colorbar(label='||r||')
        plt.xlabel('x'); plt.ylabel('y')
        plt.title(f'Residuo Helmholtz (no-equilibrio) — {model_name.upper()}')
        plt.axis('equal')
        plt.tight_layout()
        fig_res.savefig(out_dir / "potential_2d_residual.png", dpi=150)
        plt.close(fig_res)
        print("  Saved potential_2d_residual.png")

    # ================================================================
    # 8b. NUEVOS PLOTS: fuerza no-conservativa v y combinación U + v + ψ
    # (solo cuando la stream function está disponible, D == 2)
    # ================================================================
    if result_iga.get('nonconservative_force') is not None:
        fig_v = plot_nonconservative_force_2d(
            result_iga['nonconservative_force'], edges,
            title=f'Non-conservative force — {model_name.upper()}',
            crop_to_valid=True,
            skip = 4
        )
        fig_v.savefig(out_dir / "potential_2d_nonconservative_force.png", dpi=150)
        plt.close(fig_v)
        print("  Saved potential_2d_nonconservative_force.png")

    if (result_iga.get('reconstructed_field') is not None
            or result_iga.get('nonconservative_force') is not None):
        fig_comb = plot_potential_combined_2d(
            U_rec, edges,
            stream_function=result_iga.get('stream_function'),
            nonconservative_force=result_iga.get('nonconservative_force'),
            reconstructed_field=result_iga.get('reconstructed_field'),
            reconstructed_drift=result_iga.get('reconstructed_drift'),
            title=f'{model_name.upper()} — U (background) + v (red arrows) + streamlines (white arrows)',
            crop_to_valid=True,
            clip_percentile=99.9,
            skip=4
        )
        fig_comb.savefig(out_dir / "potential_2d_combined.png", dpi=150)
        plt.close(fig_comb)
        print("  Saved potential_2d_combined.png")

elif D >= 3:
    # Para D>=3, plotear cortes 2D seleccionando pares de dimensiones
    from src.potential_reconstruction.km_tools_v2 import plot_potential_slice
    U_theo_grid = theoretical['potential_grid'](edges)

    # Plot todos los pares de las primeras 3 dimensiones
    dim_pairs = [(i, j) for i in range(min(D, 3)) for j in range(i+1, min(D, 3))]
    for dims in dim_pairs:
        fig_slice = plot_potential_slice(
            U_rec, edges, dims=dims,
            fixed_coords={d: 0.0 for d in range(D) if d not in dims},
            U_theoretical=U_theo_grid,
            title=f"{model_name.upper()} — Reconstructed (slice x{dims[0]+1}-x{dims[1]+1})",
            figsize=(12, 5),
            unify_colorbar=False,
            align_minima=False,
            align_to_zero=False,
            stream_function=result_iga.get('stream_function'),        # NUEVO
            reconstructed_field=result_iga.get('reconstructed_field'),# NUEVO
            reconstructed_drift=result_iga.get('reconstructed_drift'),# NUEVO
        )
        if fig_slice is None:
                fig_slice = plt.gcf()
        fname = f"potential_slice_x{dims[0]+1}_x{dims[1]+1}.png"
        fig_slice.savefig(out_dir / fname, dpi=150)
        plt.close(fig_slice)
        print(f"  Saved {fname}")

print("\n" + "="*70)
print("PIPELINE IGA COMPLETADO EXITOSAMENTE")
print("="*70)

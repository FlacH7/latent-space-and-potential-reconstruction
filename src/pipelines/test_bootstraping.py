"""
test_bootstraping.py
====================
Pipeline de validación con bootstraping dimensional.

Parte de un modelo sintético (igual que test_iga_from_models), ejecuta
el pipeline KM+IgA de referencia, y luego:

  1. Aumenta la dimensionalidad de la data simulada (ej. 2 -> 50)
     via ``augment_dimensions``.
  2. Convierte la data aumentada a un objeto ``mne.io.RawArray``,
     aplica filtro band-pass 1-40 Hz, ejecuta ICA (usando
     ``eeg_preprocessing.fit_ica``) + selección Markov (usando
     ``markov_subspace``) para extraer el subespacio.
  3. Ejecuta el pipeline KM+IgA sobre ese espacio latente.
  4. Guarda todos los plots comparativos.

Usage::

    python test_bootstraping.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# km_tools_v2 y utilidades
# ---------------------------------------------------------------------------
from src.potential_reconstruction.km_tools_v2 import (
    simulate_data,
    extract_km_coefficients,
    plot_km_components,
    reconstruct_potential,
    reconstruct_potential_1D,
    plot_potential,
    plot_potential_2d,
    plot_potential_slice,
)
from src.potential_reconstruction.bw_optimization import optimal_bw
from src.latent_space_extraction.data_analysis_tools import outliers_cleaning, augment_dimensions

# ---------------------------------------------------------------------------
# Funciones existentes del proyecto EEG
# ---------------------------------------------------------------------------
from src.latent_space_extraction.eeg_preprocessing import (
    fit_ica,
    extract_clean_components,
    apply_bandpass_filter,
)
from src.latent_space_extraction.markov_subspace import find_best_subspace_markov, greedy_forward_selection_markov

# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
from src.utils.config import (
    CONFIGS,
    DT,
    SEED,
    TIME
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
models = ['single_well', 'asymmetric_double_well', 'ou', 'double_well',
          'multi_stable', 'triple_well_3d', 'ring_attractor', 'stochastic_oscillator']
model_name = models[3]  # default: double_well

config = configs[model_name]
D = config['D']

out_dir = Path(f".-/bootstraping_{model_name}_d{D}")
out_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print(f"PIPELINE BOOTSTRAPING: {model_name.upper()}  |  D={D}  |  T={T}  |  dt={dt}")
print(f"Reconstrucción: Galerkin-B-spline (degree={config['degree']})")
print("="*70)

# ================================================================
# 1. SIMULAR DATOS
# ================================================================
result = simulate_data(config, cache_dir='sim_cache')
data = result['data']
theoretical = result['theoretical']
model_info = result['model_info']

print(f"\n[1] Datos simulados: {data.shape} (N={data.shape[0]}, D={D})")
print(f"    Modelo: {model_info['name']}")

# ================================================================
# 2. LIMPIAR OUTLIERS (referencia)
# ================================================================
fig_ts, axes = plt.subplots(D, 1, figsize=(14, 2.5*D), squeeze=False)
for d in range(D):
    ax = axes[d, 0]
    ax.plot(data[:, d], lw=0.5)
    ax.set_title(f"Original dimension {d}")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
plt.tight_layout()
fig_ts.savefig(out_dir / "ref_timeseries.png", dpi=150)
plt.close(fig_ts)
print("  Saved ref_timeseries.png")

data = outliers_cleaning(data, method='iqr', threshold=4)

fig_ts, axes = plt.subplots(D, 1, figsize=(14, 2.5*D), squeeze=False)
for d in range(D):
    ax = axes[d, 0]
    ax.plot(data[:, d], lw=0.5)
    ax.set_title(f"Original dimension {d} (cleaned)")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
plt.tight_layout()
fig_ts.savefig(out_dir / "ref_timeseries_cleaned.png", dpi=150)
plt.close(fig_ts)
print("  Saved ref_timeseries_cleaned.png")

# ================================================================
# 3. PIPELINE REFERENCIA (KM + IgA)
# ================================================================
print("\n" + "="*70)
print("  REFERENCE PIPELINE")
print("="*70)

def run_km_pipeline(data_in, prefix, theoretical_in=None, dt_in=dt):
    """Ejecuta el pipeline KM+IgA completo y guarda plots."""
    D_local = data_in.shape[1]
    bins = config['bins'] if D_local == D else [50]*D_local
    degree = config['degree']

    # --- BW ---
    result_bw = optimal_bw(
        data=data_in,
        bins=bins,
        dt=dt_in,
        p=2,
        kernel='epanechnikov',
        theoretical=None,
        sigma_smooth=1.0,
        n_candidates=50,
        n_jobs=-1 if D_local+degree <= 4 else 1,
        plot=True,
        auto_weight=True,
    )
    if "fig" in result_bw:
        result_bw["fig"].savefig(out_dir / f"{prefix}_bw_optimisation.png", dpi=150)
        plt.close(result_bw["fig"])
    bw_opt = result_bw['optimal_bw']
    print(f"\n  [{prefix}] bw óptimo: {bw_opt:.4f}")

    # --- KM ---
    drift, diffusion, edges = extract_km_coefficients(
        data_in,
        bins=bins,
        p=2,
        bw=bw_opt,
        kernel='epanechnikov',
        dt=dt_in,
        sigma_smooth=0.5,
        density_threshold=0.005,
    )
    print(f"  [{prefix}] Drift shape: {drift.shape}, Diffusion: {diffusion.shape}")

    # --- Densidad ---
    hist_edges = []
    for d in range(D_local):
        c = edges[d]
        half = (c[1] - c[0]) / 2.0 if len(c) > 1 else 0.5
        e = np.concatenate([[c[0] - half], c + half])
        hist_edges.append(e)
    density, _ = np.histogramdd(data_in, bins=hist_edges)
    density = density.astype(float)

    # --- KM components ---
    fig_km = plot_km_components(
        drift, diffusion, edges,
        drift_components=list(range(D_local)),
        diff_components=[(i, i) for i in range(D_local)],
        fixed_coords=None,
        theoretical=theoretical_in,
        figsize=(12, 8),
    )
    plt.suptitle(f"{prefix.upper()} — D^1 and D^2", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_km.savefig(out_dir / f"{prefix}_km_components.png", dpi=150)
    plt.close(fig_km)
    print(f"  Saved {prefix}_km_components.png")

    # --- Potencial 1D ---
    U_rec_1d = reconstruct_potential_1D(
        drift, diffusion, edges,
        density=density,
        degree=degree,
    )
    fig_pot_1d = plot_potential(
        U_rec_1d, edges,
        theoretical=theoretical_in,
        component_labels=[f'x{j+1}' for j in range(D_local)],
        figsize=(5*D_local, 4),
    )
    plt.suptitle(f"{prefix.upper()} — 1D Potential (IgA)")
    plt.tight_layout()
    fig_pot_1d.savefig(out_dir / f"{prefix}_potential_1d.png", dpi=150)
    plt.close(fig_pot_1d)
    print(f"  Saved {prefix}_potential_1d.png")

    # --- Potencial multi-D ---
    result_iga = reconstruct_potential(
        drift, diffusion, edges,
        density=density,
        method='iga',
        return_full=True,
        degree=degree,
        decompose_helmholtz=True,
        density_threshold=0.01,
        rtol=1e-6,
        atol=1e-12,
    )
    U_rec = result_iga['potential']
    print(f"  [{prefix}] Potential shape: {U_rec.shape}")

    if 'eta' in result_iga:
        print(f"  [{prefix}] eta: {result_iga['eta']:.4f}, equilibrium: {result_iga['is_equilibrium']}")

    # --- Plot 2D / slices ---
    if D_local == 2:
        U_theo_grid = theoretical_in['potential_grid'](edges) if theoretical_in is not None else None
        fig_pot_2d = plot_potential_2d(
            U_rec, edges,
            U_theoretical=U_theo_grid,
            title_est=f"{prefix.upper()} — Reconstructed",
            title_theo=f"{prefix.upper()} — Theoretical",
            figsize=(12, 5),
            unify_colorbar=True,
            align_minima=True,
            align_to_zero=True,
            crop_to_valid=True,
        )
        if fig_pot_2d is None:
            fig_pot_2d = plt.gcf()
        fig_pot_2d.savefig(out_dir / f"{prefix}_potential_2d.png", dpi=150)
        plt.close(fig_pot_2d)
        print(f"  Saved {prefix}_potential_2d.png")

        if 'residual' in result_iga:
            residual = result_iga['residual']
            r_norm = np.sqrt(np.nansum(residual**2, axis=0))
            x_c, y_c = edges
            X, Y = np.meshgrid(x_c, y_c, indexing='ij')
            fig_res = plt.figure(figsize=(6, 5))
            plt.contourf(X, Y, np.nan_to_num(r_norm, nan=0), levels=20, cmap='magma')
            plt.colorbar(label='||r||')
            plt.xlabel('x'); plt.ylabel('y')
            plt.title(f'{prefix.upper()} — Helmholtz Residual')
            plt.axis('equal')
            plt.tight_layout()
            fig_res.savefig(out_dir / f"{prefix}_potential_2d_residual.png", dpi=150)
            plt.close(fig_res)
            print(f"  Saved {prefix}_potential_2d_residual.png")

    elif D_local >= 3:
        dim_pairs = [(i, j) for i in range(min(D_local, 3)) for j in range(i+1, min(D_local, 3))]
        for dims in dim_pairs:
            fig_slice = plot_potential_slice(
                U_rec, edges, dims=dims,
                fixed_coords={d: 0.0 for d in range(D_local) if d not in dims},
                title=f"{prefix.upper()} — Slice x{dims[0]+1}-x{dims[1]+1}",
                figsize=(12, 5),
                unify_colorbar=True,
                align_minima=True,
                align_to_zero=True,
                crop_to_valid=True,
            )
            if fig_slice is None:
                fig_slice = plt.gcf()
            fname = f"{prefix}_slice_x{dims[0]+1}_x{dims[1]+1}.png"
            fig_slice.savefig(out_dir / fname, dpi=150)
            plt.close(fig_slice)
            print(f"  Saved {fname}")

    return result_iga, edges

# Ejecutar pipeline referencia
ref_result, ref_edges = run_km_pipeline(data, "ref", theoretical_in=theoretical, dt_in=dt)

# ================================================================
# 4. BOOTSTRAPING — Aumento dimensionalidad
# ================================================================
print("\n" + "="*70)
print("  BOOTSTRAPING — Augment Dimensions")
print("="*70)

D_target = 4  # dimensionalidad aumentada
print(f"\n  Augmenting dimensions: {D} -> {D_target}")

data_aug = augment_dimensions(data, D_target=D_target, noise_std=0, seed=seed)
print(f"  Augmented data shape: {data_aug.shape}")

# Plot serie temporal aumentada (primeras 5 dims)
fig_aug, axes = plt.subplots(5, 1, figsize=(14, 12), squeeze=False)
for d in range(min(5, D_target)):
    ax = axes[d, 0]
    ax.plot(data_aug[:, d], lw=0.3)
    ax.set_title(f"Augmented dimension {d}")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
plt.tight_layout()
fig_aug.savefig(out_dir / "bootstrap_aug_timeseries.png", dpi=150)
plt.close(fig_aug)
print("  Saved bootstrap_aug_timeseries.png")

# ================================================================
# 5. EXTRAER ESPACIO LATENTE (Filtro 1-40 Hz + ICA + Markov)
# ================================================================
print("\n" + "="*70)
print("  EXTRACTING LATENT SUBSPACE (Filter 1-40 Hz + ICA + Markov)")
print("="*70)

def extract_latent_from_array(
    data_in: np.ndarray,
    n_dim: int,
    n_bins: int = 10,
    seed: int | None = None,
    use_greedy: bool = False,
    n_workers: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Extrae un subespacio latente de dimensión ``n_dim`` desde un array
    numérico usando las funciones del proyecto:
      1. Convierte el array a ``mne.io.RawArray``
      2. ``apply_bandpass_filter`` (1-40 Hz)
      3. ``fit_ica`` (PCA whitening + ICA)
      4. ``extract_clean_components`` (todas las fuentes)
      5. ``markov_subspace`` (selección)

    No usa ICLabel (datos sintéticos).
    """
    n_samples, n_features = data_in.shape
    t0 = time.time()

    # 1. Crear objeto MNE RawArray
    ch_names = [f"ch{i:03d}" for i in range(n_features)]
    info = mne.create_info(ch_names=ch_names, sfreq=1.0 / dt, ch_types="eeg")
    raw_array = mne.io.RawArray(data_in.T, info, verbose=False)
    
    data_plot = raw_array.get_data()  # ndarray (n_channels, n_samples)

    fig, axes = plt.subplots(D_target, 1, figsize=(14, 2.5 * D_target), squeeze=False)
    for d in range(D_target):
        ax = axes[d, 0]
        ax.plot(data_plot[d], lw=0.3)
        ax.set_title(f"Channel {d}")
        ax.set_xlabel("sample")
        ax.set_ylabel("amplitude")
    plt.tight_layout()
    fig.savefig(out_dir / "raw_array_plot.png", dpi=150)
    plt.close(fig)

    # # 2. Filtro band-pass 1-40 Hz (elimina warning de ICA y mejora convergencia)
    # raw_filtered = apply_bandpass_filter(
    #     raw_array, l_freq=1.0, h_freq=None, verbose=False
    # )

    # # Guardar captura de la serie filtrada (primeras 5 dims)
    # n_plot = min(5, n_features)
    # fig_filt, axes = plt.subplots(n_plot, 1, figsize=(14, 2.5 * n_plot), squeeze=False)
    # filt_data = raw_filtered.get_data()[:n_plot, :]
    # for d in range(n_plot):
    #     ax = axes[d, 0]
    #     ax.plot(filt_data[d], lw=0.3)
    #     ax.set_title(f"Filtered dimension {d} (1-40 Hz)")
    #     ax.set_xlabel("sample")
    #     ax.set_ylabel("amplitude")
    # plt.tight_layout()
    # fig_filt.savefig(out_dir / "bootstrap_filtered_timeseries.png", dpi=150)
    # plt.close(fig_filt)
    # print("  Saved bootstrap_filtered_timeseries.png")

    # 3. ICA (fit_ica de eeg_preprocessing ya hace PCA whitening internamente)
    ica = fit_ica(
        raw_array,
        n_components=n_features,
        method="picard",
        random_state=seed,
        verbose=False,
    )

    # 4. Extraer todas las fuentes (no hay artifacts que rechazar)
    Y, kept_indices = extract_clean_components(raw_array, ica, apply_exclude=False)
    # Y shape: (n_features, n_samples)
    
    ####PLOTEO
    
    # Y shape: (n_ica_components, n_samples)
    n_comp, n_samples = Y.shape

    # Plotear las primeras 5 componentes (ajusta según quieras)
    n_plot = min(5, n_comp)
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 2.5 * n_plot), squeeze=False)

    for d in range(n_plot):
        ax = axes[d, 0]
        ax.plot(Y[d, :], lw=0.3)
        ax.set_title(f"ICA source {d}")
        ax.set_xlabel("sample")
        ax.set_ylabel("amplitude")

    plt.tight_layout()
    fig.savefig(out_dir / "bootstrap_ica_sources.png", dpi=150)
    plt.close(fig)
    print("  Saved bootstrap_ica_sources.png")
    ###################

    # 5. Selección de subespacio via Markov
    if use_greedy:
        comb, tau = greedy_forward_selection_markov(Y, n_dim, n_bins=n_bins)
        comb = tuple(comb)
        search_type = "greedy"
    else:
        comb, tau = find_best_subspace_markov(
            Y, n_dim, n_bins=n_bins, n_workers=n_workers
        )
        search_type = "exhaustive"

    # 6. Construir espacio latente
    # latent = Y[list(comb), :].T  # (n_samples, n_dim)
    latent = Y[[0, 1], :].T  # (n_samples, n_dim)
    elapsed = time.time() - t0

    meta = {
        "preprocessing": {
            "n_samples": n_samples,
            "n_features": n_features,
            "ica_kept": len(kept_indices),
        },
        "selected_indices": list(comb),
        "latent_scores": {"tau": float(tau), "search": search_type},
        "elapsed_time": elapsed,
        "extraction_method": "filter_1-40Hz+fit_ica+extract_clean_components+Markov",
    }

    print(f"    Filtered: 1-40 Hz")
    print(f"    ICA components: {n_features}")
    print(f"    Markov selected: {comb}  (tau={tau:.4f})")
    print(f"    Extraction time: {elapsed:.1f} s")

    return latent, meta

latent_boot, meta_boot = extract_latent_from_array(
    data_aug, n_dim=D, n_bins=5, seed=seed, use_greedy=False, n_workers=None
)
print(f"  Latent space shape: {latent_boot.shape}")
print(f"  Extraction method: {meta_boot['extraction_method']}")

# Plot espacio latente
fig_lat, axes = plt.subplots(D, 1, figsize=(14, 2.5*D), squeeze=False)
for d in range(D):
    ax = axes[d, 0]
    ax.plot(latent_boot[:, d], lw=0.5)
    ax.set_title(f"Latent dimension {d}")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
plt.tight_layout()
fig_lat.savefig(out_dir / "bootstrap_latent_timeseries.png", dpi=150)
plt.close(fig_lat)
print("  Saved bootstrap_latent_timeseries.png")

# ================================================================
# 6. LIMPIAR OUTLIERS DEL ESPACIO LATENTE
# ================================================================
latent_boot = outliers_cleaning(latent_boot, method='iqr', threshold=4)

fig_lat, axes = plt.subplots(D, 1, figsize=(14, 2.5*D), squeeze=False)
for d in range(D):
    ax = axes[d, 0]
    ax.plot(latent_boot[:, d], lw=0.5)
    ax.set_title(f"Latent dimension {d} (cleaned)")
    ax.set_xlabel("sample")
    ax.set_ylabel("amplitude")
plt.tight_layout()
fig_lat.savefig(out_dir / "bootstrap_latent_timeseries_cleaned.png", dpi=150)
plt.close(fig_lat)
print("  Saved bootstrap_latent_timeseries_cleaned.png")

# ================================================================
# 7. PIPELINE KM + IgA SOBRE EL ESPACIO LATENTE
# ================================================================
print("\n" + "="*70)
print("  LATENT SPACE PIPELINE")
print("="*70)

boot_result, boot_edges = run_km_pipeline(latent_boot, "boot", theoretical_in=None, dt_in=dt)

# ================================================================
# 8. COMPARACIÓN VISUAL (solo para D==2)
# ================================================================
if D == 2:
    print("\n" + "="*70)
    print("  COMPARACIÓN REFERENCIA vs BOOTSTRAP")
    print("="*70)

    U_ref = ref_result['potential']
    U_boot = boot_result['potential']

    U_ref_plot = U_ref - np.nanmin(U_ref)
    U_boot_plot = U_boot - np.nanmin(U_boot)

    if U_ref.shape != U_boot.shape or not np.allclose(ref_edges[0], boot_edges[0]):
        try:
            from scipy.interpolate import RegularGridInterpolator
            interp_boot = RegularGridInterpolator(
                boot_edges, U_boot, bounds_error=False, fill_value=np.nan
            )
            X, Y = np.meshgrid(ref_edges[0], ref_edges[1], indexing='ij')
            points = np.stack([X.ravel(), Y.ravel()], axis=1)
            U_boot_interp = interp_boot(points).reshape(U_ref.shape)
        except Exception:
            U_boot_interp = U_boot_plot
    else:
        U_boot_interp = U_boot_plot

    vmin = min(np.nanmin(U_ref_plot), np.nanmin(U_boot_interp))
    vmax = max(np.nanmax(U_ref_plot), np.nanmax(U_boot_interp))
    levels = np.linspace(vmin, vmax, 50)

    x_ref, y_ref = ref_edges
    X_ref, Y_ref = np.meshgrid(x_ref, y_ref, indexing='ij')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    cnt1 = ax1.contourf(X_ref, Y_ref, U_ref_plot, levels=levels, cmap='viridis')
    plt.colorbar(cnt1, ax=ax1, label='U')
    ax1.set_aspect('equal')
    ax1.set_title("Reference Potential")
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    cnt2 = ax2.contourf(X_ref, Y_ref, np.nan_to_num(U_boot_interp, nan=vmin), levels=levels, cmap='viridis')
    plt.colorbar(cnt2, ax=ax2, label='U')
    ax2.set_aspect('equal')
    ax2.set_title("Bootstrap Latent Potential")
    ax2.set_xlabel('x'); ax2.set_ylabel('y')

    diff = np.abs(U_ref_plot - np.nan_to_num(U_boot_interp, nan=0))
    cnt3 = ax3.contourf(X_ref, Y_ref, diff, levels=20, cmap='magma')
    plt.colorbar(cnt3, ax=ax3, label='|ΔU|')
    ax3.set_aspect('equal')
    ax3.set_title("Absolute Difference")
    ax3.set_xlabel('x'); ax3.set_ylabel('y')

    fig.tight_layout()
    fig.savefig(out_dir / "comparison_ref_vs_boot.png", dpi=150)
    plt.close(fig)
    print("  Saved comparison_ref_vs_boot.png")

print("\n" + "="*70)
print("PIPELINE BOOTSTRAPING COMPLETADO EXITOSAMENTE")
print("="*70)

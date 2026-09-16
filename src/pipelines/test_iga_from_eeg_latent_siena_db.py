"""
test_iga_from_eeg_latent.py
===========================
Pipeline de Kramers-Moyal + IgA reconstruction a partir de un espacio
latente EEG extraido via ``extract_latent_space``.

Este script replica la estructura de ``test_iga_epileptor.py`` pero en
lugar de cargar datos desde un fichero ``.jld2``, obtiene la serie
temporal del subespacio latente de un EEG usando el pipeline completo:
    raw EEG → ICA → ICLabel → seleccion de subespacio (Markov/FC/etc.)

La serie latente se pasa luego al estimador Kramers-Moyal y al
reconstructor de potencial via Galerkin-B-spline (IgA).

Usage::

    # Usar datos MNE de muestra (default)
    python test_iga_from_eeg_latent.py

    # Con un fichero EEG propio
    python test_iga_from_eeg_latent.py --file /path/to/recording.fif --n-dim 2

    # Usar solo 1 dimension latente para el analisis KM
    python test_iga_from_eeg_latent.py --analysis-dim 1 --column 0

    # Forzar recalculo del espacio latente (ignorar cache)
    python test_iga_from_eeg_latent.py --ignore-cache
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Añade el directorio padre (programas/) a sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.latent_space_extraction.siena_eeg import load_siena_eeg
from src.latent_space_extraction.extract_latent_subspace import extract_latent_space

# km_tools_v2 must be available in the environment
from src.potential_reconstruction.km_tools_v2 import (
    extract_km_coefficients,
    plot_km_components,
    reconstruct_potential,
    reconstruct_potential_1D,
    plot_potential,
    plot_potential_2d,
    plot_potential_slice,
)
from src.potential_reconstruction.bw_optimization import optimal_bw

from src.latent_space_extraction.data_analysis_tools import outliers_cleaning

from src.utils.config import (
    CONFIGS,
    BASE_RESULTS_PATH, 
    BASE_CACHE_PATH,
    DB_SIENA_PATH,
    )
from src.utils.io import save_potential

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KM + IgA pipeline from EEG latent subspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- EEG source ----
    parser.add_argument("--file", type=str, default=None,
                        help="Path to raw EEG file (.fif, .edf, .bdf, ...). "
                             "Uses MNE sample data if omitted.")
    parser.add_argument("--patient", type=str, default="PN01",
                        help="Patient ID for Siena EEG data.")
    parser.add_argument("--record", type=str, default="1",
                        help="Patient record for Siena EEG data.")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Start time (s) for EEG segment to analyze.")
    parser.add_argument("--t-end", type=float, default=None,
                        help="End time (s) for EEG segment to analyze.")
    # ---- Cache / persistence ----
    parser.add_argument("--cache-file", type=str, default="eeg_latent_cache.npz",
                        help="Path to cache file for the latent space. "
                             "If it exists and --ignore-cache is not set, "
                             "the extraction stage is skipped and the latent "
                             "space is loaded from this file.")
    parser.add_argument("--ignore-cache", action="store_true",
                        help="Ignore an existing cache file and force "
                             "recomputation of the latent space.")
    # ---- Latent-space extraction params ----
    parser.add_argument("--latent-dim", type=int, default=2,
                        help="Dimensionality of the latent subspace to extract "
                             "from EEG (default: 2)")
    parser.add_argument("--scoring-method", type=str, default="markov",
                        choices=["markov", "conservative", "weighted",
                                 "sequential", "pareto", "independent"],
                        help="Subspace selection strategy (default: markov)")
    parser.add_argument("--fc-metric", type=str, default="variance_sum",
                        choices=["variance_sum", "first_pc_var", "total_variance"])
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Quantile bins for Markov discretisation (default: 10)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel processes for subspace search")
    # ---- Which latent dimension(s) to feed into KM ----
    parser.add_argument("--analysis-dim", type=int, default=None,
                        help="Number of latent dimensions to use for the KM "
                             "analysis. If None, uses all extracted dimensions.")
    parser.add_argument("--column", type=int, default=None,
                        help="If analysis-dim=1, which latent column to use "
                             "(0=first, 1=second, etc.). Default: 0.")
    # ---- Preprocessing params ----
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--ica-method", type=str, default="picard")
    parser.add_argument("--verbose", action="store_true", default = True)
    # ---- Output ----
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save plots (default: current)")
    parser.add_argument("--period-label", type=str, default="unknown",
                        help="Label for the period (pre-ictal, inter-ictal, "
                             "post-ictal). Used in output directory naming.")
    parser.add_argument("--save-potential", action="store_true", default=True,
                    help="Guardar datos del potencial en .npz para post-procesamiento")
    parser.add_argument("--no-save-potential", action="store_true",
                    help="Deshabilitar el guardado del potencial")
    
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
print(DB_SIENA_PATH)
def main() -> int:
    args = _parse_args()
    if args.file is None:
        args.file = Path(f"{DB_SIENA_PATH}/{args.patient}/{args.patient}-{args.record}.edf")
    
    if args.out_dir is None:
        args.out_dir = BASE_RESULTS_PATH
        
    if args.t_start is None:
        args.t_start = 0.0
        
    if args.t_end is None:
        args.t_end = 60.0
        
    verbose = "INFO" if args.verbose else None
    
    if args.out_dir is None:
        args.out_dir = BASE_RESULTS_PATH
        
    out_dir = Path(args.out_dir + f"/siena/{args.patient}-{args.record}/{args.latent_dim}_latent_dim_{args.scoring_method}/from{args.t_start}_to_{args.t_end}_{args.period_label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()

    # =====================================================================
    # 1. EXTRACT (or LOAD CACHED) LATENT SUBSPACE FROM EEG
    # =====================================================================
    print("=" * 70)
    print("  STAGE 1: EXTRACT LATENT SUBSPACE FROM EEG")
    print("=" * 70)
    
    cache_path = Path(BASE_CACHE_PATH+f"/cache_eeg_siena/{args.patient}-{args.record}/from{args.t_start}_to_{args.t_end}_latent_dim_{args.latent_dim}_{args.scoring_method}.npz")
    # cache_path = Path(args.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # -----------------------------------------------------------------
    # Try to load from cache
    # -----------------------------------------------------------------
    if cache_path.exists() and not args.ignore_cache:
        print(f"\n  [CACHE] Found existing cache: {cache_path}")
        print("  [CACHE] Loading latent space and metadata...")
        loaded = np.load(cache_path, allow_pickle=True)
        latent = loaded["latent"]
        meta = loaded["meta"].item()
        print("  [CACHE] Loaded successfully.")
        # Ensure meta has the expected keys for downstream compatibility
        if "preprocessing" not in meta or "elapsed_time" not in meta:
            print("  [WARN] Cache file seems corrupted or outdated. Recomputing...")
            args.ignore_cache = True  # force recompute on next branch

    # -----------------------------------------------------------------
    # Compute if no cache or --ignore-cache
    # -----------------------------------------------------------------
    if not cache_path.exists() or args.ignore_cache:
        if args.ignore_cache and cache_path.exists():
            print("\n  [CACHE] --ignore-cache set. Recomputing latent space...")

        print("Using Siena EEG data...")
        raw_input = load_siena_eeg(args.file, t_minutes=(args.t_start, args.t_end), verbose=verbose, preload=False)

        latent, meta = extract_latent_space(
            raw_input,
            n_dim=args.latent_dim,
            scoring_method=args.scoring_method,
            fc_metric=args.fc_metric,
            n_bins=args.n_bins,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            ica_method=args.ica_method,
            n_workers=args.workers,
            verbose=verbose,
        )

        # Save to cache for future runs
        print(f"\n  [CACHE] Saving latent space to: {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, latent=latent, meta=np.array(meta, dtype=object))
        print("  [CACHE] Saved successfully.")

    # latent shape: (n_samples, latent_dim)
    n_samples, latent_dim = latent.shape
    sfreq = meta["preprocessing"]["sfreq"]
    dt = 1.0 / sfreq
    print("dt = ", dt)

    print(f"\n  Latent space shape : {latent.shape}")
    print(f"  Sampling freq      : {sfreq:.2f} Hz")
    print(f"  dt                 : {dt:.6f} s")
    print(f"  Selected ICs       : {meta['selected_indices']}")
    print(f"  Scores             : {meta['latent_scores']}")
    print(f"  Extraction time    : {meta['elapsed_time']:.1f} s")

    # =====================================================================
    # 2. SELECT DIMENSION(S) FOR KM ANALYSIS
    # =====================================================================
    analysis_dim = args.analysis_dim if args.analysis_dim is not None else latent_dim

    if analysis_dim > latent_dim:
        raise ValueError(
            f"analysis-dim ({analysis_dim}) cannot exceed latent-dim ({latent_dim})"
        )

    if analysis_dim == 1:
        col = args.column if args.column is not None else 0
        if col >= latent_dim:
            raise ValueError(f"column ({col}) must be < latent-dim ({latent_dim})")
        data = latent[:, col:col + 1]  # shape (n_samples, 1)
        print(f"\n  Using latent column {col} for 1D KM analysis")
    else:
        data = latent[:, :analysis_dim]  # shape (n_samples, analysis_dim)
        print(f"\n  Using first {analysis_dim} latent columns for KM analysis")

    D = analysis_dim
    print(f"  Data shape for KM  : {data.shape}")

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
    
    data = outliers_cleaning(data, method='iqr', threshold= 5)
    
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

    # =====================================================================
    # 3. CONFIGURATION
    # =====================================================================
    config = CONFIGS['base_2D_model'].copy()
    config.update({
        "model_name": f"eeg_latent_d{latent_dim}_{args.scoring_method}",
        "D": D,
        "bins": [50] * D,
        "drift_components": list(range(D)),
        "diff_components": [(i, i) for i in range(D)],
        "degree": 2,
    })

    print(f"\n  KM config: {config}")

    # =====================================================================
    # 4. BANDWIDTH OPTIMISATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 2: BANDWIDTH OPTIMISATION")
    print("=" * 70)

    result_bw = optimal_bw(
        data=data,
        bins=config["bins"],
        dt=dt,
        p=2,
        kernel="epanechnikov",
        theoretical=None,
        sigma_smooth=1.0,
        n_candidates=30,
        n_jobs=-1 if D + config["degree"] <= 4 else 1,
        plot=True,
        auto_weight=True,
    )
    bw_opt = result_bw["optimal_bw"]
    print(f"\n  Optimal bandwidth: {bw_opt:.4f}")
    print(f"  Drift error: {result_bw['error_drift'][result_bw['optimal_idx']]:.4f}")
    print(f"  Diffusion error: {result_bw['error_diff'][result_bw['optimal_idx']]:.4f}")

    # Save BW plot
    if "fig" in result_bw:
        result_bw["fig"].savefig(out_dir / "bw_optimisation.png", dpi=150)
        plt.close(result_bw["fig"])

    # =====================================================================
    # 5. ESTIMATE KM COEFFICIENTS
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 3: KM COEFFICIENT ESTIMATION")
    print("=" * 70)

    drift, diffusion, edges = extract_km_coefficients(
        data,
        bins=config["bins"],
        p=2,
        bw=bw_opt,
        kernel="epanechnikov",
        dt=dt,
        sigma_smooth=1.0,
        density_threshold=0.005,
    )

    print(f"\n  Drift shape    : {drift.shape}")
    print(f"  Diffusion shape: {diffusion.shape}")
    print(f"  Bins           : {[len(e) for e in edges]}")
    # =====================================================================
    # 6. EMPIRICAL DENSITY
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 4: EMPIRICAL DENSITY")
    print("=" * 70)

    hist_edges = []
    for d in range(D):
        c = edges[d]
        half = (c[1] - c[0]) / 2.0 if len(c) > 1 else 0.5
        e = np.concatenate([[c[0] - half], c + half])
        hist_edges.append(e)

    density, _ = np.histogramdd(data, bins=hist_edges)
    density = density.astype(float)
    print(f"  Density shape: {density.shape}")

    # =====================================================================
    # 7. PLOT KM COMPONENTS
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 5: PLOT KM COMPONENTS")
    print("=" * 70)

    fig_km = plot_km_components(
        drift, diffusion, edges,
        drift_components=config["drift_components"],
        diff_components=config["diff_components"],
        fixed_coords=None,
        theoretical=None,
        figsize=(12, 8),
    )
    plt.suptitle(
        f"{config['model_name'].upper()} — D^1 and D^2 estimated",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_km.savefig(out_dir / "km_components.png", dpi=150)
    plt.close(fig_km)
    print(f"  Saved km_components.png")

    # =====================================================================
    # 8. 1D POTENTIAL RECONSTRUCTION
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 6: 1D POTENTIAL RECONSTRUCTION (IgA)")
    print("=" * 70)

    U_rec_1d = reconstruct_potential_1D(
        drift, diffusion, edges,
        density=density,
        degree=config["degree"],
    )

    fig_pot_1d = plot_potential(
        U_rec_1d, edges,
        theoretical=None,
        component_labels=[f"x{j + 1}" for j in range(D)],
        figsize=(5 * D, 4),
    )
    plt.suptitle(
        f"{config['model_name'].upper()} — Reconstructed 1D potential (IgA)"
    )
    plt.tight_layout()
    fig_pot_1d.savefig(out_dir / "potential_1d.png", dpi=150)
    plt.close(fig_pot_1d)
    print(f"  Saved potential_1d.png")

    # =====================================================================
    # 9. FULL MULTIDIMENSIONAL POTENTIAL RECONSTRUCTION
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 7: FULL MULTIDIMENSIONAL POTENTIAL (IgA)")
    print("=" * 70)

    result_iga = reconstruct_potential(
        drift, diffusion, edges,
        density=density,
        method="iga",
        return_full=True,
        degree=config["degree"],
        decompose_helmholtz=True,
        density_threshold=0.005,
        rtol=1e-6,
        atol=1e-12,
    )

    U_rec = result_iga["potential"]
    print(f"\n  Reconstructed potential shape: {U_rec.shape}")
    print(f"  Method: Galerkin-B-spline (degree={config['degree']})")

    # Quality metrics
    rank_D = result_iga["rank_D"]
    condition_D = result_iga["condition_D"]
    frac_low_rank = np.mean(rank_D < D) if np.any(~np.isnan(rank_D)) else 0.0
    print(f"  Fraction cells with rank_D < {D}: {frac_low_rank:.2%}")
    print(f"  Condition D (median): {np.nanmedian(condition_D):.2e}")
    print(f"  Condition D (max)   : {np.nanmax(condition_D):.2e}")

    # Helmholtz metrics
    if "eta" in result_iga:
        eta = result_iga["eta"]
        is_eq = result_iga["is_equilibrium"]
        print(f"  Non-equilibrium metric eta: {eta:.4f}")
        print(f"  Detailed balance? {is_eq} (threshold eta < 0.1)")
        
    # =====================================================================
    # 9b. GUARDAR DATOS DEL POTENCIAL PARA POST-PROCESAMIENTO
    # =====================================================================
    save_potential_flag = args.save_potential and not args.no_save_potential

    if save_potential_flag:
        print("\n" + "-" * 50)
        print("  SAVING POTENTIAL DATA FOR POST-PROCESSING")
        print("-" * 50)

        metadata = {
            "patient": args.patient,
            "record": args.record,
            "period_label": args.period_label,
            "t_start": args.t_start,
            "t_end": args.t_end,
            "latent_dim": latent_dim,
            "analysis_dim": D,
            "scoring_method": args.scoring_method,
        }

        potential_path = save_potential(
            result_iga=result_iga,
            edges=edges,
            density=density,
            config=config,
            out_dir=out_dir,
            metadata=metadata,
            filename="potential_data.npz",
        )
        print(f"  Potential data saved: {potential_path}")

    # =====================================================================
    # 10. PLOT POTENCIAL 2D / SLICES (D >= 2)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 8: PLOT POTENCIAL 2D / SLICES")
    print("=" * 70)

    if D == 2:
        # --- Potencial 2D reconstruido ---
        fig_pot_2d = plot_potential_2d(
            U_rec, edges,
            title_est=f"{config['model_name'].upper()} — Reconstructed",
            figsize=(12, 5),
            unify_colorbar=True,
            align_minima=True,
            align_to_zero=True,
            crop_to_valid=True,
        )
        if fig_pot_2d is None:
            fig_pot_2d = plt.gcf()
        fig_pot_2d.savefig(out_dir / "potential_2d.png", dpi=150)
        plt.close(fig_pot_2d)
        print("  Saved potential_2d.png")

        # --- Residuo de Helmholtz ||r|| ---
        if "residual" in result_iga:
            residual = result_iga["residual"]
            r_norm = np.sqrt(np.nansum(residual**2, axis=0))
            x_c, y_c = edges
            X, Y = np.meshgrid(x_c, y_c, indexing="ij")
            fig_res = plt.figure(figsize=(6, 5))
            plt.contourf(X, Y, np.nan_to_num(r_norm, nan=0), levels=20, cmap="magma")
            plt.colorbar(label="||r||")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Residuo Helmholtz (no-equilibrio) — {config['model_name'].upper()}")
            plt.axis("equal")
            plt.tight_layout()
            fig_res.savefig(out_dir / "potential_2d_residual.png", dpi=150)
            plt.close(fig_res)
            print("  Saved potential_2d_residual.png")

    elif D >= 3:
        # --- Cortes 2D para D >= 3 ---
        dim_pairs = [(i, j) for i in range(min(D, 3)) for j in range(i + 1, min(D, 3))]
        for dims in dim_pairs:
            fig_slice = plot_potential_slice(
                U_rec, edges, dims=dims,
                fixed_coords={d: 0.0 for d in range(D) if d not in dims},
                title=f"{config['model_name'].upper()} — Reconstructed (slice x{dims[0]+1}-x{dims[1]+1})",
                figsize=(12, 5),
                unify_colorbar=True,
                align_minima=True,
                align_to_zero=True,
                crop_to_valid=True,
            )
            if fig_slice is None:
                fig_slice = plt.gcf()
            fname = f"potential_slice_x{dims[0]+1}_x{dims[1]+1}.png"
            fig_slice.savefig(out_dir / fname, dpi=150)
            plt.close(fig_slice)
            print(f"  Saved {fname}")
    else:
        print("  (Skipping 2D/slice plots for D=1)")

    # =====================================================================
    # 11. SUMMARY
    # =====================================================================
    total_time = time.time() - overall_t0
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Total wall-clock time: {total_time:.1f} s")
    print(f"  Output saved to: {out_dir.absolute()}")

    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())

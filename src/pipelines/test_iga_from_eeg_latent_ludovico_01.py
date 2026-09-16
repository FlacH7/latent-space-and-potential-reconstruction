#!/usr/bin/env python3
"""
test_iga_from_eeg_latent_ludovico_01.py
=======================================
Pipeline de Kramers-Moyal + IgA reconstruction para el dataset **Ludovico_01**.

A diferencia del pipeline de EEG (test-retest Gedai), este:

1. **Carga datos desde CSV** (columnas = canales, filas = muestras) via
   ``load_ludovico_01_from_ids``.
2. **No aplica filtrado de frecuencia** (``l_freq`` / ``h_freq`` no se usan).
3. **No aplica ICA ni ICLabel** (los datos no son EEG).  Cuando el stage2
   es ``pca_ica``, se reemplaza internamente por ``pca`` para evitar
   errores de ICA sobre datos no-EEG.
4. **Frecuencia de muestreo asumida**: 1000 Hz (1 ms por muestra).
5. **Sesión y tarea son ficticias**: ``session1``, ``default``.

Los resultados se guardan bajo ``results/ludovico_01/<subject>/``.

Plots y análisis exploratorio incluidos (paridad con test_retest_gedai):

- Stage 0: matriz de correlación de canales (las topografías se omiten
  porque los datos CSV no tienen montaje EEG).
- Stage 1: singular values y variance explained del embedding Hankel.
- Stage 2: PCA variance explained, Hankel PCA singular values, FastICA
  convergence (solo si aplica), DMD eigenvalue unit circle + frequency
  damping, Diffusion Maps eigenvalue spectrum + kernel diagnostics +
  2D components. Los plots ICA-specificos (ICLabel summary, pre/post
  ICA PSDs) se omiten porque Ludovico_01 no aplica ICA.
- Stage 3: Markov tau heatmap, tau ranked, selection vs distribution,
  transition matrix.
- Latent detail: zoom de timeseries.
- Pipeline overview: energy budget, flowchart.
- PSD crudo + PSD latente + influencia de canales.

Usage::

    # Nueva API: 3 etapas
    python -m src.pipelines.test_iga_from_eeg_latent_ludovico_01 \\
        --subject data_01_23_18 --t-start 0 --t-end 10 \\
        --stage1-embedding hankel --stage2-dynamics dmd --stage3-selection top_n

    # Legacy (mapeado a stage1='hankel', stage2='dmd', stage3='top_n')
    python -m src.pipelines.test_iga_from_eeg_latent_ludovico_01 \\
        --subject data_01_23_18 --t-start 0 --t-end 10 \\
        --scoring-method hankel_dmd
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.latent_space_extraction.extract_latent_subspace import (
    extract_latent_space,
    map_legacy_scoring_method,
)

from src.potential_reconstruction.km_tools_v2 import (
    extract_km_coefficients,
    plot_km_components,
    reconstruct_potential,
    reconstruct_potential_1D,
    plot_potential,
    plot_potential_2d,
    plot_potential_slice,
    plot_nonconservative_force_2d,
    plot_potential_combined_2d,
)
from src.potential_reconstruction.bw_optimization import optimal_bw

from src.latent_space_extraction.ludovico_01_eeg import (
    load_ludovico_01_from_ids,
    DEFAULT_SFREQ,
)

from src.latent_space_extraction.data_analysis_tools import outliers_cleaning

from src.latent_space_extraction.ck_test import chapman_kolmogorov_test

from src.utils.io import save_potential

from src.plotters.trajectory_plots import plot_latent_trajectory

# --- Módulo de análisis espectral de potencia (PSD) ---
from src.spectral_analysis.psd_analysis import (
    compute_and_plot_raw_psd,
    compute_and_plot_latent_psd,
    compute_channel_influence_on_latent,
)

from src.utils.config import (
    CONFIGS,
    BASE_RESULTS_PATH,
    BASE_CACHE_PATH,
)

# Try to import Ludovico_01 path from config
try:
    from src.utils.config import DB_LUDOVICO_01_PATH
except (ImportError, AttributeError):
    DB_LUDOVICO_01_PATH = None

# --- NUEVOS IMPORTS: modulo de ploteo del pipeline (src.plotters) ---
from src.plotters import (
    # Stage 0
    plot_channel_topographies,
    plot_channel_correlation_matrix,
    # Stage 1
    plot_hankel_singular_values,
    plot_hankel_variance_explained,
    # Stage 2
    plot_pca_variance_explained,
    plot_pre_ica_component_psds,
    plot_icalabel_summary,
    plot_post_ica_component_psds,
    plot_hankel_pca_singular_values,
    plot_fastica_convergence,
    plot_dmd_eigenvalue_unit_circle,
    plot_dmd_frequency_damping,
    plot_diffusion_eigenvalue_spectrum,
    plot_diffusion_kernel_diagnostics,
    plot_diffusion_2d_components,
    # Stage 3
    plot_markov_tau_heatmap,
    plot_markov_tau_ranked,
    plot_markov_selection_vs_distribution,
    plot_markov_transition_matrix,
    # Latent detail
    plot_latent_timeseries_zoom,
    # Pipeline overview
    plot_pipeline_energy_budget,
    plot_pipeline_flowchart,
    # Style setup
    setup_plotting_style,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KM + IgA pipeline for Ludovico_01 CSV data (non-EEG)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- Data source ----
    parser.add_argument("--subject", type=str, required=True,
                        help="Subject/filename ID (e.g. data_01_23_18, without .csv)")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Override Ludovico_01 dataset root path")
    parser.add_argument("--sfreq", type=float, default=DEFAULT_SFREQ,
                        help=f"Sampling frequency in Hz (default: {DEFAULT_SFREQ})")
    parser.add_argument("--t-start", type=float, default=None,
                        help="Start time (s) for the segment to analyze.")
    parser.add_argument("--t-end", type=float, default=None,
                        help="End time (s) for the segment to analyze.")
    # ---- Cache / persistence ----
    parser.add_argument("--cache-file", type=str, default=None,
                        help="Path to cache file for the latent space.")
    parser.add_argument("--ignore-cache", action="store_true",
                        help="Ignore an existing cache file and force recomputation.")
    # ---- Latent-space extraction params ----
    parser.add_argument("--latent-dim", type=int, default=2,
                        help="Dimensionality of the latent subspace (default: 2)")
    # ---- NEW 3-stage pipeline API ----
    parser.add_argument("--stage1-embedding", type=str, default=None,
                        choices=["none", "hankel"],
                        help="Stage 1 embedding. If omitted, the legacy "
                             "--scoring-method mapping is used.")
    parser.add_argument("--stage1-params", type=str, default=None,
                        help='JSON dict of Stage-1 params, e.g. \'{"depth": 250}\'')
    parser.add_argument("--stage2-dynamics", type=str, default=None,
                        choices=["pca_ica", "pca", "dmd", "diffusion_maps", "cdhsa_specific_modes"],
                        help="Stage 2 dynamics (new API).")
    parser.add_argument("--stage2-params", type=str, default=None,
                        help='JSON dict of Stage-2 params, e.g. \'{"svd_rank": 50}\'')
    parser.add_argument("--stage3-selection", type=str, default=None,
                        choices=["top_n", "markov_fastest", "markov_slowest"],
                        help="Stage 3 selection (new API).")
    parser.add_argument("--stage3-params", type=str, default=None,
                        help='JSON dict of Stage-3 params, e.g. \'{"n_bins": 10}\'')
    # ---- Legacy scoring API (mapped onto the new stages) ----
    parser.add_argument("--scoring-method", type=str, default="hankel_dmd",
                        choices=["markov", "markov_inverted", "conservative", "weighted",
                                 "sequential", "pareto", "independent",
                                 "hankel_dmd", "diffusion_maps"],
                        help="Legacy subspace selection strategy (default: hankel_dmd). "
                             "Mapped internally onto the new 3-stage API. Ignored if "
                             "any --stageX argument is given.")
    parser.add_argument("--fc-metric", type=str, default="variance_sum",
                        choices=["variance_sum", "first_pc_var", "total_variance"])
    parser.add_argument("--n-bins", type=int, default=10,
                        help="Quantile bins for Markov discretisation (default: 10)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel processes for subspace search")
    parser.add_argument("--search-strategy", type=str, default="exhaustive",
                        choices=["exhaustive", "greedy"])
    # ---- Which latent dimension(s) to feed into KM ----
    parser.add_argument("--analysis-dim", type=int, default=None,
                        help="Number of latent dimensions to use for KM. "
                             "If None, uses all extracted dimensions.")
    parser.add_argument("--column", type=int, default=None,
                        help="If analysis-dim=1, which latent column to use (0=first).")
    # ---- Hankel (legacy convenience; also usable as stage1 param) ----
    parser.add_argument("--hankel-embedding-depth", type=int, default=None,
                        help="Hankel embedding depth T. None = auto.")
    # ---- Diffusion Maps params ----
    parser.add_argument(
        "--diffusion-sigma", type=float, default=None,
        help=("Sigma for Diffusion Maps Gaussian kernel. "
              "If None, auto-computed via bgh method (Berry-Giannakis-Harlim).")
    )
    parser.add_argument(
        "--diffusion-k", type=int, default=100,
        help="Number of nearest neighbors for sparse affinity matrix."
    )
    parser.add_argument(
        "--diffusion-time", type=float, default=0.0,
        help=("Diffusion time t >= 0. Higher values filter fine-scale noise "
              "and highlight macroscopic dynamics (multiscale filtering).")
    )
    parser.add_argument(
        "--diffusion-alpha", type=float, default=0.5,
        help=("Density normalization parameter (Coifman-Lafon). "
              "0.0=Laplacian Eigenmaps, 0.5=Diffusion Maps (default), 1.0=Fokker-Planck.")
    )
    # ---- KM ----
    parser.add_argument("--km-bins", type=int, default=40,
                        help="Number of bins per dimension for KM estimation (default: 40)")
    # ---- Output ----
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to save plots (default: current)")
    parser.add_argument("--save-potential", action="store_true", default=True,
                        help="Guardar datos del potencial en .npz para post-procesamiento")
    parser.add_argument("--no-save-potential", action="store_true",
                        help="Deshabilitar el guardado del potencial")
    # ---- CD-HSA specific modes ----
    parser.add_argument("--mode-map-path", type=str, default=None,
                        help="Path to mode_map.json (from CDHSA batch runner). "
                             "Required when stage2-dynamics=cdhsa_specific_modes.")
    parser.add_argument("--condition", type=str, default=None,
                        help="Condition name to extract from mode_map.json "
                             "(e.g. 'data_01_23_18'). Required for cdhsa_specific_modes.")
    parser.add_argument("--verbose", action="store_true", default=True)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline-spec resolution: new stage API or legacy scoring-method mapping
# ---------------------------------------------------------------------------


def _json_params(raw: str | None) -> dict:
    """Parse a JSON params dict from the CLI (empty dict when omitted)."""
    if raw is None:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Stage params must be a JSON object, got: {raw!r}")
    return parsed


def _resolve_pipeline_spec(args: argparse.Namespace) -> dict:
    """
    Resolve the effective 3-stage specification.

    * If any ``--stageX`` argument is given -> new API (missing stages take
      their defaults; CLI convenience flags --diffusion-* /
      --hankel-embedding-depth / --n-bins are injected into the JSON
      params when not already present).
    * Otherwise -> legacy ``--scoring-method`` mapping via
      :func:`map_legacy_scoring_method`.

    For Ludovico_01 (non-EEG), if stage2 is ``pca_ica`` it is replaced
    by ``pca`` to skip ICA.
    """
    use_new_api = any([
        args.stage1_embedding is not None,
        args.stage2_dynamics is not None,
        args.stage3_selection is not None,
    ])

    if use_new_api:
        s1 = None if args.stage1_embedding in (None, "none") else args.stage1_embedding
        s2 = args.stage2_dynamics or "pca"
        s3 = args.stage3_selection or "top_n"

        # Ludovico_01 is NOT EEG -> replace pca_ica with pca to avoid ICA
        if s2 == "pca_ica":
            print("  [Ludovico01] Replacing stage2 'pca_ica' with 'pca' "
                  "(no ICA for non-EEG data)")
            s2 = "pca"

        p1 = _json_params(args.stage1_params)
        p2 = _json_params(args.stage2_params)
        p3 = _json_params(args.stage3_params)

        if s2 == "cdhsa_specific_modes":
            # Auto-inject --condition (the subject name IS the condition)
            if args.condition is None:
                args.condition = args.subject
                print(f"  [Ludovico01] Auto-injecting condition='{args.condition}' "
                      f"(from --subject) for cdhsa_specific_modes")
            p2.setdefault("condition", args.condition)
            if args.mode_map_path is not None:
                p2.setdefault("mode_map_path", args.mode_map_path)

        # Inject CLI convenience flags when not overridden in the JSON
        if s1 == "hankel":
            p1.setdefault("depth", args.hankel_embedding_depth)
        if s2 == "diffusion_maps":
            p2.setdefault("sigma", args.diffusion_sigma)
            p2.setdefault("k", args.diffusion_k)
            p2.setdefault("diffusion_time", args.diffusion_time)
            p2.setdefault("alpha", args.diffusion_alpha)
        if s3 in ("markov_fastest", "markov_slowest"):
            p3.setdefault("n_bins", args.n_bins)
            p3.setdefault("search_strategy", args.search_strategy)

        return {
            "stage1_embedding": s1,
            "stage1_params": p1,
            "stage2_dynamics": s2,
            "stage2_params": p2,
            "stage3_selection": s3,
            "stage3_params": p3,
        }

    # Legacy mapping
    spec = map_legacy_scoring_method(
        args.scoring_method,
        n_dim=args.latent_dim,
        fc_metric=args.fc_metric,
        n_bins=args.n_bins,
        hankel_embedding_depth=args.hankel_embedding_depth,
        diffusion_sigma=args.diffusion_sigma,
        diffusion_k=args.diffusion_k,
        diffusion_time=args.diffusion_time,
        diffusion_alpha=args.diffusion_alpha,
        ica_method="picard",  # ignored for non-EEG, but required by signature
        search_strategy=args.search_strategy,
    )

    # Ludovico_01 is NOT EEG -> replace pca_ica with pca to avoid ICA
    if spec["stage2_dynamics"] == "pca_ica":
        print("  [Ludovico01] Replacing stage2 'pca_ica' with 'pca' "
              "(no ICA for non-EEG data)")
        spec["stage2_dynamics"] = "pca"

    return spec


def _spec_label(spec: dict) -> str:
    """Short human-readable label for the stage chain (used in paths)."""
    s1 = spec["stage1_embedding"] or "none"
    return f"{s1}+{spec['stage2_dynamics']}+{spec['stage3_selection']}"


def _spec_hash(spec: dict) -> str:
    """
    Short hash of the full stage spec (stages + params).

    Included in the cache key so that different parameter combinations
    never collide.
    """
    payload = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    if args.out_dir is None:
        args.out_dir = BASE_RESULTS_PATH
    verbose = "INFO" if args.verbose else None

    save_potential_flag = args.save_potential and not args.no_save_potential

    # Defaults for time window
    if args.t_start is None:
        args.t_start = 0.0
    if args.t_end is None:
        args.t_end = 10.0

    db_path = args.db_path or DB_LUDOVICO_01_PATH

    # Resolve the 3-stage spec
    spec = _resolve_pipeline_spec(args)
    spec_label = _spec_label(spec)
    spec_hash = _spec_hash(spec)
    print(f"\n  Pipeline spec : {spec_label}  (hash {spec_hash})")
    print(f"  Stage params  : {json.dumps(spec, default=str)}")

    # -----------------------------------------------------------------
    # Output directory: ludovico_01/<subject>/session1/...
    # -----------------------------------------------------------------
    out_dir = Path(
        args.out_dir
        + f"/ludovico_01/{args.subject}/session1"
        + f"/{args.latent_dim}_latent_dim_{spec_label}_{spec_hash}"
        + f"/from{args.t_start}s_to_{args.t_end}s_default"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()

    # =================================================================
    # 1. LOAD CSV DATA
    # =================================================================
    print("=" * 70)
    print("  STAGE 0: LOAD LUDOVICO_01 CSV DATA")
    print("=" * 70)

    try:
        raw = load_ludovico_01_from_ids(
            subject=args.subject,
            db_path=db_path,
            sfreq=args.sfreq,
            t_start=args.t_start,
            t_stop=args.t_end,
            preload=False,
            verbose=verbose,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    sfreq = args.sfreq
    print(f"  Selected channels : {len(raw.ch_names)}")
    print(f"  Channel names     : {raw.ch_names[:5]}{'...' if len(raw.ch_names) > 5 else ''}")
    print(f"  Sampling freq     : {sfreq:.2f} Hz")
    print(f"  Cropped window    : {raw.times[0]:.2f} s -> {raw.times[-1]:.2f} s "
          f"(duration: {raw.times[-1] - raw.times[0]:.2f} s)")

    # -----------------------------------------------------------------
    # PSD de los canales originales (módulo spectral_analysis).
    # l_freq/h_freq son None en este pipeline -> defaults fmin=1.0,
    # fmax=100.0 Hz (clamp automático a Nyquist si sfreq es menor).
    # -----------------------------------------------------------------
    psds_raw, freqs_raw, ch_names, mean_psd, std_psd, raw_psd_path = compute_and_plot_raw_psd(
        raw, out_dir=out_dir,
    )

    # -----------------------------------------------------------------
    # STAGE 0: EXPLORATORY PLOTS
    # Ludovico_01 es no-EEG: las topografías se omiten (sin montaje),
    # pero la matriz de correlación entre canales es útil y se genera.
    # -----------------------------------------------------------------
    print("  Generando plots exploratorios (Stage 0)...")
    try:
        setup_plotting_style()
        # plot_channel_topographies requiere montaje EEG -> se omite para CSV
        try:
            plot_channel_topographies(
                raw, out_dir=out_dir,
                fmin=None, fmax=None,
                subject=args.subject, session="session1", task="default",
            )
        except Exception as e_topo:
            print(f"  [INFO] Topografías omitidas (datos no-EEG sin montaje): {e_topo}")
        plot_channel_correlation_matrix(
            raw, out_dir=out_dir,
            subject=args.subject, session="session1", task="default",
        )
        print("  [OK] Plots exploratorios guardados.")
    except Exception as e:
        print(f"  [WARN] Error en plots exploratorios: {e}")

    # =================================================================
    # 2. EXTRACT (or LOAD CACHED) LATENT SUBSPACE
    # =================================================================
    print("\n" + "=" * 70)
    print("  STAGE 1: EXTRACT LATENT SUBSPACE FROM CSV DATA")
    print("=" * 70)

    # Cache path
    if args.cache_file is None:
        args.cache_file = Path(
            BASE_CACHE_PATH
            + f"/cache_ludovico_01/{args.subject}/session1"
            + f"/task_default_latent_dim_{args.latent_dim}_{spec_label}_{spec_hash}"
            + f"/from{args.t_start}s_to{args.t_end}s.npz"
        )
    cache_path = Path(args.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Try cache
    if cache_path.exists() and not args.ignore_cache:
        print(f"\n  [CACHE] Found existing cache: {cache_path}")
        print("  [CACHE] Loading latent space and metadata...")
        loaded = np.load(cache_path, allow_pickle=True)
        latent = loaded["latent"]
        meta = loaded["meta"].item()
        print("  [CACHE] Loaded successfully.")
        if "preprocessing" not in meta or "elapsed_time" not in meta:
            print("  [WARN] Cache file seems corrupted or outdated. Recomputing...")
            args.ignore_cache = True

    # Compute if no cache
    if not cache_path.exists() or args.ignore_cache:
        if args.ignore_cache and cache_path.exists():
            print("\n  [CACHE] --ignore-cache set. Recomputing latent space...")

        # Ludovico_01 is NOT EEG -> pass None for l_freq, h_freq to skip filtering
        latent, meta = extract_latent_space(
            raw,
            n_dim=args.latent_dim,
            stage1_embedding=spec["stage1_embedding"],
            stage1_params=spec["stage1_params"],
            stage2_dynamics=spec["stage2_dynamics"],
            stage2_params=spec["stage2_params"],
            stage3_selection=spec["stage3_selection"],
            stage3_params=spec["stage3_params"],
            l_freq=None,   # No frequency filtering for non-EEG data
            h_freq=None,
            n_workers=args.workers,
            verbose=verbose,
        )

        print(f"\n  [CACHE] Saving latent space to: {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, latent=latent, meta=np.array(meta, dtype=object))
        print("  [CACHE] Saved successfully.")

    # latent shape: (n_samples, latent_dim)
    n_samples, latent_dim = latent.shape
    dt = 1.0 / sfreq
    print(f"\n  dt = {dt:.6f} s")
    print(f"  Latent space shape : {latent.shape}")
    print(f"  Selected ICs       : {meta['selected_indices']}")
    print(f"  Scores             : {meta['latent_scores']}")
    print(f"  Extraction time    : {meta['elapsed_time']:.1f} s")

    # -----------------------------------------------------------------
    # STAGE 1: HANKEL EMBEDDING PLOTS
    # -----------------------------------------------------------------
    if spec["stage1_embedding"] == "hankel":
        print("  Generando plots Stage 1 (Hankel)...")
        try:
            from src.latent_space_extraction.hankel_dmd_extractor import (
                _build_multivariate_hankel,
            )
            depth = meta["stage1"].get("depth")
            X_input = meta["stage1"].get("input_data")
            if X_input is not None and depth is not None:
                H_plot = _build_multivariate_hankel(X_input, depth)
                plot_hankel_singular_values(
                    H_plot, out_dir=out_dir, embedding_depth=depth,
                )
                plot_hankel_variance_explained(
                    H_plot, out_dir=out_dir, embedding_depth=depth,
                )
                print("  [OK] Plots Stage 1 guardados.")
        except Exception as e:
            print(f"  [WARN] Error en plots Stage 1: {e}")

    # -----------------------------------------------------------------
    # STAGE 2: DYNAMICS PLOTS
    # Para Ludovico_01 (no-EEG), se omiten los plots que requieren ICA:
    #   - plot_pre_ica_component_psds
    #   - plot_icalabel_summary
    #   - plot_post_ica_component_psds
    #   - plot_fastica_convergence (solo aplica a pca_ica)
    # El resto (PCA, Hankel PCA, DMD, Diffusion Maps) son genéricos.
    # -----------------------------------------------------------------
    print("  Generando plots de dinamica (Stage 2)...")
    try:
        stage2_meta = meta["stage2"]
        # PCA variance explained
        plot_pca_variance_explained(stage2_meta, out_dir=out_dir)
        # ICLabel summary (skip: no ICA for non-EEG)
        # plot_icalabel_summary(stage2_meta, out_dir=out_dir)
        # Pre/post ICA PSDs (skip: no ICA for non-EEG)
        # plot_pre_ica_component_psds(raw, stage2_meta, out_dir=out_dir, ...)
        # plot_post_ica_component_psds(stage2_meta, sfreq=sfreq, out_dir=out_dir, ...)
        # Hankel PCA singular values (solo para rama Hankel)
        plot_hankel_pca_singular_values(stage2_meta, out_dir=out_dir)
        # FastICA convergence (skip: no ICA for non-EEG)
        # plot_fastica_convergence(stage2_meta, out_dir=out_dir)
        # DMD
        plot_dmd_eigenvalue_unit_circle(stage2_meta, out_dir=out_dir)
        plot_dmd_frequency_damping(stage2_meta, out_dir=out_dir)
        # Diffusion Maps
        plot_diffusion_eigenvalue_spectrum(stage2_meta, out_dir=out_dir)
        # Kernel diagnostics (puede ser costoso; best-effort)
        try:
            dm_input = stage2_meta.get("dm_input_data")
            if dm_input is None and spec["stage1_embedding"] == "hankel":
                dm_input = meta["stage1"].get("input_data")
            plot_diffusion_kernel_diagnostics(stage2_meta, dm_input, out_dir=out_dir)
        except Exception as e:
            print(f"  [WARN] Kernel diagnostics omitido: {e}")
        plot_diffusion_2d_components(meta, out_dir=out_dir)
        print("  [OK] Plots Stage 2 guardados.")
    except Exception as e:
        print(f"  [WARN] Error en plots Stage 2: {e}")

    # -----------------------------------------------------------------
    # STAGE 3: MARKOV SELECTION PLOTS
    # -----------------------------------------------------------------
    if "all_markov_taus" in meta.get("stage3", {}):
        all_taus = meta["stage3"]["all_markov_taus"]
        selected = tuple(meta["selected_indices"])
        try:
            plot_markov_tau_heatmap(all_taus, out_dir=out_dir,
                                    selected_combination=selected)
            plot_markov_tau_ranked(all_taus, out_dir=out_dir,
                                   selected_combination=selected)
            tau_sel = meta["stage3"]["scores"].get("tau")
            if tau_sel is not None and selected:
                maximize = meta["stage3"]["scores"].get("maximize", False)
                plot_markov_selection_vs_distribution(
                    all_taus, tau_sel, selected,
                    maximize=maximize, out_dir=out_dir,
                )
            print("  [OK] Plots Stage 3 (Markov) guardados.")
        except Exception as e:
            print(f"  [WARN] Error en plots Stage 3: {e}")

    # Matriz de transicion de la combinacion seleccionada (siempre disponible
    # cuando se uso markov)
    if "markov" in spec["stage3_selection"]:
        try:
            Y2 = meta["Y"]   # (D, T)
            n_bins_s3 = meta["stage3"]["scores"].get("n_bins", 10)
            selected = tuple(meta["selected_indices"])
            plot_markov_transition_matrix(Y2, selected, n_bins_s3, out_dir=out_dir)
            print("  [OK] Matriz de transicion guardada.")
        except Exception as e:
            print(f"  [WARN] Error en plot de matriz de transicion: {e}")

    # -----------------------------------------------------------------
    # PSD del espacio latente + influencia de canales sobre cada dimensión.
    # Sin montaje topográfico (datos CSV) -> barras horizontales.
    # -----------------------------------------------------------------
    psds_latent, freqs_latent, mean_latent, std_latent, latent_psd_path = compute_and_plot_latent_psd(
        latent, sfreq=sfreq, out_dir=out_dir,
    )
    influence_weights, ch_names, fig_path, data_path = compute_channel_influence_on_latent(
        raw, latent, meta, out_dir=out_dir, raw_psd_path=raw_psd_path,
    )

    # Plot latent trajectory
    plot_latent_trajectory(
        latent,
        out_dir=out_dir,
        method_name=spec_label,
    )

    # =================================================================
    # 3. SELECT DIMENSION(S) FOR KM ANALYSIS
    # =================================================================
    analysis_dim = args.analysis_dim if args.analysis_dim is not None else latent_dim

    if analysis_dim > latent_dim:
        raise ValueError(
            f"analysis-dim ({analysis_dim}) cannot exceed latent-dim ({latent_dim})"
        )

    if analysis_dim == 1:
        col = args.column if args.column is not None else 0
        if col >= latent_dim:
            raise ValueError(f"column ({col}) must be < latent-dim ({latent_dim})")
        data = latent[:, col:col + 1]
        print(f"\n  Using latent column {col} for 1D KM analysis")
    else:
        data = latent[:, :analysis_dim]
        print(f"\n  Using first {analysis_dim} latent columns for KM analysis")

    ck_result = chapman_kolmogorov_test(
        data, dt=dt, n_bins=20, threshold=0.15,
        plot=True, out_dir=out_dir, verbose=True,
    )

    if not ck_result["is_markovian"]:
        print("\n  [WARN] Latent space is NOT Markovian. KM results may be invalid.")
        print("  Consider: increasing latent_dim, increasing embedding_depth,")
        print("  or switching to a different stage combination.")
    else:
        print(f"\n  [OK] Markovian at tau* = {ck_result['tau_star']:.4f} s "
              f"({ck_result['tau_star_idx']} steps)")

    D = analysis_dim
    print(f"  Data shape for KM  : {data.shape}")

    # Plot latent time series
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

    # --- LATENT DETAIL: ZOOM ---
    try:
        plot_latent_timeseries_zoom(latent, sfreq=sfreq, out_dir=out_dir)
        print("  Saved latent_timeseries_zoom.png")
    except Exception as e:
        print(f"  [WARN] Error en plot de zoom latente: {e}")

    data = outliers_cleaning(data, method="iqr", threshold=5)

    # Plot cleaned
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

    # =================================================================
    # 4. CONFIGURATION
    # =================================================================
    config = CONFIGS['base_2D_model'].copy()
    config.update({
        "model_name": f"ludovico_01_{args.subject}_d{latent_dim}_{spec_label}",
        "D": D,
        "bins": [args.km_bins] * D,
        "drift_components": list(range(D)),
        "diff_components": [(i, i) for i in range(D)],
        "degree": 2,
    })
    print(f"\n  KM config: {config}")

    # =================================================================
    # 5. BANDWIDTH OPTIMISATION
    # =================================================================
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

    if "fig" in result_bw:
        result_bw["fig"].savefig(out_dir / "bw_optimisation.png", dpi=150)
        plt.close(result_bw["fig"])

    # =================================================================
    # 6. ESTIMATE KM COEFFICIENTS
    # =================================================================
    print("\n" + "=" * 70)
    print("  STAGE 3: KM COEFFICIENT ESTIMATION")
    print("=" * 70)

    drift, diffusion, edges = extract_km_coefficients(
        data, bins=config["bins"], p=2, bw=bw_opt,
        kernel="epanechnikov", dt=dt, sigma_smooth=1.0,
        density_threshold=0.01,
    )

    print(f"\n  Drift shape    : {drift.shape}")
    print(f"  Diffusion shape: {diffusion.shape}")
    print(f"  Bins           : {[len(e) for e in edges]}")

    # =================================================================
    # 7. EMPIRICAL DENSITY
    # =================================================================
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

    # =================================================================
    # 8. PLOT KM COMPONENTS
    # =================================================================
    print("\n" + "=" * 70)
    print("  STAGE 5: PLOT KM COMPONENTS")
    print("=" * 70)

    figs_km = plot_km_components(
        drift, diffusion, edges,
        drift_components=config["drift_components"],
        diff_components=config["diff_components"],
        fixed_coords=None, theoretical=None,
        figsize=(12, 8),
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

    # =================================================================
    # 9. 1D POTENTIAL RECONSTRUCTION
    # =================================================================
    print("\n" + "=" * 70)
    print("  STAGE 6: 1D POTENTIAL RECONSTRUCTION (IgA)")
    print("=" * 70)

    U_rec_1d = reconstruct_potential_1D(
        drift, diffusion, edges, density=density, degree=config["degree"],
    )

    fig_pot_1d = plot_potential(
        U_rec_1d, edges, theoretical=None,
        component_labels=[f"x{j + 1}" for j in range(D)],
        figsize=(5 * D, 4),
    )
    plt.suptitle(f"{config['model_name'].upper()} -- Reconstructed 1D potential (IgA)")
    plt.tight_layout()
    fig_pot_1d.savefig(out_dir / "potential_1d.png", dpi=150)
    plt.close(fig_pot_1d)
    print(f"  Saved potential_1d.png")

    # =================================================================
    # 10. FULL MULTIDIMENSIONAL POTENTIAL RECONSTRUCTION
    # =================================================================
    print("\n" + "=" * 70)
    print("  STAGE 7: FULL MULTIDIMENSIONAL POTENTIAL (IgA)")
    print("=" * 70)

    result_iga = reconstruct_potential(
        drift, diffusion, edges, density=density,
        method="iga", return_full=True, degree=config["degree"],
        decompose_helmholtz=True, density_threshold=0.01,
        rtol=5e-2, atol=1e-12, compute_stream_function=True,
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

    # =================================================================
    # 10b. GUARDAR DATOS DEL POTENCIAL PARA POST-PROCESAMIENTO
    # =================================================================
    if save_potential_flag:
        print("\n" + "-" * 50)
        print("  SAVING POTENTIAL DATA FOR POST-PROCESSING")
        print("-" * 50)

        metadata = {
            "subject": args.subject,
            "session": "session1",
            "task": "default",
            "t_start": args.t_start,
            "t_end": args.t_end,
            "latent_dim": latent_dim,
            "analysis_dim": D,
            "pipeline": spec_label,
            "pipeline_spec": {k: v for k, v in spec.items()
                             if k.endswith("params") or k.startswith("stage")},
            "scoring_method": args.scoring_method,
        }

        potential_path = save_potential(
            result_iga=result_iga, edges=edges, density=density,
            config=config, out_dir=out_dir, metadata=metadata,
            filename="potential_data.npz",
        )
        print(f"  Potential data saved: {potential_path}")

    # =================================================================
    # 11. PLOT POTENCIAL 2D / SLICES
    # =================================================================
    print("\n" + "=" * 70)
    print("  STAGE 8: PLOT POTENCIAL 2D / SLICES")
    print("=" * 70)

    if D == 2:
        # Potential 2D (no streamlines)
        fig_pot_2d = plot_potential_2d(
            U_rec, edges,
            title_est=f"{config['model_name'].upper()} -- Reconstructed",
            figsize=(12, 5), unify_colorbar=True,
            align_minima=True, align_to_zero=True,
            crop_to_valid=True, show_streamlines=False,
            stream_function=result_iga.get('stream_function'),
            reconstructed_field=result_iga.get('reconstructed_field'),
        )
        if fig_pot_2d is None:
            fig_pot_2d = plt.gcf()
        fig_pot_2d.savefig(out_dir / "potential_2d.png", dpi=150)
        plt.close(fig_pot_2d)
        print("  Saved potential_2d.png")

        # Potential 2D with streamlines
        fig_pot_2d = plot_potential_2d(
            U_rec, edges,
            title_est=f"{config['model_name'].upper()} -- Reconstructed",
            figsize=(12, 5), unify_colorbar=True,
            align_minima=True, align_to_zero=True,
            crop_to_valid=True,
            stream_function=result_iga.get('stream_function'),
            reconstructed_field=result_iga.get('reconstructed_drift'),
        )
        if fig_pot_2d is None:
            fig_pot_2d = plt.gcf()
        fig_pot_2d.savefig(out_dir / "potential_2d_streamlines.png", dpi=150)
        plt.close(fig_pot_2d)
        print("  Saved potential_2d_streamlines.png")

        # Residual
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
            plt.title(f"Residuo Helmholtz (no-equilibrio) -- {config['model_name'].upper()}")
            plt.axis("equal")
            plt.tight_layout()
            fig_res.savefig(out_dir / "potential_2d_residual.png", dpi=150)
            plt.close(fig_res)
            print("  Saved potential_2d_residual.png")

        # Non-conservative force
        if result_iga.get('nonconservative_force') is not None:
            fig_v = plot_nonconservative_force_2d(
                result_iga['nonconservative_force'], edges,
                title=f"Fuerza no-conservativa v = f + D·∇U — {config['model_name'].upper()}",
                crop_to_valid=True,
            )
            fig_v.savefig(out_dir / "potential_2d_nonconservative_force.png", dpi=150)
            plt.close(fig_v)
            print("  Saved potential_2d_nonconservative_force.png")

        # Combined
        if (result_iga.get('reconstructed_field') is not None
                or result_iga.get('nonconservative_force') is not None):
            fig_comb = plot_potential_combined_2d(
                U_rec, edges,
                stream_function=result_iga.get('stream_function'),
                nonconservative_force=result_iga.get('nonconservative_force'),
                reconstructed_field=result_iga.get('reconstructed_drift'),
                title=f"{config['model_name'].upper()} — U (fondo) + v (flechas rojas) + streamlines (blanco)",
                crop_to_valid=True,
            )
            fig_comb.savefig(out_dir / "potential_2d_combined.png", dpi=150)
            plt.close(fig_comb)
            print("  Saved potential_2d_combined.png")

    elif D >= 3:
        dim_pairs = [(i, j) for i in range(min(D, 3)) for j in range(i + 1, min(D, 3))]
        for dims in dim_pairs:
            fig_slice = plot_potential_slice(
                U_rec, edges, dims=dims,
                fixed_coords={d: 0.0 for d in range(D) if d not in dims},
                title=f"{config['model_name'].upper()} -- Reconstructed (slice x{dims[0]+1}-x{dims[1]+1})",
                figsize=(12, 5), unify_colorbar=True,
                align_minima=True, align_to_zero=True, crop_to_valid=True,
            )
            if fig_slice is None:
                fig_slice = plt.gcf()
            fname = f"potential_slice_x{dims[0]+1}_x{dims[1]+1}.png"
            fig_slice.savefig(out_dir / fname, dpi=150)
            plt.close(fig_slice)
            print(f"  Saved {fname}")
    else:
        print("  (Skipping 2D/slice plots for D=1)")

    # =================================================================
    # 12. SUMMARY
    # =================================================================
    # --- PIPELINE OVERVIEW (cross-stage plots) ---
    try:
        plot_pipeline_energy_budget(
            X_filtered=meta.get("preprocessing", {}).get("X_filtered"),
            meta=meta, latent=latent, out_dir=out_dir,
        )
        print("  Saved pipeline_energy_budget.png")
        plot_pipeline_flowchart(
            meta=meta, out_dir=out_dir,
            l_freq=None, h_freq=None,
            n_channels=len(raw.ch_names), sfreq=sfreq,
        )
        print("  Saved pipeline_flowchart.png")
    except Exception as e:
        print(f"  [WARN] Error en plots overview: {e}")

    total_time = time.time() - overall_t0
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Subject             : {args.subject}")
    print(f"  Pipeline            : {spec_label} (hash {spec_hash})")
    print(f"  Time window         : {args.t_start:.1f}s -> {args.t_end:.1f}s")
    print(f"  Total wall-clock    : {total_time:.1f} s")
    print(f"  Output saved to     : {out_dir.absolute()}")
    if save_potential_flag:
        print(f"  Potential data      : {out_dir / 'potential_data.npz'}")

    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())

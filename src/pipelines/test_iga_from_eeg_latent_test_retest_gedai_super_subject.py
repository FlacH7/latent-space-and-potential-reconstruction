#!/usr/bin/env python3
"""
test_iga_from_eeg_latent_test_retest_gedai_super_subject.py
============================================================
Pipeline IgA (Kramers-Moyal + reconstruction via Galerkin-B-spline) for
**super-subject** EEG recordings from the test-retest Gedai dataset
(EEGLAB ``.set``/``.fdt``).

A super-subject is the time-axis concatenation of *N* individual
subjects' EEG recordings sharing the same ``session`` and ``task``.
The concatenated raw is processed **as if it were a single subject's
recording from the start** -- i.e. the full IgA pipeline (Stage 0
load -> 3-stage latent extraction -> KM -> IgA potential reconstruction)
runs once on the concatenated raw, with no per-subject boundary
information leaking downstream.

This is a sibling of
``test_iga_from_eeg_latent_test_retest_gedai.py`` and shares its
CLI/stage API; the only differences are:

* ``--subject`` is replaced by ``--super-subject`` (1-indexed int).
* ``--subject-ids`` (JSON list) optionally overrides the auto-resolved
  subject pool.  Otherwise the pool is built from
  ``--subjects-per-super-subject`` and ``--subject-start-offset``.
* Output directory:
  ``test_retest_gedai_super_subject/super_subject-{id}/{session}/
   {latent_dim}_latent_dim_{spec_label}_{spec_hash}/
   from{t_start}s_to_{t_end}s_{task}``
* Cache directory:
  ``cache_eeg_test_retest_gedai_super_subject/super_subject-{id}/
   {session}/task_{task}_latent_dim_{latent_dim}_{spec_label}_{spec_hash}/
   from{t_start}s_to_{t_end}s.npz``

HIGH-DIMENSIONAL MODE (D >= 4)
------------------------------
When ``analysis_dim >= 4``, the downstream KM/BW/potential pipeline is
**split** into independent sub-jobs of dimension 2 or 3 to avoid the
curse of dimensionality.  The splitting rule is:

* If ``D`` is divisible by 2  -> chunks of 2 (e.g. D=4 -> [[0,1],[2,3]])
* elif ``D`` is divisible by 3 -> chunks of 3 (e.g. D=9 -> three 3D jobs)
* else -> last 3 dims go into a single 3D job + 2D jobs for the rest
  (e.g. D=5 -> [[0,1],[2,3,4]]; D=7 -> [[0,1],[2,3],[4,5,6]])

Each sub-job runs the full BW optimisation + KM extraction + potential
reconstruction + 2D/slice plotting **independently** (in parallel via
``joblib.Parallel(prefer="threads")``).  The per-job figures are then
**composited** into a single horizontal-row PNG per canonical figure
name (``potential_2d.png``, ``bw_optimisation.png``,
``km_components_drift.png``, etc.).  Each subplot is labeled with the
dimensions it represents (``dim1-2``, ``dim3-4``, ``dim3-4-5``...).

Per-job ``potential_data_{dim_label}.npz`` files are saved with the
dim_indices recorded in the metadata, so post-processing knows which
dimensions each potential came from.

Usage
-----
From the project root::

    # Auto-resolve subject pool: super_subject 1 = subjects 1..20
    python -m src.pipelines.test_iga_from_eeg_latent_test_retest_gedai_super_subject \\
        --super-subject 1 --session session1 --task eyesclosed \\
        --t-start 0 --t-end 300 \\
        --stage1-embedding hankel --stage2-dynamics dmd \\
        --stage3-selection top_n

    # Explicit subject list (non-contiguous partition)
    python -m src.pipelines.test_iga_from_eeg_latent_test_retest_gedai_super_subject \\
        --super-subject 2 --session session1 --task eyesclosed \\
        --subject-ids '[21,22,23,...,40]' \\
        --t-start 0 --t-end 300 \\
        --stage1-embedding hankel --stage2-dynamics dmd \\
        --stage3-selection top_n
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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

from src.latent_space_extraction.super_subject_eeg import load_super_subject_eeg

from src.latent_space_extraction.data_analysis_tools import outliers_cleaning

from src.latent_space_extraction.ck_test import chapman_kolmogorov_test

from src.utils.io import save_potential

from src.plotters.trajectory_plots import plot_latent_trajectory

# --- Power-spectral-density analysis module ---
from src.spectral_analysis.psd_analysis import (
    compute_and_plot_raw_psd,
    compute_and_plot_latent_psd,
    compute_channel_influence_on_latent,
)

from src.utils.config import (
    CONFIGS,
    BASE_RESULTS_PATH,
    BASE_CACHE_PATH,
    DB_TEST_RETEST_GEDAI_PATH,
)

# --- Plotting module (src.plotters) ---
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
        description=(
            "KM + IgA pipeline for SUPER-SUBJECT test-retest EEG "
            "(concatenation of N subjects) preprocessed with Gedai "
            "(EEGLAB .set/.fdt)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- Super-subject source ----
    parser.add_argument(
        "--super-subject", type=int, required=True,
        help="1-indexed super-subject identifier (1, 2, 3, ...).",
    )
    parser.add_argument(
        "--subject-ids", type=str, default=None,
        help=(
            "JSON list of subject indices that compose the super-subject, "
            'e.g. \'[1,2,...,20]\'. When omitted, the pool is built from '
            "--subjects-per-super-subject and --subject-start-offset."
        ),
    )
    parser.add_argument(
        "--subjects-per-super-subject", type=int, default=20,
        help="Subjects per super-subject (auto-resolution). Default: 20.",
    )
    parser.add_argument(
        "--subject-start-offset", type=int, default=1,
        help=(
            "Index of the first subject in the dataset (typically 1 for "
            "sub-01).  Used by auto-resolution. Default: 1."
        ),
    )
    # ---- Session / task ----
    parser.add_argument(
        "--session", type=str, required=True,
        help="Session ID (e.g. session1)",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        help="Task label (eyesclosed, eyesopen, mathematic, memory, music)",
    )
    parser.add_argument(
        "--channel-intersection", type=str, default=None,
        help=(
            "JSON list of channel names representing the global channel "
            "intersection across all super-subjects. When provided, the "
            "concatenated raw is restricted to exactly these channels before "
            "any processing. This ensures CD-HSA mode dimensionality "
            "consistency across super-subjects with different channel counts."
        ),
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Override Gedai dataset root path",
    )
    parser.add_argument(
        "--t-start", type=float, default=None,
        help="Per-subject start time (s) of the EEG segment to analyze.",
    )
    parser.add_argument(
        "--t-end", type=float, default=None,
        help="Per-subject end time (s) of the EEG segment to analyze.",
    )
    # ---- Cache / persistence ----
    parser.add_argument(
        "--cache-file", type=str, default=None,
        help="Path to cache file for the latent space.",
    )
    parser.add_argument(
        "--ignore-cache", action="store_true",
        help="Ignore an existing cache file and force recomputation.",
    )
    # ---- Latent-space extraction params ----
    parser.add_argument(
        "--latent-dim", type=int, default=2,
        help="Dimensionality of the latent subspace (default: 2)",
    )
    # ---- NEW 3-stage pipeline API ----
    parser.add_argument(
        "--stage1-embedding", type=str, default=None,
        choices=["none", "hankel"],
        help="Stage 1 embedding. If omitted, the legacy "
             "--scoring-method mapping is used.",
    )
    parser.add_argument(
        "--stage1-params", type=str, default=None,
        help='JSON dict of Stage-1 params, e.g. \'{"depth": 250}\'',
    )
    parser.add_argument(
        "--stage2-dynamics", type=str, default=None,
        choices=["pca_ica", "pca", "dmd", "diffusion_maps", "cdhsa_specific_modes"],
        help="Stage 2 dynamics (new API).",
    )
    parser.add_argument(
        "--stage2-params", type=str, default=None,
        help='JSON dict of Stage-2 params, e.g. \'{"svd_rank": 50}\'',
    )
    parser.add_argument(
        "--stage3-selection", type=str, default=None,
        choices=["top_n", "markov_fastest", "markov_slowest"],
        help="Stage 3 selection (new API).",
    )
    parser.add_argument(
        "--stage3-params", type=str, default=None,
        help='JSON dict of Stage-3 params, e.g. \'{"n_bins": 10}\'',
    )
    # ---- Legacy scoring API (mapped onto the new stages) ----
    parser.add_argument(
        "--scoring-method", type=str, default="hankel_dmd",
        choices=["markov", "markov_inverted", "conservative", "weighted",
                 "sequential", "pareto", "independent",
                 "hankel_dmd", "diffusion_maps"],
        help="Legacy subspace selection strategy (default: hankel_dmd). "
             "Mapped internally onto the new 3-stage API. Ignored if "
             "any --stageX argument is given.",
    )
    parser.add_argument(
        "--fc-metric", type=str, default="variance_sum",
        choices=["variance_sum", "first_pc_var", "total_variance"],
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Quantile bins for Markov discretisation (default: 10)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel processes for subspace search",
    )
    parser.add_argument(
        "--search-strategy", type=str, default="exhaustive",
        choices=["exhaustive", "greedy"],
    )
    # ---- Which latent dimension(s) to feed into KM ----
    parser.add_argument(
        "--analysis-dim", type=int, default=None,
        help="Number of latent dimensions to use for KM. "
             "If None, uses all extracted dimensions.",
    )
    parser.add_argument(
        "--column", type=int, default=None,
        help="If analysis-dim=1, which latent column to use (0=first).",
    )
    # ---- Preprocessing params ----
    parser.add_argument("--l-freq", type=float, default=1.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--ica-method", type=str, default="picard")
    parser.add_argument("--verbose", action="store_true", default=True)
    # ---- Hankel (legacy convenience; also usable as stage1 param) ----
    parser.add_argument(
        "--hankel-embedding-depth", type=int, default=None,
        help="Hankel embedding depth T. None = auto "
             "(clip(sfreq*0.25, 50, 200)).",
    )
    # ---- Diffusion Maps params ----
    parser.add_argument(
        "--diffusion-sigma", type=float, default=None,
        help=("Sigma for Diffusion Maps Gaussian kernel. "
              "If None, auto-computed via bgh method (Berry-Giannakis-Harlim)."),
    )
    parser.add_argument(
        "--diffusion-k", type=int, default=100,
        help="Number of nearest neighbors for sparse affinity matrix.",
    )
    parser.add_argument(
        "--diffusion-time", type=float, default=0.0,
        help=("Diffusion time t >= 0. Higher values filter fine-scale noise "
              "and highlight macroscopic dynamics (multiscale filtering)."),
    )
    parser.add_argument(
        "--diffusion-alpha", type=float, default=0.5,
        help=("Density normalization parameter (Coifman-Lafon). "
              "0.0=Laplacian Eigenmaps, 0.5=Diffusion Maps (default), 1.0=Fokker-Planck."),
    )
    # ---- Output ----
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to save plots (default: current)",
    )
    # ---- Kramers-Moyal bins ----
    parser.add_argument(
        "--km-bins", type=int, default=40,
        help="Number of bins per dimension for KM estimation (default: 40)",
    )
    # ---- Save potential ----
    parser.add_argument(
        "--save-potential", action="store_true", default=True,
        help="Guardar datos del potencial en .npz para post-procesamiento",
    )
    parser.add_argument(
        "--no-save-potential", action="store_true",
        help="Deshabilitar el guardado del potencial",
    )
    # ---- High-dimensional split control ----
    parser.add_argument(
        "--split-threshold", type=int, default=4,
        help="Analysis-dim threshold above which the pipeline splits into "
             "sub-jobs of dim 2 or 3 (default: 4). Set to a large value "
             "(e.g. 99) to disable splitting entirely.",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Run sub-jobs sequentially instead of via joblib.Parallel. "
             "Useful for debugging.",
    )

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


def _json_int_list(raw: str | None) -> list[int] | None:
    """Parse a JSON list of integers from the CLI (None when omitted)."""
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError(f"subject-ids must be a JSON list of ints, got: {raw!r}")
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

    Returns a dict with keys ``stage1_embedding``, ``stage1_params``,
    ``stage2_dynamics``, ``stage2_params``, ``stage3_selection``,
    ``stage3_params``.
    """
    use_new_api = any([
        args.stage1_embedding is not None,
        args.stage2_dynamics is not None,
        args.stage3_selection is not None,
    ])

    if use_new_api:
        s1 = None if args.stage1_embedding in (None, "none") else args.stage1_embedding
        s2 = args.stage2_dynamics or "pca_ica"
        s3 = args.stage3_selection or "top_n"

        p1 = _json_params(args.stage1_params)
        p2 = _json_params(args.stage2_params)
        p3 = _json_params(args.stage3_params)

        # Inject CLI convenience flags when not overridden in the JSON
        if s1 == "hankel":
            p1.setdefault("depth", args.hankel_embedding_depth)
        if s2 == "diffusion_maps":
            p2.setdefault("sigma", args.diffusion_sigma)
            p2.setdefault("k", args.diffusion_k)
            p2.setdefault("diffusion_time", args.diffusion_time)
            p2.setdefault("alpha", args.diffusion_alpha)
        if s2 == "pca_ica":
            p2.setdefault("ica_method", args.ica_method)
        if s2 == "cdhsa_specific_modes":
            # Auto-inject --task as condition so the mode_map resolves
            # the correct condition index without the user specifying it.
            p2.setdefault("condition", args.task)
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
    return map_legacy_scoring_method(
        args.scoring_method,
        n_dim=args.latent_dim,
        fc_metric=args.fc_metric,
        n_bins=args.n_bins,
        hankel_embedding_depth=args.hankel_embedding_depth,
        diffusion_sigma=args.diffusion_sigma,
        diffusion_k=args.diffusion_k,
        diffusion_time=args.diffusion_time,
        diffusion_alpha=args.diffusion_alpha,
        ica_method=args.ica_method,
        search_strategy=args.search_strategy,
    )


def _spec_label(spec: dict) -> str:
    """Short human-readable label for the stage chain (used in paths)."""
    s1 = spec["stage1_embedding"] or "none"
    return f"{s1}+{spec['stage2_dynamics']}+{spec['stage3_selection']}"


def _spec_hash(spec: dict) -> str:
    """
    Short hash of the full stage spec (stages + params).

    Included in the cache key so that different parameter combinations
    never collide (requirement: the cache key must contain the hashes of
    the 3-stage parameters).
    """
    payload = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Super-subject label helpers
# ---------------------------------------------------------------------------

def _super_subject_label(super_subject_id: int) -> str:
    """BIDS-style label for the super-subject, e.g. ``super_subject-01``."""
    return f"super_subject-{super_subject_id:02d}"


# ---------------------------------------------------------------------------
# HIGH-DIMENSIONAL SPLIT HELPERS
# ---------------------------------------------------------------------------

def _split_dimensions(D: int) -> list[list[int]]:
    """
    Partition ``D`` latent dimensions into independent sub-jobs of dim 2 or 3.

    Rules (in priority order):
      1. If ``D`` is divisible by 2  -> chunks of 2.
      2. elif ``D`` is divisible by 3 -> chunks of 3.
      3. else -> the LAST 3 dims go in a single 3D chunk; the remaining
         dims (which are guaranteed even) are split into 2D chunks.

    For ``D <= 3`` there is no real split -- the function returns a single
    chunk covering all dimensions.

    Examples
    --------
    >>> _split_dimensions(1)
    [[0]]
    >>> _split_dimensions(2)
    [[0, 1]]
    >>> _split_dimensions(3)
    [[0, 1, 2]]
    >>> _split_dimensions(4)
    [[0, 1], [2, 3]]
    >>> _split_dimensions(5)
    [[0, 1], [2, 3, 4]]
    >>> _split_dimensions(6)
    [[0, 1], [2, 3], [4, 5]]
    >>> _split_dimensions(7)
    [[0, 1], [2, 3], [4, 5, 6]]
    >>> _split_dimensions(9)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> _split_dimensions(11)
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10]]
    """
    if D <= 0:
        raise ValueError(f"D must be positive, got D={D}")
    if D <= 3:
        return [list(range(D))]
    if D % 2 == 0:
        return [list(range(i, i + 2)) for i in range(0, D, 2)]
    if D % 3 == 0:
        return [list(range(i, i + 3)) for i in range(0, D, 3)]
    # D not divisible by 2 nor 3 -> last 3 dims as one 3D chunk + 2D chunks
    # for the (even-length) remainder.
    last_3 = list(range(D - 3, D))
    rest = list(range(0, D - 3))
    chunks = [rest[i:i + 2] for i in range(0, len(rest), 2)]
    chunks.append(last_3)
    return chunks


def _dim_label(indices: list[int]) -> str:
    """
    Human-readable label for a sub-job, 1-indexed to match the user's
    "dim1-2" convention.

    Examples
    --------
    >>> _dim_label([0, 1])
    'dim1-2'
    >>> _dim_label([2, 3])
    'dim3-4'
    >>> _dim_label([2, 3, 4])
    'dim3-4-5'
    """
    return "dim" + "-".join(str(i + 1) for i in indices)


def _build_job_config(
    base_config: dict,
    d_job: int,
    args: argparse.Namespace,
    latent_dim: int,
    ss_label: str,
    session: str,
    task: str,
    spec_label: str,
) -> dict:
    """
    Build a per-job KM config dict.  Mirrors the original "CONFIGURATION"
    block but parameterised by the sub-job dimensionality ``d_job``.
    """
    config = base_config.copy()
    config.update({
        "model_name": (
            f"testretest_gedai_super_subject_{ss_label}_"
            f"{session}_{task}_d{latent_dim}_{spec_label}"
        ),
        "D": d_job,
        "bins": [args.km_bins] * d_job,
        "drift_components": list(range(d_job)),
        "diff_components": [(i, i) for i in range(d_job)],
        "degree": 2,
    })
    return config


# ---------------------------------------------------------------------------
# PER-JOB DOWNSTREAM PIPELINE
# ---------------------------------------------------------------------------

# Canonical figure names that may be produced by a single sub-job.
# Used by the compositor to know which PNGs to look for in each per-job dir.
_CANONICAL_FIGURE_NAMES_D2 = [
    "bw_optimisation",
    "km_components_drift",
    "km_components_diffusion",
    "potential_1d",
    "potential_2d",
    "potential_2d_streamlines",
    "potential_2d_residual",
    "potential_2d_nonconservative_force",
    "potential_2d_combined",
]
_CANONICAL_FIGURE_NAMES_D3 = [
    "bw_optimisation",
    "km_components_drift",
    "km_components_diffusion",
    "potential_1d",
    "potential_slice_x1_x2",
    "potential_slice_x1_x3",
    "potential_slice_x2_x3",
]


def _run_km_job(
    data: np.ndarray,
    dt: float,
    config: dict,
    dim_indices: list[int],
    dim_label: str,
    out_dir: Path,
    save_potential_flag: bool,
    metadata: dict,
    save_potential_filename: str,
) -> dict:
    """
    Run the full downstream pipeline (BW + KM + potential + plots) for ONE
    sub-series of dimension ``d_job = len(dim_indices)``.

    All figures are saved to ``out_dir`` with their canonical names (no
    per-job suffix on the filename -- the per-job isolation comes from
    ``out_dir`` being a per-job subdir).

    Returns a metadata dict useful for the compositor:
      ``{dim_indices, dim_label, D, out_dir, bw_opt, drift, diffusion,
         edges, density, result_iga, config, save_potential_filename}``
    """
    d_job = data.shape[1]
    model_name = config["model_name"]
    print(f"\n  [{dim_label}] Running KM job: D={d_job}, "
          f"data shape={data.shape}, out_dir={out_dir}")
    sys.stdout.flush()

    # Each worker process has its own matplotlib state.  Ensure the
    # plotting style is applied (idempotent; cheap).
    try:
        setup_plotting_style()
    except Exception:
        pass

    # =====================================================================
    # 5. BANDWIDTH OPTIMISATION  (always run, d_job <= 3)
    # =====================================================================
    print(f"  [{dim_label}] Bandwidth optimisation ...")
    sys.stdout.flush()
    # n_jobs=1 inside optimal_bw to avoid nested parallelism; the outer
    # joblib.Parallel handles parallelism across sub-jobs.
    result_bw = optimal_bw(
        data=data,
        bins=config["bins"],
        dt=dt,
        p=2,
        kernel="epanechnikov",
        theoretical=None,
        sigma_smooth=1.0,
        n_candidates=30,
        n_jobs=1,
        plot=True,
        auto_weight=True,
    )
    bw_opt = result_bw["optimal_bw"]
    sys.stdout.flush()
    print(f"  [{dim_label}] Optimal bandwidth: {bw_opt:.4f}")
    print(f"  [{dim_label}] Drift error: "
          f"{result_bw['error_drift'][result_bw['optimal_idx']]:.4f}")
    print(f"  [{dim_label}] Diffusion error: "
          f"{result_bw['error_diff'][result_bw['optimal_idx']]:.4f}")

    if "fig" in result_bw and result_bw["fig"] is not None:
        fig_bw = result_bw["fig"]
        # Add dim_label as suptitle to make the figure self-identifying
        try:
            fig_bw.suptitle(f"BW optimisation -- {dim_label}", fontsize=12)
        except Exception:
            pass
        fig_bw.savefig(out_dir / "bw_optimisation.png", dpi=150, bbox_inches="tight")
        plt.close(fig_bw)

    # =====================================================================
    # 6. ESTIMATE KM COEFFICIENTS
    # =====================================================================
    print(f"  [{dim_label}] KM coefficient estimation ...")
    sys.stdout.flush()

    drift, diffusion, edges = extract_km_coefficients(
        data,
        bins=config["bins"],
        p=2,
        bw=bw_opt,
        kernel="epanechnikov",
        dt=dt,
        sigma_smooth=1.0,
        density_threshold=0.01,
    )
    print(f"  [{dim_label}] Drift shape    : {drift.shape}")
    print(f"  [{dim_label}] Diffusion shape: {diffusion.shape}")
    print(f"  [{dim_label}] Bins           : {[len(e) for e in edges]}")

    # =====================================================================
    # 7. EMPIRICAL DENSITY
    # =====================================================================
    hist_edges = []
    for d in range(d_job):
        c = edges[d]
        half = (c[1] - c[0]) / 2.0 if len(c) > 1 else 0.5
        e = np.concatenate([[c[0] - half], c + half])
        hist_edges.append(e)

    density, _ = np.histogramdd(data, bins=hist_edges)
    density = density.astype(float)
    print(f"  [{dim_label}] Density shape: {density.shape}")

    # =====================================================================
    # 8. PLOT KM COMPONENTS
    # =====================================================================
    print(f"  [{dim_label}] Plot KM components ...")
    figs_km = plot_km_components(
        drift, diffusion, edges,
        drift_components=config["drift_components"],
        diff_components=config["diff_components"],
        fixed_coords=None,
        theoretical=None,
        figsize=(12, 8),
    )
    for key, fname in [("drift", "km_components_drift.png"),
                       ("diffusion", "km_components_diffusion.png")]:
        fig_km = figs_km.get(key)
        if fig_km is None:
            continue
        try:
            fig_km.suptitle(
                f"KM {key} -- {dim_label} -- {model_name}",
                fontsize=13,
            )
            fig_km.tight_layout(rect=[0, 0, 1, 0.95])
        except Exception:
            pass
        fig_km.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig_km)
    print(f"  [{dim_label}] Saved km_components_drift.png, "
          f"km_components_diffusion.png")

    # =====================================================================
    # 9. 1D POTENTIAL RECONSTRUCTION
    # =====================================================================
    print(f"  [{dim_label}] 1D potential reconstruction ...")
    U_rec_1d = reconstruct_potential_1D(
        drift, diffusion, edges,
        density=density,
        degree=config["degree"],
    )

    fig_pot_1d = plot_potential(
        U_rec_1d, edges,
        theoretical=None,
        component_labels=[f"x{j + 1}" for j in range(d_job)],
        figsize=(5 * d_job, 4),
    )
    try:
        fig_pot_1d.suptitle(
            f"Reconstructed 1D potential (IgA) -- {dim_label} -- "
            f"{model_name.upper()}",
            fontsize=12,
        )
        fig_pot_1d.tight_layout(rect=[0, 0, 1, 0.95])
    except Exception:
        pass
    fig_pot_1d.savefig(out_dir / "potential_1d.png", dpi=150, bbox_inches="tight")
    plt.close(fig_pot_1d)
    print(f"  [{dim_label}] Saved potential_1d.png")

    # =====================================================================
    # 10. FULL MULTIDIMENSIONAL POTENTIAL RECONSTRUCTION
    # =====================================================================
    print(f"  [{dim_label}] Full multidim potential reconstruction ...")
    sys.stdout.flush()
    result_iga = reconstruct_potential(
        drift, diffusion, edges,
        density=density,
        method="iga",
        return_full=True,
        degree=config["degree"],
        decompose_helmholtz=True,
        density_threshold=0.01,
        rtol=1e-3,
        atol=1e-12,
        compute_stream_function=True,
    )

    U_rec = result_iga["potential"]
    print(f"  [{dim_label}] Reconstructed potential shape: {U_rec.shape}")
    print(f"  [{dim_label}] Method: Galerkin-B-spline (degree={config['degree']})")

    # Quality metrics
    rank_D = result_iga["rank_D"]
    condition_D = result_iga["condition_D"]
    frac_low_rank = (
        np.mean(rank_D < d_job) if np.any(~np.isnan(rank_D)) else 0.0
    )
    print(f"  [{dim_label}] Fraction cells with rank_D < {d_job}: "
          f"{frac_low_rank:.2%}")
    print(f"  [{dim_label}] Condition D (median): "
          f"{np.nanmedian(condition_D):.2e}")
    print(f"  [{dim_label}] Condition D (max)   : "
          f"{np.nanmax(condition_D):.2e}")

    if "eta" in result_iga:
        eta = result_iga["eta"]
        is_eq = result_iga["is_equilibrium"]
        print(f"  [{dim_label}] Non-equilibrium metric eta: {eta:.4f}")
        print(f"  [{dim_label}] Detailed balance? {is_eq} "
              f"(threshold eta < 0.1)")

    # =====================================================================
    # 10b. SAVE POTENTIAL DATA FOR POST-PROCESSING
    # =====================================================================
    if save_potential_flag:
        print(f"  [{dim_label}] Saving potential data ...")
        # Augment metadata with dim-specific info.
        meta_with_dims = dict(metadata)
        meta_with_dims["dim_indices"] = list(dim_indices)
        meta_with_dims["dim_label"] = dim_label
        meta_with_dims["job_D"] = d_job
        potential_path = save_potential(
            result_iga=result_iga,
            edges=edges,
            density=density,
            config=config,
            out_dir=out_dir,
            metadata=meta_with_dims,
            filename=save_potential_filename,
        )
        print(f"  [{dim_label}] Potential data saved: {potential_path}")

    # =====================================================================
    # 11. PLOT POTENTIAL 2D / SLICES
    # =====================================================================
    print(f"  [{dim_label}] Plot potential 2D / slices ...")
    if d_job == 2:
        # --- potential_2d.png (without streamlines) ---
        fig_pot_2d = plot_potential_2d(
            U_rec, edges,
            title_est=f"{model_name.upper()} -- Reconstructed -- {dim_label}",
            figsize=(12, 5),
            unify_colorbar=True,
            align_minima=True,
            align_to_zero=True,
            crop_to_valid=True,
            show_streamlines=False,
            stream_function=result_iga.get('stream_function'),
            reconstructed_field=result_iga.get('reconstructed_field'),
        )
        if fig_pot_2d is None:
            fig_pot_2d = plt.gcf()
        fig_pot_2d.savefig(out_dir / "potential_2d.png", dpi=150,
                            bbox_inches="tight")
        plt.close(fig_pot_2d)
        print(f"  [{dim_label}] Saved potential_2d.png")

        # --- potential_2d_streamlines.png ---
        fig_pot_2d_sl = plot_potential_2d(
            U_rec, edges,
            title_est=f"{model_name.upper()} -- Reconstructed -- {dim_label}",
            figsize=(12, 5),
            unify_colorbar=True,
            align_minima=True,
            align_to_zero=True,
            crop_to_valid=True,
            stream_function=result_iga.get('stream_function'),
            reconstructed_field=result_iga.get('reconstructed_field'),
        )
        if fig_pot_2d_sl is None:
            fig_pot_2d_sl = plt.gcf()
        fig_pot_2d_sl.savefig(out_dir / "potential_2d_streamlines.png",
                              dpi=150, bbox_inches="tight")
        plt.close(fig_pot_2d_sl)
        print(f"  [{dim_label}] Saved potential_2d_streamlines.png")

        # --- potential_2d_residual.png ---
        if "residual" in result_iga:
            residual = result_iga["residual"]
            r_norm = np.sqrt(np.nansum(residual**2, axis=0))
            x_c, y_c = edges
            X, Y = np.meshgrid(x_c, y_c, indexing="ij")
            fig_res, ax_res = plt.subplots(figsize=(6, 5))
            cs = ax_res.contourf(
                X, Y, np.nan_to_num(r_norm, nan=0),
                levels=20, cmap="magma",
            )
            # Capture the return value of contourf (a QuadContourSet,
            # NOT an AxesImage) so colorbar works without relying on
            # ax.images which would be empty for contourf.
            fig_res.colorbar(cs, ax=ax_res, label="||r||")
            ax_res.set_xlabel("x")
            ax_res.set_ylabel("y")
            ax_res.set_title(
                f"Residuo Helmholtz -- {dim_label} -- {model_name.upper()}"
            )
            ax_res.axis("equal")
            fig_res.tight_layout()
            fig_res.savefig(out_dir / "potential_2d_residual.png",
                            dpi=150, bbox_inches="tight")
            plt.close(fig_res)
            print(f"  [{dim_label}] Saved potential_2d_residual.png")

        # --- potential_2d_nonconservative_force.png ---
        if result_iga.get('nonconservative_force') is not None:
            fig_v = plot_nonconservative_force_2d(
                result_iga['nonconservative_force'], edges,
                title=f"Fuerza no-conservativa v -- {dim_label} -- "
                      f"{model_name.upper()}",
                crop_to_valid=True,
            )
            if fig_v is not None:
                fig_v.savefig(out_dir / "potential_2d_nonconservative_force.png",
                              dpi=150, bbox_inches="tight")
                plt.close(fig_v)
                print(f"  [{dim_label}] Saved potential_2d_nonconservative_force.png")

        # --- potential_2d_combined.png ---
        if (result_iga.get('reconstructed_field') is not None
                or result_iga.get('nonconservative_force') is not None):
            fig_comb = plot_potential_combined_2d(
                U_rec, edges,
                stream_function=result_iga.get('stream_function'),
                nonconservative_force=result_iga.get('nonconservative_force'),
                reconstructed_field=result_iga.get('reconstructed_field'),
                title=f"{model_name.upper()} -- {dim_label} -- "
                      f"U + v + streamlines",
                crop_to_valid=True,
            )
            if fig_comb is not None:
                fig_comb.savefig(out_dir / "potential_2d_combined.png",
                                 dpi=150, bbox_inches="tight")
                plt.close(fig_comb)
                print(f"  [{dim_label}] Saved potential_2d_combined.png")

    elif d_job == 3:
        # 3 slices: (0,1), (0,2), (1,2)
        dim_pairs = [(0, 1), (0, 2), (1, 2)]
        for dims in dim_pairs:
            fig_slice = plot_potential_slice(
                U_rec, edges, dims=dims,
                fixed_coords={d: 0.0 for d in range(d_job) if d not in dims},
                title=f"{model_name.upper()} -- {dim_label} -- "
                      f"slice x{dims[0]+1}-x{dims[1]+1}",
                figsize=(12, 5),
                unify_colorbar=True,
                align_minima=True,
                align_to_zero=True,
                crop_to_valid=True,
            )
            if fig_slice is None:
                fig_slice = plt.gcf()
            fname = f"potential_slice_x{dims[0]+1}_x{dims[1]+1}.png"
            fig_slice.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig_slice)
            print(f"  [{dim_label}] Saved {fname}")
    else:
        print(f"  [{dim_label}] (No 2D/slice plots for d_job={d_job})")

    sys.stdout.flush()
    return {
        "dim_indices": list(dim_indices),
        "dim_label": dim_label,
        "D": d_job,
        "out_dir": out_dir,
        "bw_opt": bw_opt,
        "drift": drift,
        "diffusion": diffusion,
        "edges": edges,
        "density": density,
        "result_iga": result_iga,
        "config": config,
        "save_potential_filename": save_potential_filename,
    }


# ---------------------------------------------------------------------------
# COMPOSITE FIGURE BUILDER
# ---------------------------------------------------------------------------

def _compose_horizontal(
    image_paths: list[Path],
    labels: list[str],
    out_path: Path,
    dpi: int = 150,
    title: str | None = None,
) -> Path | None:
    """
    Composite N PNG images into a single horizontal-row image (1 row x N
    columns).  Each subplot is labeled with the corresponding ``labels``
    entry (typically the dim_label of the sub-job).

    Returns the output path, or ``None`` if no images were provided.
    """
    n = len(image_paths)
    if n == 0:
        return None

    # Read images
    images = []
    for p in image_paths:
        try:
            im = plt.imread(str(p))
            images.append(im)
        except Exception as e:
            print(f"  [compose] WARN: could not read {p}: {e}")

    if not images:
        return None

    # Single image: just copy
    if len(images) == 1:
        shutil.copy(str(image_paths[0]), str(out_path))
        print(f"  [compose] Single image -> {out_path.name}")
        return out_path

    # Determine per-subplot width based on image aspect ratios
    # Each subplot gets equal width; subplot height = max image height
    sub_w = 7  # inches per subplot
    max_h_in = max(im.shape[0] / dpi for im in images)
    fig_h = max_h_in + 1.0   # +1 inch for title
    fig_w = sub_w * len(images)

    fig, axes = plt.subplots(
        1, len(images),
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    for ax, im, label in zip(axes[0], images, labels):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(label, fontsize=12, pad=8)

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    # Use constrained_layout-friendly approach
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    except Exception:
        pass

    fig.savefig(str(out_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [compose] {len(images)} images -> {out_path.name}")
    return out_path


def _composite_all_figures(
    job_results: list[dict],
    out_dir: Path,
) -> None:
    """
    For every canonical figure name, gather the per-job PNGs and composite
    them into a single horizontal-row image saved to ``out_dir`` with the
    canonical name.

    Per-job figures that don't exist for some jobs are simply skipped
    (the composite will only include the jobs that produced that figure).
    """
    if not job_results:
        return

    # Union of all canonical figure names that any job might produce.
    all_fig_names = sorted(
        set(_CANONICAL_FIGURE_NAMES_D2) | set(_CANONICAL_FIGURE_NAMES_D3)
    )

    for fig_name in all_fig_names:
        pairs = []  # list of (path, label)
        for jr in job_results:
            p = jr["out_dir"] / f"{fig_name}.png"
            if p.exists():
                pairs.append((p, jr["dim_label"]))

        if not pairs:
            continue

        out_path = out_dir / f"{fig_name}.png"
        _compose_horizontal(
            image_paths=[p for p, _ in pairs],
            labels=[lbl for _, lbl in pairs],
            out_path=out_path,
            title=fig_name.replace("_", " ").title(),
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    if args.out_dir is None:
        args.out_dir = BASE_RESULTS_PATH
    verbose = "INFO" if args.verbose else None

    save_potential_flag = args.save_potential and not args.no_save_potential

    # Defaults
    if args.t_start is None:
        args.t_start = 0.0
    if args.t_end is None:
        args.t_end = 60.0

    db_path = args.db_path or DB_TEST_RETEST_GEDAI_PATH

    # Resolve the effective 3-stage spec (new API or legacy mapping)
    spec = _resolve_pipeline_spec(args)
    spec_label = _spec_label(spec)
    spec_hash = _spec_hash(spec)
    print(f"\n  Pipeline spec : {spec_label}  (hash {spec_hash})")
    print(f"  Stage params  : {json.dumps(spec, default=str)}")

    # Resolve subject pool for the super-subject
    subject_ids = _json_int_list(args.subject_ids)
    ss_label = _super_subject_label(args.super_subject)
    print(f"  Super-subject : {ss_label}")
    if subject_ids is not None:
        print(f"  Subject pool  : explicit list ({len(subject_ids)} subjects)")
    else:
        print(
            f"  Subject pool  : auto-resolved "
            f"({args.subjects_per_super_subject} subjects starting at "
            f"{args.subject_start_offset})"
        )

    # -----------------------------------------------------------------
    # Output directory:
    # test_retest_gedai_super_subject/super_subject-{id}/{session}/
    #   {latent_dim}_latent_dim_{spec_label}_{spec_hash}/
    #   from{t_start}s_to_{t_end}s_{task}
    # -----------------------------------------------------------------
    out_dir = Path(
        str(args.out_dir)
        + f"/test_retest_gedai_super_subject/{ss_label}/{args.session}"
        + f"/{args.latent_dim}_latent_dim_{spec_label}_{spec_hash}"
        + f"/from{args.t_start}s_to_{args.t_end}s_{args.task}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()

    # =====================================================================
    # 1. LOAD SUPER-SUBJECT EEG SEGMENT (concatenated raw)
    # =====================================================================
    print("=" * 70)
    print("  STAGE 0: LOAD SUPER-SUBJECT EEG SEGMENT (GEDAI, concatenated)")
    print("=" * 70)

    try:
        print(f"  Loading super-subject EEG ({len(subject_ids) if subject_ids else args.subjects_per_super_subject} subjects)...")
        sys.stdout.flush()
        raw = load_super_subject_eeg(
            super_subject_id=args.super_subject,
            session=args.session,
            task=args.task,
            subject_ids=subject_ids,
            subjects_per_super_subject=args.subjects_per_super_subject,
            subject_start_offset=args.subject_start_offset,
            db_path=db_path,
            t_start=args.t_start,
            t_stop=args.t_end,
            preload=False,
            verbose=verbose,
        )
        sys.stdout.flush()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    sfreq = raw.info["sfreq"]
    print(f"  Super-subject ID   : {args.super_subject} ({ss_label})")
    print(f"  Channels (common)  : {len(raw.ch_names)}")
    print(f"  Channel names      : {raw.ch_names}")
    print(f"  Sampling freq      : {sfreq:.2f} Hz")
    print(f"  Cropped window     : {raw.times[0]:.2f} s -> {raw.times[-1]:.2f} s "
          f"(duration: {raw.times[-1] - raw.times[0]:.2f} s)")

    # -----------------------------------------------------------------
    # Apply global channel intersection (CD-HSA consistency)
    # -----------------------------------------------------------------
    channel_intersection = None
    if args.channel_intersection is not None:
        channel_intersection = json.loads(args.channel_intersection)
        if not isinstance(channel_intersection, list):
            raise ValueError("--channel-intersection must be a JSON list of channel names")
        # Validate that all requested channels exist in the raw
        missing = [ch for ch in channel_intersection if ch not in raw.ch_names]
        if missing:
            raise ValueError(
                f"--channel-intersection references {len(missing)} channels "
                f"not found in raw: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        n_before = len(raw.ch_names)
        raw.pick(channel_intersection)
        print(f"  [ChannelIntersection] Applied global intersection: "
              f"{n_before} -> {len(raw.ch_names)} channels")
        sys.stdout.flush()

    # -----------------------------------------------------------------
    # PSD of the original (concatenated) EEG channels
    # -----------------------------------------------------------------
    psds_raw, freqs_raw, ch_names, mean_psd, std_psd, raw_psd_path = compute_and_plot_raw_psd(
        raw, out_dir=out_dir, fmin=args.l_freq, fmax=args.h_freq, bandwidth=2.5,
    )

    # --- STAGE 0: EXPLORATORY PLOTS ---
    print("  Generando plots exploratorios (Stage 0)...")
    try:
        setup_plotting_style()
        plot_channel_topographies(
            raw, out_dir=out_dir,
            fmin=args.l_freq, fmax=args.h_freq,
            subject=ss_label, session=args.session, task=args.task,
        )
        plot_channel_correlation_matrix(
            raw, out_dir=out_dir,
            subject=ss_label, session=args.session, task=args.task,
        )
        print("  [OK] Plots exploratorios guardados.")
    except Exception as e:
        print(f"  [WARN] Error en plots exploratorios: {e}")

    # =====================================================================
    # 2. EXTRACT (or LOAD CACHED) LATENT SUBSPACE FROM CONCATENATED EEG
    # =====================================================================
    print("\n" + "=" * 70)
    print("  STAGE 1: EXTRACT LATENT SUBSPACE FROM SUPER-SUBJECT EEG")
    print("=" * 70)

    if args.cache_file is None:
        args.cache_file = Path(
            str(BASE_CACHE_PATH)
            + f"/cache_eeg_test_retest_gedai_super_subject/{ss_label}/{args.session}"
            + f"/task_{args.task}_latent_dim_{args.latent_dim}_{spec_label}_{spec_hash}"
            + f"/from{args.t_start}s_to_{args.t_end}s.npz"
        )
    cache_path = Path(args.cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load from cache
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

    # Compute if no cache or --ignore-cache
    if not cache_path.exists() or args.ignore_cache:
        if args.ignore_cache and cache_path.exists():
            print("\n  [CACHE] --ignore-cache set. Recomputing latent space...")

        print("\n  [INFO] Starting latent space extraction (this may take several minutes)...")
        sys.stdout.flush()

        latent, meta = extract_latent_space(
            raw,
            n_dim=args.latent_dim,
            stage1_embedding=spec["stage1_embedding"],
            stage1_params=spec["stage1_params"],
            stage2_dynamics=spec["stage2_dynamics"],
            stage2_params=spec["stage2_params"],
            stage3_selection=spec["stage3_selection"],
            stage3_params=spec["stage3_params"],
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            n_workers=args.workers,
            verbose=verbose,
            channel_intersection=channel_intersection,
        )
        sys.stdout.flush()

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
    # -----------------------------------------------------------------
    print("  Generando plots de dinamica (Stage 2)...")
    try:
        stage2_meta = meta["stage2"]
        # PCA variance (para pca_ica rama MNE)
        plot_pca_variance_explained(stage2_meta, out_dir=out_dir)
        # ICLabel summary
        plot_icalabel_summary(stage2_meta, out_dir=out_dir)
        # Pre/post ICA PSDs (solo para rama MNE)
        plot_pre_ica_component_psds(
            raw, stage2_meta, out_dir=out_dir,
            fmin=args.l_freq, fmax=args.h_freq,
        )
        plot_post_ica_component_psds(
            stage2_meta, sfreq=sfreq, out_dir=out_dir,
            fmin=args.l_freq, fmax=args.h_freq,
        )
        # Hankel PCA singular values (solo para pca_ica rama Hankel)
        plot_hankel_pca_singular_values(stage2_meta, out_dir=out_dir)
        # FastICA convergence
        plot_fastica_convergence(stage2_meta, out_dir=out_dir)
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
        # CD-HSA Specific Modes
        if stage2_meta.get("dynamics") == "cdhsa_specific_modes":
            try:
                from src.plotters.cdhsa_plots import (
                    plot_cdhsa_eigenvalue_spectrum,
                    plot_cdhsa_mode_structure,
                    plot_cdhsa_projection_power,
                )
                plot_cdhsa_eigenvalue_spectrum(stage2_meta, out_dir=out_dir)
                plot_cdhsa_mode_structure(stage2_meta, out_dir=out_dir)
                plot_cdhsa_projection_power(stage2_meta, Y2=meta["Y"], out_dir=out_dir)
            except Exception as e:
                print(f"  [WARN] CD-HSA plots omitidos: {e}")
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

    # Transition matrix of the selected combination (always available when markov was used)
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
    # PSD of the latent space + channel influence per latent dimension
    # -----------------------------------------------------------------
    psds_latent, freqs_latent, mean_latent, std_latent, latent_psd_path = compute_and_plot_latent_psd(
        latent, sfreq=sfreq, out_dir=out_dir, fmin=args.l_freq, fmax=args.h_freq,
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

    # =====================================================================
    # 3. SELECT DIMENSION(S) FOR KM ANALYSIS
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
        data = latent[:, col:col + 1]
        print(f"\n  Using latent column {col} for 1D KM analysis")
    else:
        data = latent[:, :analysis_dim]
        print(f"\n  Using first {analysis_dim} latent columns for KM analysis")

    # --- Chapman-Kolmogorov test on the FULL data (global, per user choice) ---
    ck_result = chapman_kolmogorov_test(
        data,
        dt=dt,
        n_bins=20,
        threshold=0.15,
        plot=True,
        out_dir=out_dir,
        verbose=True,
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

    # Plot latent time series (all D dims in a single figure, unchanged)
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

    # =====================================================================
    # 4. CONFIGURATION + HIGH-DIMENSIONAL SPLIT DISPATCH
    # =====================================================================
    # Compute the dimension split.  For D < split_threshold (default 4),
    # the split returns a single chunk covering all dims -> single job,
    # identical to the legacy behaviour.
    jobs = _split_dimensions(D) if D >= args.split_threshold else [list(range(D))]
    n_jobs_total = len(jobs)

    print("\n" + "=" * 70)
    print("  STAGE 4: CONFIGURATION + KM JOB DISPATCH")
    print("=" * 70)
    print(f"  analysis_dim (D)        : {D}")
    print(f"  split_threshold         : {args.split_threshold}")
    print(f"  Number of sub-jobs      : {n_jobs_total}")
    print(f"  Sub-job dim partitions  : "
          f"{[_dim_label(j) for j in jobs]}")
    print(f"  Parallel execution      : "
          f"{n_jobs_total > 1 and not args.no_parallel}")
    sys.stdout.flush()

    # Common metadata (used by save_potential)
    metadata_common = {
        "super_subject": args.super_subject,
        "super_subject_label": ss_label,
        "subject_ids": subject_ids or "auto-resolved",
        "subjects_per_super_subject": args.subjects_per_super_subject,
        "subject_start_offset": args.subject_start_offset,
        "session": args.session,
        "task": args.task,
        "t_start": args.t_start,
        "t_end": args.t_end,
        "latent_dim": latent_dim,
        "global_analysis_dim": D,
        "n_sub_jobs": n_jobs_total,
        "sub_job_partitions": [_dim_label(j) for j in jobs],
        "pipeline": spec_label,
        "pipeline_spec": {k: v for k, v in spec.items()
                          if k.endswith("params") or k.startswith("stage")},
        "scoring_method": args.scoring_method,
    }

    # Ensure plotting style is set (idempotent) so workers don't race
    # on first-time style setup.
    try:
        setup_plotting_style()
    except Exception:
        pass

    base_config = CONFIGS['base_2D_model'].copy()

    if n_jobs_total == 1:
        # ----------------------------------------------------------------
        # SINGLE JOB: behave exactly as the legacy pipeline.
        # ----------------------------------------------------------------
        dim_indices = jobs[0]
        d_job = len(dim_indices)
        config = _build_job_config(
            base_config, d_job, args, latent_dim, ss_label,
            args.session, args.task, spec_label,
        )
        data_job = data[:, dim_indices]

        metadata = dict(metadata_common)
        metadata["analysis_dim"] = d_job
        metadata["dim_indices"] = list(dim_indices)
        metadata["dim_label"] = _dim_label(dim_indices)

        print(f"\n  KM config: {config}")
        sys.stdout.flush()

        _run_km_job(
            data=data_job,
            dt=dt,
            config=config,
            dim_indices=dim_indices,
            dim_label=_dim_label(dim_indices),
            out_dir=out_dir,
            save_potential_flag=save_potential_flag,
            metadata=metadata,
            save_potential_filename="potential_data.npz",
        )
    else:
        # ----------------------------------------------------------------
        # MULTIPLE JOBS: dispatch in parallel, then composite figures.
        # ----------------------------------------------------------------
        per_job_root = out_dir / "_per_job"
        if per_job_root.exists():
            shutil.rmtree(per_job_root, ignore_errors=True)
        per_job_root.mkdir(parents=True, exist_ok=True)

        # Build per-job spec dicts (do this in the main thread so the
        # workers receive plain args, no closures).
        job_specs = []
        for i, dim_indices in enumerate(jobs):
            d_job = len(dim_indices)
            config = _build_job_config(
                base_config, d_job, args, latent_dim, ss_label,
                args.session, args.task, spec_label,
            )
            data_job = data[:, dim_indices].copy()   # copy -> contiguous,
                                                     # avoid false sharing
            dim_label = _dim_label(dim_indices)
            job_out_dir = per_job_root / f"job_{i:02d}_{dim_label}"
            job_out_dir.mkdir(parents=True, exist_ok=True)

            metadata = dict(metadata_common)
            metadata["analysis_dim"] = d_job
            metadata["dim_indices"] = list(dim_indices)
            metadata["dim_label"] = dim_label

            save_potential_filename = f"potential_data_{dim_label}.npz"

            job_specs.append({
                "data": data_job,
                "dt": dt,
                "config": config,
                "dim_indices": list(dim_indices),
                "dim_label": dim_label,
                "out_dir": job_out_dir,
                "save_potential_flag": save_potential_flag,
                "metadata": metadata,
                "save_potential_filename": save_potential_filename,
            })

        print(f"\n  Dispatching {len(job_specs)} sub-jobs "
              f"(parallel={not args.no_parallel}) ...")
        sys.stdout.flush()

        if args.no_parallel or len(job_specs) == 1:
            # Sequential
            job_results = []
            for js in job_specs:
                job_results.append(_run_km_job(**js))
        else:
            # Parallel via joblib PROCESSES (not threads).
            #
            # We MUST use processes, not threads, because matplotlib's
            # mathtext parser (used by bw_optimization.py labels like
            # r'$\eta_{\rm DB}$ (dimensionless)') relies on module-level
            # mutable state that is NOT thread-safe.  Concurrent calls
            # to fig.tight_layout() from multiple threads corrupt the
            # parser and raise:
            #   ValueError: Expected end of text, found '$'
            #
            # Processes give each worker its own matplotlib state.
            # Numpy arrays in job_specs are small (~10 MB each for
            # 600k x 2 floats) so pickling overhead is negligible.
            try:
                from joblib import Parallel, delayed
            except ImportError:
                print("  [WARN] joblib not available; falling back to sequential.")
                job_results = [_run_km_job(**js) for js in job_specs]
            else:
                # n_jobs=-1 -> use all CPUs.  prefer="processes" forces
                # process-based parallelism even on non-POSIX systems.
                # max_nbytes=None keeps arrays in memory (no memmap disk
                # spill), which is fine for our sizes.
                job_results = Parallel(
                    n_jobs=-1, backend="loky", verbose=0,
                )(
                    delayed(_run_km_job)(**js) for js in job_specs
                )

        # ----------------------------------------------------------------
        # MOVE per-job potential_data files into the canonical out_dir.
        # ----------------------------------------------------------------
        if save_potential_flag:
            print("\n  Moving per-job potential_data files to out_dir ...")
            for jr in job_results:
                src = jr["out_dir"] / jr["save_potential_filename"]
                if src.exists():
                    dst = out_dir / jr["save_potential_filename"]
                    shutil.move(str(src), str(dst))
                    print(f"    {jr['dim_label']}: {dst.name}")

        # ----------------------------------------------------------------
        # COMPOSITE per-job figures into canonical single-row PNGs.
        # ----------------------------------------------------------------
        print("\n  Compositing per-job figures into canonical single-row PNGs ...")
        sys.stdout.flush()
        _composite_all_figures(job_results, out_dir)

        # ----------------------------------------------------------------
        # CLEANUP per-job temp dir (only per-job figures; potential_data
        # was already moved out).
        # ----------------------------------------------------------------
        shutil.rmtree(per_job_root, ignore_errors=True)
        print("  [OK] Per-job temp dir cleaned up.")

    # =====================================================================
    # 12. SUMMARY
    # =====================================================================
    # --- PIPELINE OVERVIEW (cross-stage plots) ---
    try:
        plot_pipeline_energy_budget(
            X_filtered=meta.get("preprocessing", {}).get("X_filtered"),
            meta=meta, latent=latent, out_dir=out_dir,
        )
        print("  Saved pipeline_energy_budget.png")
        plot_pipeline_flowchart(
            meta=meta, out_dir=out_dir,
            l_freq=args.l_freq, h_freq=args.h_freq,
            n_channels=len(raw.ch_names), sfreq=sfreq,
        )
        print("  Saved pipeline_flowchart.png")
    except Exception as e:
        print(f"  [WARN] Error en plots overview: {e}")

    total_time = time.time() - overall_t0
    print("\n" + "=" * 70)
    print("  SUPER-SUBJECT PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Super-subject       : {args.super_subject} ({ss_label})")
    print(f"  Session             : {args.session}")
    print(f"  Task                : {args.task}")
    print(f"  Pipeline            : {spec_label} (hash {spec_hash})")
    print(f"  Per-subject window  : {args.t_start:.1f}s -> {args.t_end:.1f}s")
    print(f"  Concatenated length : {raw.times[-1] - raw.times[0]:.1f} s")
    print(f"  Analysis dim (D)    : {D}")
    print(f"  Sub-jobs            : {n_jobs_total} "
          f"({_dim_label(jobs[0]) if jobs else 'n/a'}"
          f"{', ...' if n_jobs_total > 1 else ''})")
    print(f"  Total wall-clock    : {total_time:.1f} s")
    print(f"  Output saved to     : {out_dir.absolute()}")
    if save_potential_flag:
        if n_jobs_total == 1:
            print(f"  Potential data      : {out_dir / 'potential_data.npz'}")
        else:
            print(f"  Potential data      : {out_dir} / potential_data_dim*.npz "
                  f"({n_jobs_total} files)")

    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())

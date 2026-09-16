"""
Latent Subspace Extraction — Modular 3-Stage Pipeline
=====================================================

Refactored orchestrator: the monolithic pipeline is now a composition of
three decoupled stages with a uniform interface::

    Stage 1 — Embedding   : None | "hankel"
    Stage 2 — Dynamics    : "pca_ica" | "pca" | "dmd" | "diffusion_maps" | "cdhsa_specific_modes"
    Stage 3 — Selection   : "top_n" | "markov_fastest" | "markov_slowest"

Arbitrary combinations are allowed and comparable under a single API,
e.g. "Hankel + PCA + Markov", "Hankel + DM + top-n" (NLSA), "no Hankel +
ICA + Markov", ...

Typical usage (new API)::

    from extract_latent_subspace import extract_latent_space

    # Legacy-style: no Hankel, ICA + Markov fastest
    latent, meta = extract_latent_space(
        raw, n_dim=2,
        stage1_embedding=None,
        stage2_dynamics="pca_ica",
        stage2_params={"n_components": None, "ica_method": "picard"},
        stage3_selection="markov_fastest",
        stage3_params={"n_bins": 5, "search_strategy": "exhaustive"},
    )

    # NLSA: Hankel + Diffusion Maps + top-n
    latent, meta = extract_latent_space(
        raw, n_dim=2,
        stage1_embedding="hankel",
        stage1_params={"depth": 250},
        stage2_dynamics="diffusion_maps",
        stage2_params={"svd_rank": 50, "sigma": None, "k": 100, "alpha": 0.5},
        stage3_selection="top_n",
    )

Backward compatibility
----------------------
The legacy ``scoring_method=...`` keyword (and its associated parameters)
is still accepted and mapped onto the new stages — see
:func:`map_legacy_scoring_method`.  Legacy meta keys
(``selected_indices``, ``latent_scores``, ``preprocessing``, ``Y``,
``Y_shape``, ``elapsed_time``) are preserved at the top level.

Return value
------------
latent : np.ndarray, shape (n_samples, n_dim)
    The extracted latent subspace (time along rows).  With the Hankel
    embedding, ``n_samples = n_times - depth + 1``.
meta : dict
    ``meta["pipeline"]`` records the exact stage chain; each stage injects
    its own sub-dict (``meta["stage1"]``, ``meta["stage2"]``,
    ``meta["stage3"]``); legacy keys are kept for compatibility.
"""

from __future__ import annotations

import time
import sys
import warnings
from pathlib import Path
from typing import Literal

import mne
import numpy as np

from src.latent_space_extraction.eeg_preprocessing import (
    extract_filtered_data_matrix,
    load_raw_eeg,
)
from src.latent_space_extraction.pipeline import (
    PipelineContext,
    build_stage1,
    build_stage2,
    build_stage3,
)


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------

DEFAULT_L_FREQ: float = 1.0
DEFAULT_H_FREQ: float = 40.0
DEFAULT_ICA_METHOD: str = "picard"
DEFAULT_ICA_RANDOM_STATE: int = 42
DEFAULT_N_BINS: int = 5
DEFAULT_FC_METRIC: str = "variance_sum"

_LEGACY_SCORING_METHODS = (
    "markov", "markov_inverted", "conservative", "weighted", "sequential",
    "pareto", "independent", "hankel_dmd", "diffusion_maps",
)


# ---------------------------------------------------------------------------
# Legacy → new-API mapping
# ---------------------------------------------------------------------------

def map_legacy_scoring_method(
    scoring_method: str,
    *,
    n_dim: int = 2,
    fc_metric: str = DEFAULT_FC_METRIC,
    n_bins: int = DEFAULT_N_BINS,
    alpha: float | None = None,
    primary_criterion: str = "markov",
    sequential_K: int | None = None,
    hankel_embedding_depth: int | None = None,
    diffusion_sigma: float | None = None,
    diffusion_k: int = 100,
    diffusion_time: float = 0.0,
    diffusion_alpha: float = 0.5,
    n_components: int | float | None = None,
    ica_method: str = DEFAULT_ICA_METHOD,
    ica_random_state: int | None = DEFAULT_ICA_RANDOM_STATE,
    retained_labels: list[str] | None = None,
    search_strategy: str = "exhaustive",
) -> dict:
    """
    Map a legacy ``scoring_method`` call onto the new 3-stage spec.

    Returns
    -------
    spec : dict
        ``{"stage1_embedding", "stage1_params", "stage2_dynamics",
        "stage2_params", "stage3_selection", "stage3_params"}`` ready to
        be unpacked into :func:`extract_latent_space`.

    Mapping table
    -------------
    ==================  ============================================
    legacy method       new stage chain
    ==================  ============================================
    ``markov``          None + pca_ica + markov_fastest
    ``markov_inverted`` None + pca_ica + markov_slowest
    ``hankel_dmd``      hankel + dmd (rank=n_dim) + top_n
    ``diffusion_maps``  None + diffusion_maps (n_components=n_dim) + top_n
    ``conservative``    None + pca_ica + legacy_fc:conservative
    ``weighted``        None + pca_ica + legacy_fc:weighted
    ``sequential``      None + pca_ica + legacy_fc:sequential
    ``pareto``          None + pca_ica + legacy_fc:pareto
    ``independent``     None + pca_ica + legacy_fc:independent
    ==================  ============================================
    """
    if scoring_method not in _LEGACY_SCORING_METHODS:
        raise ValueError(
            f"Unknown legacy scoring_method {scoring_method!r}. "
            f"Valid options: {list(_LEGACY_SCORING_METHODS)}"
        )

    ica_params = {
        "n_components": n_components,
        "ica_method": ica_method,
        "ica_random_state": ica_random_state,
        "retained_labels": retained_labels,
    }

    if scoring_method == "markov":
        return {
            "stage1_embedding": None,
            "stage1_params": {},
            "stage2_dynamics": "pca_ica",
            "stage2_params": ica_params,
            "stage3_selection": "markov_fastest",
            "stage3_params": {"n_bins": n_bins, "search_strategy": search_strategy},
        }

    if scoring_method == "markov_inverted":
        return {
            "stage1_embedding": None,
            "stage1_params": {},
            "stage2_dynamics": "pca_ica",
            "stage2_params": ica_params,
            "stage3_selection": "markov_slowest",
            "stage3_params": {"n_bins": n_bins, "search_strategy": search_strategy},
        }

    if scoring_method == "hankel_dmd":
        return {
            "stage1_embedding": "hankel",
            "stage1_params": {"depth": hankel_embedding_depth},
            "stage2_dynamics": "dmd",
            # rank = n_dim reproduces the legacy numerics exactly
            "stage2_params": {"rank": n_dim},
            "stage3_selection": "top_n",
            "stage3_params": {},
        }

    if scoring_method == "diffusion_maps":
        return {
            "stage1_embedding": None,
            "stage1_params": {},
            "stage2_dynamics": "diffusion_maps",
            # n_components = n_dim reproduces the legacy branch
            "stage2_params": {
                "n_components": n_dim,
                "sigma": diffusion_sigma,
                "k": diffusion_k,
                "diffusion_time": diffusion_time,
                "alpha": diffusion_alpha,
            },
            "stage3_selection": "top_n",
            "stage3_params": {},
        }

    # Conservative-fraction based legacy strategies
    return {
        "stage1_embedding": None,
        "stage1_params": {},
        "stage2_dynamics": "pca_ica",
        "stage2_params": ica_params,
        "stage3_selection": "legacy_fc",
        "stage3_params": {
            "method": scoring_method,
            "fc_metric": fc_metric,
            "alpha": alpha,
            "primary_criterion": primary_criterion,
            "sequential_K": sequential_K,
            "n_bins": n_bins,
            "search_strategy": search_strategy,
        },
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def extract_latent_space(
    raw_input: mne.io.Raw | str | Path,
    *,
    # ---- Subspace dimension ----
    n_dim: int = 2,
    # ---- Stage 1: Embedding ----
    stage1_embedding: Literal[None, "hankel"] = None,
    stage1_params: dict | None = None,
    # ---- Stage 2: Dynamics ----
    stage2_dynamics: Literal["pca_ica", "pca", "dmd", "diffusion_maps", "cdhsa_specific_modes"] = "pca_ica",
    stage2_params: dict | None = None,
    # ---- Stage 3: Selection ----
    stage3_selection: Literal["top_n", "markov_fastest", "markov_slowest"] = "top_n",
    stage3_params: dict | None = None,
    # ---- Preprocessing params ----
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    # ---- Search / compute params ----
    n_workers: int | None = None,
    verbose: bool | str | None = None,
    # ---- Channel intersection (CD-HSA consistency) ----
    channel_intersection: list[str] | None = None,
    # ---- Legacy API (deprecated; mapped onto the stages above) ----
    scoring_method: str | None = None,
    fc_metric: str = DEFAULT_FC_METRIC,
    n_bins: int = DEFAULT_N_BINS,
    alpha: float | None = None,
    primary_criterion: Literal["fc", "markov"] = "markov",
    sequential_K: int | None = None,
    hankel_embedding_depth: int | None = None,
    diffusion_sigma: float | None = None,
    diffusion_k: int = 100,
    diffusion_time: float = 0.0,
    diffusion_alpha: float = 0.5,
    n_components: int | float | None = None,
    ica_method: str = DEFAULT_ICA_METHOD,
    ica_random_state: int | None = DEFAULT_ICA_RANDOM_STATE,
    retained_labels: list[str] | None = None,
    search_strategy: Literal["exhaustive", "greedy"] = "exhaustive",
) -> tuple[np.ndarray, dict]:
    """
    Extract a low-dimensional latent subspace from an EEG recording with
    the modular 3-stage pipeline.

    Parameters
    ----------
    raw_input : mne.io.Raw | str | Path
        MNE Raw object in memory, or path to a raw data file.
    n_dim : int, default 2
        Target dimensionality of the latent subspace.
    stage1_embedding : None | "hankel", default None
        Optional delay embedding.  ``"hankel"`` builds the multivariate
        block-Hankel matrix ``H ∈ R^{(N_c·T)×(N_t−T+1)}``;
        ``stage1_params={"depth": None}`` auto-computes
        ``T = clip(sfreq * 0.25, 50, 200)``.
    stage2_dynamics : str, default "pca_ica"
        Dynamics decomposition: ``"pca"`` (truncated SVD), ``"pca_ica"``
        (MNE ICA on real channels / sklearn FastICA on Hankel PCs),
        ``"dmd"`` (Hankel-DMD / direct AR-1), ``"diffusion_maps"`` (DM on
        time instants / NLSA on Hankel PCs).
    stage2_params : dict | None
        Per-dynamics parameters (e.g. ``n_components``, ``svd_rank``,
        ``sigma``, ``k``, ``diffusion_time``, ``alpha``, ``rank``,
        ``ica_method``, ...).
    stage3_selection : str, default "top_n"
        Final mode selection: ``"top_n"`` (simple ranking),
        ``"markov_fastest"`` / ``"markov_slowest"`` (Markovian subspace
        criterion via ``find_best_subspace_markov`` with
        ``maximize=False`` / ``True``).
    stage3_params : dict | None
        e.g. ``{"n_bins": 5, "search_strategy": "exhaustive"}``.
    l_freq, h_freq : float
        Band-pass filter cut-offs (Hz).
    n_workers : int | None
        Parallel processes for the combinatorial Markov search.
    verbose : bool | str | None
        MNE verbosity level.
    scoring_method : str | None, optional
        **Deprecated legacy API.**  When given, it overrides the three
        stage arguments via :func:`map_legacy_scoring_method`.
    fc_metric, n_bins, alpha, primary_criterion, sequential_K,
    hankel_embedding_depth, diffusion_sigma, diffusion_k, diffusion_time,
    diffusion_alpha, n_components, ica_method, ica_random_state,
    retained_labels, search_strategy
        Legacy parameters, only used together with ``scoring_method``.

    Returns
    -------
    latent : np.ndarray, shape (n_samples, n_dim)
    meta : dict
    """
    t0 = time.time()

    # =====================================================================
    # Legacy API → new stage spec
    # =====================================================================
    legacy_method_used: str | None = None
    if scoring_method is not None:
        legacy_method_used = scoring_method
        warnings.warn(
            f"[extract_latent_space] scoring_method={scoring_method!r} is "
            f"deprecated; use stage1_embedding / stage2_dynamics / "
            f"stage3_selection instead. Mapping legacy arguments onto the "
            f"new 3-stage API.",
            DeprecationWarning,
            stacklevel=2,
        )
        spec = map_legacy_scoring_method(
            scoring_method,
            n_dim=n_dim,
            fc_metric=fc_metric,
            n_bins=n_bins,
            alpha=alpha,
            primary_criterion=primary_criterion,
            sequential_K=sequential_K,
            hankel_embedding_depth=hankel_embedding_depth,
            diffusion_sigma=diffusion_sigma,
            diffusion_k=diffusion_k,
            diffusion_time=diffusion_time,
            diffusion_alpha=diffusion_alpha,
            n_components=n_components,
            ica_method=ica_method,
            ica_random_state=ica_random_state,
            retained_labels=retained_labels,
            search_strategy=search_strategy,
        )
        stage1_embedding = spec["stage1_embedding"]
        stage1_params = spec["stage1_params"]
        stage2_dynamics = spec["stage2_dynamics"]
        stage2_params = spec["stage2_params"]
        stage3_selection = spec["stage3_selection"]
        stage3_params = spec["stage3_params"]

    stage1_params = dict(stage1_params or {})
    stage2_params = dict(stage2_params or {})
    stage3_params = dict(stage3_params or {})

    pipeline_str = (
        f"{stage1_embedding or 'none'}+{stage2_dynamics}+{stage3_selection}"
    )
    print("\n" + "=" * 70)
    print(f"  LATENT-SPACE PIPELINE: {pipeline_str}")
    print("=" * 70)

    # =====================================================================
    # Load raw if a path was given
    # =====================================================================
    if isinstance(raw_input, (str, Path)):
        raw = load_raw_eeg(str(raw_input), verbose=verbose)
    else:
        raw = raw_input

    sfreq = float(raw.info["sfreq"])
    dt = 1.0 / sfreq

    # =====================================================================
    # Shared context
    # =====================================================================
    ctx = PipelineContext(
        raw=raw,
        sfreq=sfreq,
        dt=dt,
        l_freq=l_freq,
        h_freq=h_freq,
        n_workers=n_workers,
        verbose=verbose,
        stage1_name=stage1_embedding,
        channel_intersection=channel_intersection,
    )

    # =====================================================================
    # Stage 0 — filtered channel matrix
    # ---------------------------------------------------------------------
    # The MNE-ICA branch (stage2="pca_ica" without Hankel) filters the Raw
    # object internally (legacy behaviour of run_full_preprocessing), so it
    # must NOT be pre-filtered here.  Every other combination operates on
    # the band-pass filtered channel matrix X.
    # =====================================================================
    stage2_uses_raw_directly = (stage2_dynamics == "pca_ica" and stage1_embedding is None)

    if stage2_uses_raw_directly:
        X = None
        print("\n[Stage 0] MNE-ICA branch: filtering deferred to Stage 2 "
              "(run_full_preprocessing).")
    else:
        t_pre = time.time()
        print("[Stage 0] Band-pass filtering raw data...")
        sys.stdout.flush()
        X, _raw_filtered, sfreq = extract_filtered_data_matrix(
            raw, l_freq=l_freq, h_freq=h_freq, verbose=verbose,
        )
        print(f"\n[Stage 0] Filtered data matrix: shape={X.shape} "
              f"| sfreq={sfreq:.1f} Hz | {time.time() - t_pre:.2f}s")
        sys.stdout.flush()

    # =====================================================================
    # Stage 1 — Embedding
    # =====================================================================
    t1 = time.time()
    stage1 = build_stage1(stage1_embedding, stage1_params)
    if stage2_uses_raw_directly:
        # No Hankel possible here (stage1 is None by the guard above);
        # Stage 2 will receive the Raw object through the context.
        embedded, meta1 = None, {
            "embedding": None,
            "note": "deferred to Stage 2 (MNE ICA branch)",
            "n_time_lost": 0,
        }
    else:
        if stage1_embedding == "hankel":
            print(f"[Stage 1] Building Hankel embedding (input {X.shape})...")
            sys.stdout.flush()
        embedded, meta1 = stage1.fit_transform(X, ctx=ctx)
    ctx.stage1_meta = meta1

    print(f"[Stage 1] embedding={stage1_embedding!r} | "
          f"in={meta1.get('input_shape', '-')} → out={meta1.get('output_shape', '-')}"
          f" | {time.time() - t1:.2f}s")
    sys.stdout.flush()

    # =====================================================================
    # Stage 2 — Dynamics
    # =====================================================================
    t2 = time.time()
    stage2 = build_stage2(stage2_dynamics, stage2_params)
    print(f"[Stage 2] Running dynamics='{stage2_dynamics}'...")
    sys.stdout.flush()
    Y2, meta2 = stage2.fit_transform(embedded, ctx=ctx, n_dim=n_dim)

    # Empate 5.1 — complex outputs (DMD) are already real-valued inside
    # the Stage-2 classes; enforce defensively at the handoff.
    if np.iscomplexobj(Y2):
        Y2 = np.real(Y2)

    n_time_lost = int(meta1.get("n_time_lost", 0) or 0)
    # Add extra time lost by Stage 2 (e.g. block-Hankel in cdhsa_specific_modes)
    n_time_lost += int(meta2.get("n_time_lost_by_block_hankel", 0) or 0)
    print(f"[Stage 2] dynamics={stage2_dynamics!r} (branch={meta2.get('branch', '-')}) | "
          f"in={meta2.get('input_shape', '-')} → out={tuple(Y2.shape)}"
          f" | {time.time() - t2:.2f}s")
    sys.stdout.flush()

    # =====================================================================
    # Stage 3 — Mode selection
    # =====================================================================
    t3 = time.time()
    stage3 = build_stage3(stage3_selection, stage3_params)
    print(f"[Stage 3] Running selection='{stage3_selection}'...")
    sys.stdout.flush()
    Y_sel, meta3 = stage3.fit_transform(Y2, ctx=ctx, n_dim=n_dim)
    sys.stdout.flush()

    # Final transpose: (n_dim, T') → (n_samples, n_dim)
    latent = Y_sel.T
    print(f"[Stage 3] selection={stage3_selection!r} | "
          f"selected={meta3['selected_indices']} | latent={latent.shape}"
          f" | {time.time() - t3:.2f}s")
    sys.stdout.flush()

    elapsed = time.time() - t0

    # =====================================================================
    # Metadata — pipeline traceability (Empate 6) + legacy keys
    # =====================================================================
    latent_scores: dict = {}
    for key in (
        "frequencies_hz", "damping_rates", "eigenvalues",
        "continuous_eigenvalues", "eigenvalues_latent", "sigma_used",
        "epsilon_fitted", "k_neighbors", "diffusion_time", "alpha",
        "spectral_gap", "singular_values",
    ):
        if key in meta2:
            latent_scores[key] = meta2[key]
    if "eigenvalues" in meta2:
        latent_scores["all_eigenvalues"] = meta2["eigenvalues"]
    latent_scores.update(meta3.get("scores", {}))

    n_samples_latent = latent.shape[0]
    preprocessing: dict = {
        "l_freq": l_freq,
        "h_freq": h_freq,
        "sfreq": sfreq,
        "dt": dt,
        "D": int(Y2.shape[0]),
        "T": int(Y2.shape[1]),
        "n_samples_latent": int(n_samples_latent),
        "n_time_lost_by_embedding": n_time_lost,
        "excluded_indices": meta2.get("excluded_indices", []),
        "kept_indices": meta2.get("kept_indices", list(range(Y2.shape[0]))),
    }
    if X is not None:
        preprocessing["n_channels"] = int(X.shape[0])
        preprocessing["n_times"] = int(X.shape[1])
        preprocessing["X_filtered"] = X
    if stage1_embedding == "hankel":
        preprocessing["embedding_depth"] = meta1.get("depth")
        preprocessing["hankel_shape"] = meta1.get("output_shape")
    if "singular_values" in meta2:
        preprocessing["singular_values"] = meta2["singular_values"]
    if "preprocessing" in meta2:  # full legacy Stage-I dict (MNE branch)
        preprocessing["stage_I"] = meta2["preprocessing"]
        preprocessing["n_channels"] = meta2["preprocessing"]["n_channels"]

    meta = {
        # --- new pipeline traceability ---
        "pipeline": {
            "stage1": stage1_embedding,
            "stage2": stage2_dynamics,
            "stage3": stage3_selection,
            "params": {
                "stage1_params": _jsonable(stage1_params),
                "stage2_params": _jsonable(stage2_params),
                "stage3_params": _jsonable(stage3_params),
            },
            "chain": pipeline_str,
        },
        "stage1": meta1,
        "stage2": {k: v for k, v in meta2.items() if k != "preprocessing"},
        "stage3": meta3,
        # --- legacy top-level keys (do not break test_iga_...) ---
        "selected_indices": list(meta3["selected_indices"]),
        "scoring_method": legacy_method_used or pipeline_str,
        "latent_scores": latent_scores,
        "preprocessing": preprocessing,
        "Y": Y2,
        "Y_shape": tuple(Y2.shape),
        "elapsed_time": elapsed,
    }

    print(f"\n[Pipeline] Done in {elapsed:.2f}s — latent shape {latent.shape}")
    sys.stdout.flush()
    return latent, meta


def _jsonable(params: dict) -> dict:
    """Convert a params dict into a JSON/cache-friendly copy."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = repr(v)
    return out


# ---------------------------------------------------------------------------
# Quick sampled conservative-fraction diagnostic (legacy, kept for compat)
# ---------------------------------------------------------------------------

def sample_conservative_fraction(
    Y: np.ndarray,
    n_dim: int,
    *,
    n_samples: int = 500,
    metric: str = "variance_sum",
    n_workers: int | None = None,
) -> dict:
    """
    Evaluate the conservative fraction on the first *n_samples* combinations
    of *n_dim* components (legacy diagnostic, kept for compatibility).
    """
    from itertools import combinations

    from src.latent_space_extraction.conservative_fraction import (
        score_many_combinations_fc,
    )

    D, T = Y.shape
    all_combs = list(combinations(range(D), n_dim))
    total = len(all_combs)

    if n_samples >= total:
        sampled = all_combs
        n_samples = total
    else:
        sampled = all_combs[:n_samples]

    print(f"\n[SampleFC] Sampling {n_samples}/{total} combinations (N={n_dim}, D={D})")

    scored = score_many_combinations_fc(
        Y, sampled, metric=metric, n_workers=n_workers, show_progress=False
    )
    fc_values = np.array([fc for _, fc in scored])

    vmin, vmax = float(fc_values.min()), float(fc_values.max())
    vmean = float(fc_values.mean())
    vstd = float(fc_values.std())
    cv = vstd / vmean if vmean > 0 else 0.0

    report = {
        "fc_values": fc_values,
        "combinations": sampled,
        "n_evaluated": n_samples,
        "n_total": total,
        "metric": metric,
        "min": vmin,
        "max": vmax,
        "range": vmax - vmin,
        "mean": vmean,
        "std": vstd,
        "cv": cv,
        "constant": np.allclose(fc_values, fc_values[0]),
        "degenerate": cv < 1e-6,
    }

    print(f"  min={vmin:.6f}  max={vmax:.6f}  range={report['range']:.6f}")
    print(f"  mean={vmean:.6f}  std={vstd:.6f}  CV={cv:.6f}")
    print(f"  constant={report['constant']}  degenerate={report['degenerate']}")

    return report


# ---------------------------------------------------------------------------
# Diagnostic: does the dataset support discriminative subspace selection?
# (legacy, kept for compatibility)
# ---------------------------------------------------------------------------

def diagnose_subspace_discrimination(
    Y: np.ndarray,
    n_dim: int = 2,
    *,
    fc_metrics: list[str] | None = None,
    n_bins: int = 5,
    n_workers: int | None = None,
    cv_threshold: float = 0.05,
) -> dict:
    """
    Exhaustively evaluate *all* combinations and report whether each
    criterion discriminates on this dataset (legacy diagnostic, kept for
    compatibility).
    """
    from itertools import combinations

    from src.latent_space_extraction.conservative_fraction import (
        score_many_combinations_fc,
    )
    from src.latent_space_extraction.markov_subspace import (
        score_many_combinations_markov,
    )

    if fc_metrics is None:
        fc_metrics = ["variance_sum", "first_pc_var", "total_variance"]

    D, T = Y.shape
    all_combs = list(combinations(range(D), n_dim))
    n_combs = len(all_combs)

    print(f"\n[Diagnostic] Evaluating discriminative power on {n_combs:,} combinations")
    print(f"[Diagnostic] D={D}, T={T}, N={n_dim}, n_bins={n_bins}\n")

    report: dict = {
        "n_combinations_evaluated": n_combs,
        "n_dim": n_dim,
        "D": D,
        "conservative_fraction": {},
        "markov_time": {},
        "recommendation": {},
    }

    print("=" * 60)
    print("  CONSERVATIVE FRACTION ANALYSIS")
    print("=" * 60)

    for metric in fc_metrics:
        scored = score_many_combinations_fc(
            Y, all_combs, metric=metric, n_workers=n_workers, show_progress=False
        )
        fc_values = np.array([fc for _, fc in sorted(scored, key=lambda x: x[0])])

        vmin, vmax = float(fc_values.min()), float(fc_values.max())
        vrange = vmax - vmin
        vmean = float(fc_values.mean())
        vstd = float(fc_values.std())
        cv = vstd / vmean if vmean > 0 else 0.0
        discriminates = cv > cv_threshold

        best_idx = int(np.argmax(fc_values))
        worst_idx = int(np.argmin(fc_values))

        report["conservative_fraction"][metric] = {
            "discriminates": discriminates,
            "min": vmin,
            "max": vmax,
            "range": vrange,
            "mean": vmean,
            "std": vstd,
            "cv": cv,
            "best_comb": all_combs[best_idx],
            "best_value": vmax,
            "worst_comb": all_combs[worst_idx],
            "worst_value": vmin,
        }

        verdict = "YES" if discriminates else "NO"
        print(f"\n  Metric: {metric}")
        print(f"    Range  : {vmin:.6f} -> {vmax:.6f} (span = {vrange:.6f})")
        print(f"    Mean   : {vmean:.6f}")
        print(f"    Std    : {vstd:.6f}")
        print(f"    CV     : {cv:.4f}")
        print(f"    Best   : {all_combs[best_idx]} (fc={vmax:.6f})")
        print(f"    Worst  : {all_combs[worst_idx]} (fc={vmin:.6f})")
        print(f"    Discriminates? {verdict} (CV threshold = {cv_threshold})")

    print("\n" + "=" * 60)
    print("  MARKOV TIME ANALYSIS")
    print("=" * 60)

    scored_tau = score_many_combinations_markov(
        Y, all_combs, n_bins=n_bins, n_workers=n_workers, show_progress=False
    )
    tau_values = np.array([tau for _, tau in sorted(scored_tau, key=lambda x: x[0])])

    tau_finite = tau_values[np.isfinite(tau_values)]
    n_inf = int(np.sum(~np.isfinite(tau_values)))

    if len(tau_finite) > 0:
        vmin_tau, vmax_tau = float(tau_finite.min()), float(tau_finite.max())
        vrange_tau = vmax_tau - vmin_tau
        vmean_tau = float(tau_finite.mean())
        vstd_tau = float(tau_finite.std())
        cv_tau = vstd_tau / vmean_tau if vmean_tau > 0 else 0.0
    else:
        vmin_tau = vmax_tau = vrange_tau = vmean_tau = vstd_tau = cv_tau = float("nan")

    discriminates_tau = cv_tau > cv_threshold
    best_idx_tau = int(np.argmin(tau_values))
    worst_idx_tau = int(np.argmax(np.where(np.isfinite(tau_values), tau_values, -np.inf)))

    report["markov_time"] = {
        "discriminates": discriminates_tau,
        "min": vmin_tau,
        "max": vmax_tau,
        "range": vrange_tau,
        "mean": vmean_tau,
        "std": vstd_tau,
        "cv": cv_tau,
        "n_inf": n_inf,
        "best_comb": all_combs[best_idx_tau],
        "best_value": float(tau_values[best_idx_tau]),
        "worst_comb": all_combs[worst_idx_tau],
        "worst_value": float(tau_values[worst_idx_tau]),
    }

    verdict_tau = "YES" if discriminates_tau else "NO"
    print(f"\n  Metric: Markov relaxation time (n_bins={n_bins})")
    print(f"    Range    : {vmin_tau:.6f} -> {vmax_tau:.6f} (span = {vrange_tau:.6f})")
    print(f"    Mean     : {vmean_tau:.6f}")
    print(f"    Std      : {vstd_tau:.6f}")
    print(f"    CV       : {cv_tau:.4f}")
    print(f"    Inf count: {n_inf}/{n_combs} non-ergodic")
    print(f"    Best     : {all_combs[best_idx_tau]} (tau={tau_values[best_idx_tau]:.6f})")
    print(f"    Worst    : {all_combs[worst_idx_tau]} (tau={tau_values[worst_idx_tau]:.6f})")
    print(f"    Discriminates? {verdict_tau} (CV threshold = {cv_threshold})")

    print("\n" + "=" * 60)
    print("  RECOMMENDATION")
    print("=" * 60)

    fc_discriminates_any = any(
        r["discriminates"] for r in report["conservative_fraction"].values()
    )

    if discriminates_tau and not fc_discriminates_any:
        method = "markov"
        rationale = (
            f"Markov time discriminates strongly (CV={cv_tau:.3f}) while "
            f"conservative fraction does not across all metrics. "
            f"Use 'markov' as the primary scoring method."
        )
    elif fc_discriminates_any and not discriminates_tau:
        method = "conservative"
        rationale = (
            "Conservative fraction discriminates but Markov time does not. "
            "Use 'conservative' or a weighted strategy."
        )
    elif discriminates_tau and fc_discriminates_any:
        method = "weighted"
        rationale = (
            "Both criteria discriminate. Consider 'weighted' or 'pareto' "
            "to exploit their complementarity."
        )
    else:
        method = "markov"
        rationale = (
            "Neither criterion discriminates strongly on this dataset. "
            "Default to 'markov' (empirically robust)."
        )

    report["recommendation"] = {"method": method, "rationale": rationale}
    print(f"\n  Recommended scoring method: {method}")
    print(f"  Rationale: {rationale}\n")

    return report

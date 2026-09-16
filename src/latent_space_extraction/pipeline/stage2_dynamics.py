"""
Stage 2 — Dynamics Analysis
===========================

Second stage of the modular latent-space pipeline.  It decomposes the
(embedded or raw) data matrix into a set of dynamical modes/scores.

Supported dynamics and their behaviour with/without the Hankel embedding:

=================  ==============================  ============================
Stage 2            Input without Hankel            Input with Hankel
=================  ==============================  ============================
``pca``            Truncated SVD of X (N_c × N_t)  Truncated SVD of H
``pca_ica``        MNE ICA over X (real channels)  sklearn FastICA over PCs of H
``dmd``            Direct DMD on X (spatial AR-1)  DMD on H (legacy pipeline)
``diffusion_maps`` DM over time instants of X      NLSA: SVD of H + DM on PCs
``cdhsa_specific_modes``  **Requires Hankel.** Projects the block-Hankel
                  onto the condition-specific modes from CD-HSA Step D.
                  Input H (530 × T) → block-Hankel (530×L, T-L+1) →
                  W_sel.T @ H_block → (top_n, T-L+1).
=================  ==============================  ============================

Shape convention (unbreakable rule)
-----------------------------------
Input and output matrices flow as ``(n_features, n_samples)``.  The output
of every Stage-2 variant is a **real** matrix ``Y`` of shape ``(D, T')``
whose rows are candidate latent time series for Stage 3.  ``T'`` may be
shorter than the original ``N_t`` when the Hankel embedding is active
(``T' = N_t − depth + 1``); the effective time axis is propagated through
the metadata.

Critical handoffs implemented here (see refactor specification §5)
------------------------------------------------------------------
* **Empate 1 — Hankel → ICA:** MNE ICA / ICLabel are *not* used (virtual
  delay channels have no topography).  Instead: truncated SVD of H →
  whitening → ``sklearn.decomposition.FastICA``.  No artefact rejection is
  needed (input data was already cleaned with GEDAI).
* **Empate 2 — Hankel → Diffusion Maps (NLSA):** the raw Hankel columns
  live in ≈ R^{N_c·T} (curse of dimensionality).  The Hankel matrix is
  first truncated to ``svd_rank`` principal components and the DM graph is
  built on the points ``V_r Σ_r`` in ``R^{svd_rank}`` (Nonlinear Laplacian
  Spectral Analysis).  ``sigma`` keeps being estimated via the
  Berry-Giannakis-Harlim method (BGH) inside
  ``diffusion_maps_extractor``.
* **Empate 3 — no Hankel → Diffusion Maps:** the DM operates over time
  instants: each column of X is a point in ``R^{N_c}`` (dimensionality is
  low, so k-NN is efficient without pre-PCA).  This matches the legacy
  branch, which transposed internally.
* **Empate 4 — DMD with/without Hankel:** with Hankel the legacy
  ``eeg_hankel_dmd_core`` is used directly (average-reference +
  drop-last-channel live inside it).  Without Hankel a direct DMD is
  computed on X — mathematically an order-1 autoregressive model in
  channel space ("dynamic PCA"); it does **not** capture temporal memory
  or nonlinear attractors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

from src.latent_space_extraction.hankel_dmd_extractor import eeg_hankel_dmd_core
from src.latent_space_extraction.diffusion_maps_extractor import (
    extract_diffusion_maps_latent_space,
)

from src.latent_space_extraction.pipeline.stage1_embedding import PipelineContext


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_block_hankel(
    H: np.ndarray,
    L: int,
) -> np.ndarray:
    """
    Build a block-Hankel matrix by stacking L temporally lagged copies
    of the input matrix.

    Given H ∈ R^{p × T}, the block-Hankel is::

        H_block = [H(0); H(1); ...; H(L-1)]

    where H(lag) = H[:, lag : T-L+1+lag], yielding
    H_block ∈ R^{p·L × T-L+1}.

    This matches the internal representation used by CD-HSA Step A
    (build_block_hankel), ensuring that modes learned by CD-HSA can
    be directly applied to new Hankel matrices.

    Parameters
    ----------
    H : ndarray, shape (p, T)
        Input Hankel matrix (channels × delays, time).
    L : int
        Number of temporal lags to stack.

    Returns
    -------
    H_block : ndarray, shape (p*L, T-L+1)
    """
    p, T = H.shape
    T_out = T - L + 1
    # Pre-allocate for efficiency
    H_block = np.empty((p * L, T_out), dtype=H.dtype)
    for lag in range(L):
        H_block[lag * p:(lag + 1) * p, :] = H[:, lag:lag + T_out]
    return H_block


def _truncated_svd(
    A: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Truncated SVD of ``A`` (m × n) keeping ``rank`` components.

    Uses :func:`scipy.sparse.linalg.svds` (memory-friendly for large
    Hankel matrices) and returns the triple ``(U, s, Vh)`` sorted by
    descending singular value.

    Sign convention
    ---------------
    Singular vectors are only defined up to a sign, and ``svds`` (ARPACK)
    starts from a random vector, so the *same* matrix can yield opposite
    signs across calls/machines.  We canonicalise the sign (the
    largest-magnitude loading of each left singular vector is forced
    positive — same convention as sklearn's ``svd_flip``, implemented
    inline to stay version-agnostic), making the decomposition
    deterministic run-to-run — important for test-retest reproducibility
    and cache coherence.
    """
    m, n = A.shape
    max_rank = min(m, n) - 1
    if rank > max_rank:
        raise ValueError(
            f"[Stage2] Requested rank {rank} but matrix of shape {A.shape} "
            f"admits at most {max_rank}. Reduce n_components / svd_rank."
        )
    U, s, Vh = svds(A.astype(np.float64), k=rank)
    order = np.argsort(s)[::-1]
    U, s, Vh = U[:, order], s[order], Vh[order, :]
    U, Vh = _canonical_svd_sign(U, Vh)
    return U, s, Vh


def _canonical_svd_sign(
    U: np.ndarray,
    Vh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic sign convention for SVD factors (sklearn ``svd_flip``
    semantics, u-based): for each component, the loading of largest
    absolute value in the left singular vector is forced positive, and
    the matching right singular vector is flipped accordingly.
    """
    max_abs_idx = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_idx, np.arange(U.shape[1])])
    signs[signs == 0.0] = 1.0
    return U * signs, Vh * signs[:, np.newaxis]


def _resolve_n_components(
    n_components: int | None,
    n_dim: int,
    max_allowed: int,
    default: int = 50,
) -> int:
    """
    Resolve the Stage-2 working rank.

    ``None`` → ``min(default, max_allowed)``; always at least ``n_dim`` so
    that Stage 3 can select ``n_dim`` rows.
    """
    if n_components is None:
        n_components = min(default, max_allowed)
    n_components = int(n_components)
    if n_components < n_dim:
        raise ValueError(
            f"[Stage2] n_components ({n_components}) must be >= n_dim "
            f"({n_dim}) so that Stage 3 can select {n_dim} modes."
        )
    return n_components


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class PCADynamics:
    """
    Truncated PCA (SVD) of the input matrix.

    Works on both X (channels × time) and H (Hankel): returns the PC time
    courses ``Σ Vᵀ`` of shape ``(n_components, T')``.

    Parameters (via ``stage2_params``)
    ----------------------------------
    n_components : int | None, default 50
        Number of principal components retained.
    """

    name = "pca"

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

    def fit_transform(
        self,
        data: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
        **params,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        n_components = _resolve_n_components(
            self.n_components, n_dim, max_allowed=min(data.shape) - 1,
        )

        U, s, Vh = _truncated_svd(data, n_components)
        scores = np.diag(s) @ Vh  # (n_components, T') — PC time courses

        meta = {
            "dynamics": "pca",
            "n_components": n_components,
            "input_shape": tuple(data.shape),
            "output_shape": tuple(scores.shape),
            "singular_values": s.tolist(),
            "components_": U,  # (n_features, n_components) — left singular vectors
            "elapsed_time": time.time() - t0,
        }
        return scores, meta


# ---------------------------------------------------------------------------
# PCA + ICA (MNE on real channels / sklearn on Hankel PCs)
# ---------------------------------------------------------------------------

class PCAICADynamics:
    """
    ICA decomposition with two mutually exclusive branches.

    * **Without Hankel** (legacy Stage I): MNE ICA + ICLabel artefact
      rejection over the real channel signals via
      :func:`eeg_preprocessing.run_full_preprocessing`.  Requires
      ``ctx.raw``.
    * **With Hankel** (Empate 1): truncated SVD of H to ``n_components``
      (default 50), whitening ``Z = Σ^{-1/2} Vᵀ`` — implemented as the
      mathematically equivalent standardisation of the PC time courses —
      and ``sklearn.decomposition.FastICA`` over the whitened PCs.  MNE
      ICA / ICLabel are **not** used (virtual delay channels have no
      scalp topography) and no artefact rejection is applied (input data
      was already cleaned with GEDAI).

    Parameters (via ``stage2_params``)
    ----------------------------------
    n_components : int | None
        Without Hankel: MNE ICA components (``None`` → all channels − 1).
        With Hankel: SVD rank before FastICA (default 50).
    ica_method : str, default "picard"
        MNE ICA algorithm (no-Hankel branch only).
    ica_random_state : int | None, default 42
    retained_labels : list[str] | None
        ICLabel classes to keep (no-Hankel branch only).
    ica_solver : str, default "fastica"
        Solver for the Hankel branch (currently only ``"fastica"``).
    """

    name = "pca_ica"

    def __init__(
        self,
        n_components: int | float | None = None,
        ica_method: str = "picard",
        ica_random_state: int | None = 42,
        retained_labels: list[str] | None = None,
        ica_solver: str = "fastica",
    ):
        self.n_components = n_components
        self.ica_method = ica_method
        self.ica_random_state = ica_random_state
        self.retained_labels = retained_labels
        self.ica_solver = ica_solver

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        data: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
        **params,
    ) -> tuple[np.ndarray, dict]:
        if ctx.has_hankel:
            return self._fit_hankel_branch(data, ctx=ctx, n_dim=n_dim)
        return self._fit_mne_branch(ctx=ctx, n_dim=n_dim)

    # ------------------------------------------------------------------
    # Branch A — no Hankel: legacy MNE ICA + ICLabel
    # ------------------------------------------------------------------
    def _fit_mne_branch(
        self,
        *,
        ctx: PipelineContext,
        n_dim: int,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        if ctx.raw is None:
            raise ValueError(
                "[Stage2/pca_ica] Without Hankel, stage2='pca_ica' requires "
                "an mne.Raw object (MNE ICA needs real channels with names "
                "and positions). Pass a Raw object to extract_latent_space, "
                "or choose stage2='pca' / 'dmd' / 'diffusion_maps'."
            )

        # Lazy import: keeps the module usable when mne_icalabel is absent
        # and the MNE branch is never exercised.
        from src.latent_space_extraction.eeg_preprocessing import (
            run_full_preprocessing,
        )

        prep = run_full_preprocessing(
            ctx.raw,
            l_freq=ctx.l_freq,
            h_freq=ctx.h_freq,
            n_components=self.n_components,
            ica_method=self.ica_method,
            ica_random_state=self.ica_random_state,
            retained_labels=self.retained_labels,
            verbose=ctx.verbose,
        )

        Y = prep["Y"]  # (D, T), zero-mean rows
        meta = {
            "dynamics": "pca_ica",
            "branch": "mne_ica_iclabel",
            "input_shape": (prep["n_channels"], prep["T"]),
            "output_shape": tuple(Y.shape),
            "ica_method": self.ica_method,
            "excluded_indices": prep["excluded_indices"],
            "kept_indices": prep["kept_indices"],
            "labels": prep["labels"],
            "preprocessing": prep,  # full legacy Stage-I dict
            "elapsed_time": time.time() - t0,
        }
        return Y, meta

    # ------------------------------------------------------------------
    # Branch B — Hankel: SVD whitening + sklearn FastICA (Empate 1)
    # ------------------------------------------------------------------
    def _fit_hankel_branch(
        self,
        H: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        if self.ica_solver != "fastica":
            raise ValueError(
                f"[Stage2/pca_ica] ica_solver={self.ica_solver!r} not "
                f"supported in the Hankel branch; use 'fastica' (sklearn)."
            )

        # n_components: SVD rank before ICA (default 50, spec §5 Empate 1)
        if isinstance(self.n_components, float):
            raise ValueError(
                "[Stage2/pca_ica] float n_components (variance fraction) is "
                "only supported by the MNE branch (no Hankel)."
            )
        n_components = _resolve_n_components(
            self.n_components, n_dim, max_allowed=min(H.shape) - 1,
            default=50,
        )

        # (a) Truncated SVD of H
        print(f"  [Stage2/pca_ica] Step (a): Truncated SVD of H {H.shape} → rank {n_components}...")
        sys.stdout.flush()
        U, s, Vh = _truncated_svd(H, n_components)
        print(f"  [Stage2/pca_ica] SVD done ({time.time() - t0:.2f}s). Whitening...")
        sys.stdout.flush()

        # (b) Whitened PC matrix
        Z = np.diag(s) @ Vh                    # PC time courses Σ Vᵀ
        Z = Z / Z.std(axis=1, keepdims=True)   # whitened: unit-variance rows

        # (c) sklearn FastICA over the whitened PCs.
        print(f"  [Stage2/pca_ica] Step (c): FastICA on {n_components} whitened PCs ({Z.shape[1]:,} time points)...")
        sys.stdout.flush()
        from sklearn.decomposition import FastICA

        ica = FastICA(
            n_components=n_components,
            whiten="unit-variance",
            random_state=self.ica_random_state,
            max_iter=1000,
            tol=1e-4,
        )
        S = ica.fit_transform(Z.T)  # (T', n_components) source time series
        Y = S.T                     # (D, T') — Stage-2 output convention
        Y = Y - Y.mean(axis=1, keepdims=True)

        # Channel-space mixing matrix for influence analysis:
        # Full map: H → U Σ Vᵀ → Z → ICA → S
        # S = Z @ W_ica.T  where W_ica = ica.components_ (unmixing)
        # Z = diag(s) @ Vh  (PC time courses)
        # So S.T ≈ H.T @ (Vh.T * s) @ W_ica.T ... but for influence we
        # store the SVD left singular vectors (same as PCA) so that
        # _try_hankel_aggregation can reduce them to channel space.
        # The ICA unmixing is nonlinear w.r.t. channels; storing U gives
        # the PCA-level channel influence as best linear approximation.
        meta = {
            "dynamics": "pca_ica",
            "branch": "svd_whitening_fastica_sklearn",
            "input_shape": tuple(H.shape),
            "output_shape": tuple(Y.shape),
            "svd_rank": n_components,
            "singular_values": s.tolist(),
            "components_": U,  # (n_features, n_components) — SVD left vectors (PCA-level)
            "ica_solver": "fastica",
            "ica_random_state": self.ica_random_state,
            "n_iter_": int(getattr(ica, "n_iter_", -1)),
            "elapsed_time": time.time() - t0,
        }
        return Y, meta


# ---------------------------------------------------------------------------
# DMD
# ---------------------------------------------------------------------------

class DMDDynamics:
    """
    Dynamic Mode Decomposition with/without the Hankel embedding.

    * **With Hankel** (Empate 4.1): calls
      :func:`hankel_dmd_extractor.eeg_hankel_dmd_core` directly on the
      original channel matrix (kept in ``ctx.stage1_meta['input_data']``)
      — this preserves the legacy numerics exactly (average-reference +
      drop-last-channel).  Returns the POD-mode scores of shape
      ``(rank, T')``.
    * **Without Hankel** (Empate 4.2): direct DMD on X,
      ``X' ≈ A X, A = X' X†`` with eigendecomposition of the
      ``(N_c × N_c)`` operator; modes are ordered by |frequency| and the
      projection of X onto the dominant modes is returned.  This is a
      low-order **"dynamic PCA"**: an order-1 autoregressive model in
      channel space that does *not* capture temporal memory or nonlinear
      attractors.

    Parameters (via ``stage2_params``)
    ----------------------------------
    rank : int | None
        Number of DMD/POD modes retained (D of the Stage-2 output).
        ``None`` → 10.  For exact legacy ``hankel_dmd`` behaviour the
        orchestrator maps ``rank = n_dim``.
    """

    name = "dmd"

    def __init__(self, rank: int | None = None):
        self.rank = rank

    def fit_transform(
        self,
        data: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
        **params,
    ) -> tuple[np.ndarray, dict]:
        if ctx.has_hankel:
            return self._fit_hankel_branch(ctx=ctx, n_dim=n_dim)
        return self._fit_direct_branch(data, ctx=ctx, n_dim=n_dim)

    # ------------------------------------------------------------------
    # With Hankel — legacy eeg_hankel_dmd_core
    # ------------------------------------------------------------------
    def _fit_hankel_branch(
        self,
        *,
        ctx: PipelineContext,
        n_dim: int,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        X = ctx.stage1_meta.get("input_data")
        depth = ctx.stage1_meta.get("depth")
        if X is None or depth is None:
            raise ValueError(
                "[Stage2/dmd] Hankel branch requires Stage-1 metadata with "
                "'input_data' and 'depth' (run stage1='hankel' first)."
            )
        if ctx.dt is None:
            raise ValueError("[Stage2/dmd] ctx.dt is required for DMD spectra.")

        rank = _resolve_n_components(
            self.rank, n_dim, max_allowed=(X.shape[0] - 1) * depth,
            default=10,
        )

        # eeg_hankel_dmd_core handles the single-channel duplication and
        # the average-reference + drop-last-channel internally.
        if X.shape[0] == 1:
            V = np.vstack([X, X.copy()])
        else:
            V = X

        L, s_vals, H, osc, scores = eeg_hankel_dmd_core(V, depth, rank, ctx.dt)

        scores = np.asarray(scores, dtype=np.float64)  # (rank, T'), real

        meta = {
            "dynamics": "dmd",
            "branch": "hankel_eeg_hankel_dmd_core",
            "embedding_depth": depth,
            "latent_rank": rank,
            "input_shape": tuple(X.shape),
            "output_shape": tuple(scores.shape),
            "hankel_shape": tuple(H.shape),
            "singular_values": s_vals.tolist(),
            # Spectral metadata (Empate 6 / legacy latent_scores)
            "frequencies_hz": osc["freq"].real.tolist(),
            "damping_rates": osc["damp"].real.tolist(),
            "eigenvalues": osc["lambda"].tolist(),
            "continuous_eigenvalues": osc["omega"].tolist(),
            "modes": np.real(L),  # (n_hankel_cols, rank) — left DMD modes
            "elapsed_time": time.time() - t0,
        }
        return scores, meta

    # ------------------------------------------------------------------
    # Without Hankel — direct DMD on X (spatial AR-1)
    # ------------------------------------------------------------------
    def _fit_direct_branch(
        self,
        X: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        if ctx.dt is None:
            raise ValueError("[Stage2/dmd] ctx.dt is required for DMD spectra.")

        n_ch, n_times = X.shape
        rank = _resolve_n_components(
            self.rank, n_dim, max_allowed=min(n_ch, n_times - 1),
            default=min(10, n_ch),
        )

        X1 = X[:, :-1]
        X2 = X[:, 1:]
        # Reduced propagator: A = X2 X1^†   (N_c × N_c)
        A = X2 @ np.linalg.pinv(X1)

        lam, W = np.linalg.eig(A)
        omega = np.log(lam.astype(complex)) / ctx.dt
        freq = np.imag(omega) / (2 * np.pi)
        damp = np.real(omega)

        # Order modes by |frequency| (ascending): dominant slow modes first.
        order = np.argsort(np.abs(freq))
        W_sorted = W[:, order]

        # Modal coordinates: projection of X onto the left modal basis.
        # Empate 5.1: complex DMD output → real part before Stage 3.
        coords = np.linalg.pinv(W_sorted) @ X   # (N_c, T), complex in general
        scores = np.real(coords[:rank, :])      # (rank, T)
        scores = scores - scores.mean(axis=1, keepdims=True)

        meta = {
            "dynamics": "dmd",
            "branch": "direct_ar1_channel_space",
            "note": (
                "DMD without Hankel is an order-1 autoregressive model in "
                "channel space ('dynamic PCA'); it does not capture "
                "temporal memory or nonlinear attractors."
            ),
            "latent_rank": rank,
            "input_shape": tuple(X.shape),
            "output_shape": tuple(scores.shape),
            "frequencies_hz": freq.real[order][:rank].tolist(),
            "damping_rates": damp.real[order][:rank].tolist(),
            "eigenvalues": lam[order][:rank].tolist(),
            "modes": np.real(W_sorted[:, :rank]),  # (N_c, rank) — right eigenvectors
            "elapsed_time": time.time() - t0,
        }
        return scores, meta


# ---------------------------------------------------------------------------
# Diffusion Maps
# ---------------------------------------------------------------------------

class DiffusionMapsDynamics:
    """
    Diffusion Maps (Coifman & Lafon 2006) via pyDiffMap.

    * **Without Hankel** (Empate 3): the DM operates over *time instants* —
      each column of X is a point in ``R^{N_c}``.  Dimensionality is low
      (≈ 50), so k-NN is efficient without pre-PCA.  This is exactly the
      legacy branch.
    * **With Hankel** (Empate 2, NLSA): the Hankel matrix is first
      truncated via SVD to ``svd_rank`` components and the DM is built on
      the points ``V_r Σ_r`` in ``R^{svd_rank}`` (Nonlinear Laplacian
      Spectral Analysis: linear SSA over the Hankel matrix + nonlinear DM
      over the principal modes).  ``sigma`` is estimated via the
      Berry-Giannakis-Harlim (BGH) method when ``None``.

    Parameters (via ``stage2_params``)
    ----------------------------------
    n_components : int | None, default 10
        Number of non-trivial diffusion coordinates retained (D of the
        Stage-2 output).  pyDiffMap already excludes the trivial
        eigenvalue λ₀ = 1.
    svd_rank : int | None
        **Required (with default 50) for the Hankel branch**: SVD
        truncation rank of H before the DM (NLSA).
    sigma : float | None
        Gaussian-kernel bandwidth; ``None`` → BGH auto-estimation.
    k : int, default 100
        Nearest neighbours for the sparse affinity matrix.
    diffusion_time : float, default 0.0
        Diffusion time t ≥ 0 (multiscale filtering).
    alpha : float, default 0.5
        Density normalisation (0.0 = Laplacian Eigenmaps, 0.5 = classical
        DM, 1.0 = Fokker-Planck).
    """

    name = "diffusion_maps"

    def __init__(
        self,
        n_components: int | None = None,
        svd_rank: int | None = 50,
        sigma: float | None = None,
        k: int = 100,
        diffusion_time: float = 0.0,
        alpha: float = 0.5,
    ):
        self.n_components = n_components
        self.svd_rank = svd_rank
        self.sigma = sigma
        self.k = k
        self.diffusion_time = diffusion_time
        self.alpha = alpha

    def fit_transform(
        self,
        data: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
        **params,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        n_samples = data.shape[1]
        n_components = _resolve_n_components(
            self.n_components, n_dim, max_allowed=n_samples - 2, default=10,
        )

        if ctx.has_hankel:
            # --- Empate 2 (NLSA) -------------------------------------
            # Truncate H to svd_rank via SVD; the DM input is the matrix
            # of points V_r Σ_r in R^{svd_rank} (one point per column of H).
            svd_rank = self.svd_rank
            if svd_rank is None:
                svd_rank = 50
            svd_rank = int(svd_rank)
            U, s, Vh = _truncated_svd(data, min(svd_rank, min(data.shape) - 1))
            points = (Vh.T * s)  # (T', svd_rank) ≡ columns of V_r Σ_r
            dm_input = points.T  # (svd_rank, T') — features × samples
            branch = "nlsa_svd_then_dm"
            extra_meta = {
                "svd_rank": int(dm_input.shape[0]),
                "singular_values": s.tolist(),
            }
        else:
            # --- Empate 3 --------------------------------------------
            # DM directly over time instants of X (each column a point in
            # R^{N_c}); extract_diffusion_maps_latent_space transposes
            # internally — identical to the legacy branch.
            dm_input = data
            branch = "dm_over_time_instants"
            extra_meta = {}

        latent_rows, dm_meta = extract_diffusion_maps_latent_space(
            X=dm_input,
            n_dim=n_components,
            sigma=self.sigma,
            k=self.k,
            diffusion_time=self.diffusion_time,
            alpha=self.alpha,
            sfreq=ctx.sfreq,
        )
        # latent_rows: (T', n_components) → transpose to (D, T')
        Y = np.asarray(latent_rows.T, dtype=np.float64)
        Y = Y - Y.mean(axis=1, keepdims=True)

        meta = {
            "dynamics": "diffusion_maps",
            "branch": branch,
            "n_components": n_components,
            "input_shape": tuple(data.shape),
            "output_shape": tuple(Y.shape),
            "sigma_used": dm_meta.get("sigma_used"),
            "epsilon_fitted": dm_meta.get("epsilon_fitted"),
            "k_neighbors": dm_meta.get("k_neighbors"),
            "diffusion_time": dm_meta.get("diffusion_time"),
            "alpha": dm_meta.get("alpha"),
            "eigenvalues": dm_meta.get("eigenvalues"),
            "eigenvalues_latent": dm_meta.get("eigenvalues_latent"),
            "spectral_gap": dm_meta.get("spectral_gap"),
            "elapsed_time": time.time() - t0,
            **extra_meta,
        }
        return Y, meta


# ---------------------------------------------------------------------------
# CD-HSA Specific Modes
# ---------------------------------------------------------------------------

class CDHSASpecificModesDynamics:
    """
    Projection onto CD-HSA condition-specific modes (Step D).

    Requires Hankel embedding (Stage 1 = ``"hankel"``).  The input Hankel
    matrix ``H ∈ R^{p × T}`` (where ``p = n_channels × depth``) is first
    lifted to block-Hankel space ``H_block ∈ R^{p·L × T-L+1}`` by stacking
    ``L`` temporally lagged copies — matching the internal representation
    used by CD-HSA Step A.  The block-Hankel is then projected onto the
    top-``n_components`` specific modes of the target condition::

        α = W_sel.T @ H_block    →  (n_components, T - L + 1)

    The modes ``W_specific`` live in the block-Hankel space
    (dimension ``p·L``) and were learned by CD-HSA from pooled
    residuals across all super-subjects.

    Parameters (via ``stage2_params``)
    ----------------------------------
    mode_map_path : str (required)
        Path to the ``mode_map.json`` generated by
        ``src.cdhsa.extract_mode_indices``.  Contains per-condition mode
        indices, eigenvalues, and metadata (including ``L``, ``p_hankel``
        and the NPZ file path).
    npz_path : str | None
        Explicit path to ``cdhsa_arrays.npz``.  When ``None``, the NPZ is
        resolved from the mode_map's ``metadata.source_dir``.
    condition : str
        Name of the target condition/task (e.g. ``"eyesclosed"``).
        **Auto-injected by the pipeline from ``--task``** via
        ``_resolve_pipeline_spec``.  Matched against the keys in the
        mode_map's ``conditions`` dict (which corresponds to
        ``metadata.tasks``).  When ``None``, the first condition is used.
    top_n : int | None
        Number of specific modes to project onto.  When ``None`` the
        ``top_n_actual`` from the mode_map is used.
    n_components : int | None
        Alias for ``top_n`` (for interface uniformity with other
        Stage-2 strategies).

    Example ``stage2_params`` for the batch JSON::

        {"mode_map_path": "params/mode_map_session1_eyesclosed_music_100.0s-200.0s.json"}

    The pipeline injects ``"condition": args.task`` automatically,
    so the same JSON works for all tasks in the batch.
    """

    name = "cdhsa_specific_modes"

    def __init__(
        self,
        mode_map_path: str | None = None,
        npz_path: str | None = None,
        condition: str | None = None,
        top_n: int | None = None,
        n_components: int | None = None,
    ):
        self.mode_map_path = mode_map_path
        self.npz_path = npz_path
        self.condition = condition
        # n_components is the canonical Stage-2 name; allow both
        self.top_n = n_components if top_n is None else top_n

    def fit_transform(
        self,
        data: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
        **params,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()

        if not ctx.has_hankel:
            raise ValueError(
                "[Stage2/cdhsa_specific_modes] Requires Hankel embedding "
                "(stage1='hankel').  CD-HSA modes live in the block-Hankel "
                "space and cannot be applied to raw channel data."
            )

        # ------------------------------------------------------------------
        # 1. Load mode_map.json
        # ------------------------------------------------------------------
        import json

        if self.mode_map_path is None:
            raise ValueError(
                "[Stage2/cdhsa_specific_modes] 'mode_map_path' is required "
                "in stage2_params.  Pass the path to the mode_map.json "
                "generated by src.cdhsa.extract_mode_indices."
            )

        mode_map_path = Path(self.mode_map_path)
        if not mode_map_path.exists():
            raise FileNotFoundError(
                f"[Stage2/cdhsa_specific_modes] mode_map not found: "
                f"{mode_map_path}"
            )

        with open(mode_map_path, encoding="utf-8") as f:
            mode_map = json.load(f)

        # ------------------------------------------------------------------
        # 2. Resolve condition (auto-injected from --task by the pipeline)
        # ------------------------------------------------------------------
        conditions = mode_map["conditions"]
        tasks = list(conditions.keys())
        C = len(tasks)

        if self.condition is not None:
            if self.condition not in conditions:
                raise ValueError(
                    f"[Stage2/cdhsa_specific_modes] condition={self.condition!r} "
                    f"not found in mode_map. Available: {tasks}"
                )
            task_name = self.condition
            c_idx = tasks.index(task_name)
            print(f"  [Stage2/cdhsa_specific_modes] Resolved condition="
                  f"'{task_name}' (index {c_idx}) from --task")
        else:
            # Fallback: use the first condition
            c_idx = 0
            task_name = tasks[0]
            print(f"  [Stage2/cdhsa_specific_modes] WARNING: no condition specified, "
                  f"using first: '{task_name}' (index 0)")

        cond_info = conditions[task_name]
        mode_indices = cond_info["mode_indices_in_W"]

        # ------------------------------------------------------------------
        # 3. Resolve L (block-Hankel lag) from mode_map metadata
        # ------------------------------------------------------------------
        meta_cfg = mode_map["metadata"]["pipeline_config"]
        L = meta_cfg.get("L")
        if L is None:
            raise ValueError(
                "[Stage2/cdhsa_specific_modes] 'L' not found in mode_map "
                "metadata.pipeline_config.  The CD-HSA config must include "
                "the block-Hankel lag parameter."
            )
        L = int(L)

        # ------------------------------------------------------------------
        # 4. Load W_specific and lambda_specific from NPZ
        # ------------------------------------------------------------------
        if self.npz_path is not None:
            npz_file = Path(self.npz_path)
        else:
            source_dir = Path(mode_map["metadata"]["source_dir"])
            npz_file = source_dir / "cdhsa_arrays.npz"

        if not npz_file.exists():
            raise FileNotFoundError(
                f"[Stage2/cdhsa_specific_modes] NPZ not found: {npz_file}"
            )

        npz_data = np.load(npz_file, allow_pickle=True)
        W_all = npz_data["D__W_specific"]
        lam_all = (
            npz_data["D__lambda_specific"]
            if "D__lambda_specific" in npz_data
            else None
        )
        r_specific = npz_data["D__r_specific"]

        # Extract W(c) for this condition
        rc = int(r_specific[c_idx])
        if W_all.ndim == 3:
            W_c = W_all[c_idx, :, :rc]
        else:
            offset = int(np.sum(r_specific[:c_idx]))
            W_c = W_all[:, offset:offset + rc]

        lam_c = None
        if lam_all is not None:
            if lam_all.ndim == 2:
                lam_c = lam_all[c_idx, :rc]
            else:
                offset = int(np.sum(r_specific[:c_idx]))
                lam_c = lam_all[offset:offset + rc]

        # ------------------------------------------------------------------
        # 5. Select top-N modes
        # ------------------------------------------------------------------
        top_n = self.top_n
        if top_n is None:
            top_n = cond_info["top_n_actual"]
        top_n = int(top_n)

        if top_n > rc:
            print(f"  [Stage2/cdhsa_specific_modes] WARNING: top_n ({top_n}) > "
                  f"r_specific ({rc}), clamping.")
            top_n = rc

        if top_n < n_dim:
            raise ValueError(
                f"[Stage2/cdhsa_specific_modes] top_n ({top_n}) must be >= "
                f"n_dim ({n_dim}) so that Stage 3 can select {n_dim} modes."
            )

        # Use the pre-computed indices from mode_map (already sorted by eigenvalue)
        selected_indices = mode_indices[:top_n]
        W_sel = W_c[:, selected_indices]  # (p*L, top_n)

        # ------------------------------------------------------------------
        # 6. Build block-Hankel from the input Hankel matrix H
        # ------------------------------------------------------------------
        # data here is the output of Stage 1: H ∈ R^{p × T}
        # where p = n_channels * hankel_depth
        H = data
        p, T = H.shape

        # Validate expected dimensionality
        p_hankel_expected = mode_map["metadata"]["p_hankel"]
        if p_hankel_expected is not None and p != p_hankel_expected:
            # Check if a channel_intersection was applied
            ch_inter = getattr(ctx, 'channel_intersection', None)
            if ch_inter is not None:
                n_ch_inter = len(ch_inter)
                depth = ctx.stage1_meta.get("depth")
                n_ch_actual = p // depth if (depth and p > 0) else 0
                raise ValueError(
                    f"[Stage2/cdhsa_specific_modes] Hankel row dim mismatch: "
                    f"got {p}, CD-HSA expects {p_hankel_expected}. "
                    f"Channel intersection was applied ({n_ch_inter} channels), "
                    f"but the resulting Hankel ({n_ch_actual} ch x depth) still doesn't match. "
                    f"Check that the hankel_depth and the CD-HSA training used the same "
                    f"channel set and depth."
                )
            print(f"  [Stage2/cdhsa_specific_modes] WARNING: Hankel row dim "
                  f"mismatch: got {p}, CD-HSA expects {p_hankel_expected}. "
                  f"Proceeding anyway — ensure the same hankel_depth was used.")

        # Validate W_sel dimensionality
        if W_sel.shape[0] != p * L:
            raise ValueError(
                f"[Stage2/cdhsa_specific_modes] W_sel row dim "
                f"({W_sel.shape[0]}) != p * L ({p} * {L} = {p * L}). "
                f"Check that hankel_depth and L match the CD-HSA run."
            )

        if T <= L:
            raise ValueError(
                f"[Stage2/cdhsa_specific_modes] Not enough time points: "
                f"T={T}, L={L}. Need T > L for block-Hankel."
            )

        print(f"  [Stage2/cdhsa_specific_modes] Building block-Hankel: "
              f"({p} x {T}) + L={L} → ({p * L} x {T - L + 1})")
        sys.stdout.flush()

        H_block = _build_block_hankel(H, L)  # (p*L, T-L+1)

        # ------------------------------------------------------------------
        # 7. Project onto specific modes
        # ------------------------------------------------------------------
        print(f"  [Stage2/cdhsa_specific_modes] Projecting onto {top_n} "
              f"specific modes of condition '{task_name}' (c_idx={c_idx})...")
        sys.stdout.flush()

        Y = W_sel.T @ H_block  # (top_n, T-L+1)
        Y = np.asarray(Y, dtype=np.float64)
        Y = Y - Y.mean(axis=1, keepdims=True)  # zero-mean rows

        # ------------------------------------------------------------------
        # 8. Collect eigenvalues for selected modes
        # ------------------------------------------------------------------
        selected_eigenvalues = None
        if lam_c is not None:
            selected_eigenvalues = [float(lam_c[i]) for i in selected_indices]

        # ------------------------------------------------------------------
        # 9. Compute mode norms for diagnostics
        # ------------------------------------------------------------------
        mode_norms = np.linalg.norm(W_sel, axis=0).tolist()

        # ------------------------------------------------------------------
        # 9b. Load common-subspace info from A6 (step A6 of CD-HSA)
        # ------------------------------------------------------------------
        # The common modes are the r0 first SVD directions validated by
        # the null-distribution test in Step A6.  They are *not* the
        # "remaining" specific modes — they come from a different
        # decomposition stage entirely.
        #
        # Data sources (tried in order):
        #   1. mode_map["common_subspace"] — JSON-safe dump of A6__* keys
        #   2. NPZ directly — A6__r0, A6__lambda0, A6__U0 (if stored)
        r0 = None
        common_eigenvalues = None  # lambda0: eigenvalues of the r0 common directions
        W_common = None            # U0: (p, r0) common basis vectors (if available)

        common_subspace = mode_map.get("common_subspace", {})
        if common_subspace:
            r0_raw = common_subspace.get("r0")
            if r0_raw is not None:
                r0 = int(r0_raw)
            lambda0_raw = common_subspace.get("lambda0")
            if lambda0_raw is not None:
                if isinstance(lambda0_raw, list):
                    common_eigenvalues = [float(v) for v in lambda0_raw]
                else:
                    common_eigenvalues = [float(lambda0_raw)]
            # U0 may or may not be in the JSON (large array);
            # fall back to NPZ if needed for spatial structure plots.
            if "U0" in common_subspace:
                U0_raw = common_subspace["U0"]
                if isinstance(U0_raw, list):
                    W_common = np.array(U0_raw, dtype=np.float64)

        # If r0 not found in mode_map JSON, try NPZ directly
        if r0 is None and "A6__r0" in npz_data:
            r0 = int(npz_data["A6__r0"])
        if common_eigenvalues is None and "A6__lambda0" in npz_data:
            lam0 = npz_data["A6__lambda0"]
            common_eigenvalues = lam0.flatten().tolist()
        if W_common is None and "A6__U0" in npz_data:
            W_common = npz_data["A6__U0"].astype(np.float64)

        n_common = r0 if r0 is not None else 0

        # Compute common mode norms if W_common is available
        common_mode_norms = None
        if W_common is not None and n_common > 0:
            common_mode_norms = np.linalg.norm(W_common, axis=0).tolist()

        print(f"  [Stage2/cdhsa_specific_modes] Mode breakdown: "
              f"r0={r0} common (A6) + {top_n} specific (D, of {rc} total) = "
              f"{n_common + rc} modes")

        # ------------------------------------------------------------------
        # 10. Metadata
        # ------------------------------------------------------------------
        n_time_lost_block = L - 1

        meta = {
            "dynamics": "cdhsa_specific_modes",
            "branch": "block_hankel_projection",
            "condition": task_name,
            "condition_index": c_idx,
            "L": L,
            "top_n": top_n,
            "mode_indices": selected_indices,
            "eigenvalues": selected_eigenvalues,
            "all_condition_eigenvalues": (
                lam_c.tolist() if lam_c is not None else None
            ),
            "mode_norms": mode_norms,
            "r0_common": r0,
            "common_eigenvalues": common_eigenvalues,
            "common_mode_norms": common_mode_norms,
            "n_common_modes": n_common,
            "W_common": W_common,
            "mode_map_path": str(mode_map_path),
            "npz_path": str(npz_file),
            "W_sel_shape": list(W_sel.shape),
            "input_shape": tuple(H.shape),
            "hankel_block_shape": tuple(H_block.shape),
            "output_shape": tuple(Y.shape),
            "n_time_lost_by_block_hankel": n_time_lost_block,
            "total_specific_modes": rc,
            "p_hankel": p,
            "n_channels": ctx.stage1_meta.get("n_channels"),
            "hankel_depth": ctx.stage1_meta.get("depth"),
            "channel_intersection": (
                list(ctx.channel_intersection) if ctx.channel_intersection else None
            ),
            "elapsed_time": time.time() - t0,
            # Keep W_sel for plotting (channel-influence-like analysis)
            "W_sel": W_sel,
        }

        print(f"  [Stage2/cdhsa_specific_modes] Done: {H.shape} → "
              f"block {H_block.shape} → Y {Y.shape} ({time.time() - t0:.2f}s)")
        sys.stdout.flush()

        return Y, meta


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

STAGE2_REGISTRY: dict[str, type] = {
    "pca": PCADynamics,
    "pca_ica": PCAICADynamics,
    "dmd": DMDDynamics,
    "diffusion_maps": DiffusionMapsDynamics,
    "cdhsa_specific_modes": CDHSASpecificModesDynamics,
}


def build_stage2(name: str, params: dict | None = None):
    """
    Instantiate a Stage-2 dynamics class by name.

    Raises
    ------
    ValueError
        If the dynamics name is unknown.
    """
    params = dict(params or {})
    if name not in STAGE2_REGISTRY:
        valid = ", ".join(repr(k) for k in STAGE2_REGISTRY)
        raise ValueError(
            f"[Stage2] Unknown dynamics {name!r}. Valid options: {valid}"
        )
    return STAGE2_REGISTRY[name](**params)

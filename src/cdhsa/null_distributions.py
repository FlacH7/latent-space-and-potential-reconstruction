"""
null_distributions.py - Null distributions for CD-HSA common rank
================================================================

Two null models for calibrating the common rank r0:

1. **Random subspace null** (from MATLAB): replaces each U_sc with a
   Haar-random orthonormal basis of the same ambient dimension d and
   local rank r_sc. Preserves the rank structure but NOT the Hankel
   temporal correlation.

2. **Hankel-preserving null** (recommended by tutor, NOT in MATLAB code):
   applies a random channel rotation Q_sc ∈ O(p), lifted to I_L ⊗ Q_sc,
   preserving temporal autocorrelation. Strictly more conservative.

Dependencies: numpy, cdhsa.a6_common_rank
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.cdhsa.a6_common_rank import (
    common_basis_from_U,
    crossvalidate_common_rank,
)


# ============================================================================
# 1. Random subspace null (faithful MATLAB translation)
# ============================================================================

def random_subspace_null(
    U_cell: list[list[NDArray[np.floating]]],
    r_values: NDArray[np.integer],
    opts: dict | None = None,
) -> dict:
    """
    Haar-random null preserving every local rank r_sc.

    For every null replicate, each observed local basis U_sc is replaced
    by an independent random orthonormal basis having the SAME ambient
    dimension d and the SAME local rank r_sc.

    Parameters
    ----------
    U_cell : list of list of arrays
        U_cell[s][c] shape (d, r_sc) — observed local bases.
    r_values : array of int
        Candidate rank values.
    opts : dict, optional
        n_null (int, default 100): number of null replicates.
        alpha (float, default 0.05): quantile level for threshold.
        n_folds (int, default 5): folds for CV within null.
        seed (int, default 11): RNG seed.

    Returns
    -------
    null : dict with keys:
        lambda : ndarray (n_null, qmax) — null commonality eigenvalues.
        cv_min : ndarray (n_null, len(r_values)) — null CV_min values.
        lambda_q : ndarray (qmax,) — (1-alpha) quantile per position.
        cv_min_q : ndarray (len(r_values),) — (1-alpha) quantile.
        alpha, n_null

    CRITICAL ANALYSIS vs MATLAB
    ---------------------------
    Faithful translation of ``random_subspace_null.m``.

    **Known limitation (tutor's criticism):**
    This null generates completely random orthonormal bases. In a Hankel
    matrix of rank r_sc, the columns of U_sc have intrinsic temporal
    smoothness (they are Hankel singular vectors). A random basis of the
    same rank has NO temporal structure, so the resulting pooled B matrix
    has systematically lower singular values than expected under any
    plausible null. This makes the null commonality spectrum too low,
    leading to ANTI-CONSERVATIVE inference (too many components declared
    significant).

    Computational note: runtime scales ~linearly with n_null. For
    development use 50-100; for final inference use 1000+.
    """
    if opts is None:
        opts = {}
    n_null = opts.get("n_null", 100)
    alpha = opts.get("alpha", 0.05)
    n_folds = opts.get("n_folds", 5)
    seed = opts.get("seed", 11)

    S = len(U_cell)
    C = len(U_cell[0])
    d = U_cell[0][0].shape[0]
    r_values = np.atleast_1d(np.asarray(r_values, dtype=int))
    qmax = int(np.max(r_values))

    lambda_null = np.full((n_null, qmax), np.nan)
    cvmin_null = np.full((n_null, len(r_values)), np.nan)

    # Fixed fold assignment across null replicates (reduces MC variance)
    cv_opts = {"n_folds": n_folds, "seed": seed + 100000}

    rng = np.random.default_rng(seed)

    for b in range(n_null):
        # Generate random orthonormal bases with same rank
        U0: list[list[NDArray]] = [[None] * C for _ in range(S)]
        for s in range(S):
            for c in range(C):
                rsc = U_cell[s][c].shape[1]
                Z = rng.standard_normal((d, rsc))
                Q, _ = np.linalg.qr(Z)
                U0[s][c] = Q[:, :rsc]

        # Null commonality spectrum
        _, lam = common_basis_from_U(U0, qmax)
        lambda_null[b, : len(lam)] = lam

        # Null CV
        cvb = crossvalidate_common_rank(U0, r_values, cv_opts)
        cvmin_null[b, :] = cvb["min_condition"]

    return {
        "lambda": lambda_null,
        "cv_min": cvmin_null,
        "lambda_q": np.nanquantile(lambda_null, 1 - alpha, axis=0),
        "cv_min_q": np.nanquantile(cvmin_null, 1 - alpha, axis=0),
        "alpha": alpha,
        "n_null": n_null,
    }


# ============================================================================
# 2. Hankel-preserving null (tutor's recommended improvement)
# ============================================================================

def hankel_preserving_null(
    X: list[list[NDArray[np.floating]]],
    L: int,
    U_cell: list[list[NDArray[np.floating]]],
    r_values: NDArray[np.integer],
    opts: dict | None = None,
) -> dict:
    """
    Hankel-preserving null via channel rotation I_L ⊗ Q_sc.

    For each null replicate and each recording (s, c):
      1. Sample a random channel rotation Q_sc ∈ O(p) (Haar measure).
      2. Apply to data: X_rot = Q_sc @ X_{sc}.
      3. Rebuild the Hankel matrix and re-run truncated SVD.

    The lifted operator I_L ⊗ Q_sc preserves the temporal autocorrelation
    structure of the Hankel matrix because it rotates each p-row block
    identically. The resulting null bases have the same spectral
    smoothness as the observed bases, producing a STRICTER (more
    conservative) null than the random subspace null.

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] shape (p, T) — original EEG data.
    L : int
        Hankel embedding depth.
    U_cell : list of list of arrays
        Observed local bases (used only to extract ranks r_sc).
    r_values : array of int
        Candidate rank values.
    opts : dict, optional
        n_null (int, default 100)
        alpha (float, default 0.05)
        n_folds (int, default 5)
        seed (int, default 11)

    Returns
    -------
    null : dict
        Same structure as ``random_subspace_null``.

    CRITICAL ANALYSIS
    -----------------
    This function is NOT in the provided MATLAB code. It was described
    in the tutor's PDF conversation as the correct null model.

    Why this is better than random_subspace_null:
    - A Hankel singular vector of an oscillation at frequency f0 has
      temporal structure (roughly sinusoidal with envelope). A random
      basis vector has NO temporal structure.
    - When pooling random bases, the resulting commonality λ_j reflects
      only the "accidental" alignment of random vectors, which is low.
    - When pooling Hankel-preserving rotated bases, the temporal
      structure is preserved, so accidental alignment is higher,
      producing a more realistic (higher) null threshold.
    - This makes the test MORE CONSERVATIVE: fewer spurious detections.

    Computational cost: O(n_null * S * C * (p*L*K + d²*r_sc)) per
    replicate, vs O(n_null * S * C * d*r_sc) for the random null.
    Typically 5-20x slower because of Hankel reconstruction and SVD.
    """
    from src.cdhsa.a_common_subspace import build_block_hankel, truncated_left_svd

    if opts is None:
        opts = {}
    n_null = opts.get("n_null", 100)
    alpha = opts.get("alpha", 0.05)
    n_folds = opts.get("n_folds", 5)
    seed = opts.get("seed", 11)

    S = len(X)
    C = len(X[0])
    p = X[0][0].shape[0]
    r_values = np.atleast_1d(np.asarray(r_values, dtype=int))
    qmax = int(np.max(r_values))

    lambda_null = np.full((n_null, qmax), np.nan)
    cvmin_null = np.full((n_null, len(r_values)), np.nan)

    cv_opts = {"n_folds": n_folds, "seed": seed + 100000}

    rng = np.random.default_rng(seed)

    for b in range(n_null):
        U0: list[list[NDArray]] = [[None] * C for _ in range(S)]
        for s in range(S):
            for c in range(C):
                rsc = U_cell[s][c].shape[1]

                # Random channel rotation Q ∈ O(p)
                Z = rng.standard_normal((p, p))
                Q, _ = np.linalg.qr(Z)

                # Apply I_L ⊗ Q to the data, then rebuild Hankel + SVD
                X_rot = Q @ X[s][c]
                H_rot = build_block_hankel(X_rot, L)
                H_norm = np.linalg.norm(H_rot, "fro")
                if H_norm > np.finfo(np.float64).eps:
                    H_rot = H_rot / H_norm

                U0[s][c] = truncated_left_svd(H_rot, rsc)

        _, lam = common_basis_from_U(U0, qmax)
        lambda_null[b, : len(lam)] = lam

        cvb = crossvalidate_common_rank(U0, r_values, cv_opts)
        cvmin_null[b, :] = cvb["min_condition"]

    return {
        "lambda": lambda_null,
        "cv_min": cvmin_null,
        "lambda_q": np.nanquantile(lambda_null, 1 - alpha, axis=0),
        "cv_min_q": np.nanquantile(cvmin_null, 1 - alpha, axis=0),
        "alpha": alpha,
        "n_null": n_null,
    }

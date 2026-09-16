"""
cdhsa/c_geometry.py - Direction-sensitive subspace geometry test (Step C extended)
=============================================================================

For each common-mode block W_k and local projector P_sc = U_sc U_sc^T,
define the off-block leakage (tangent component)::

    L_sc,k = (I - W_k W_k^T) P_sc W_k

L_sc,k captures how the local subspace ROTATES relative to the common
subspace. Unlike scalar alignment a = ||U^T w||^2, the tangent component
retains DIRECTION/SIGN information, making balanced rotations detectable.

The omnibus statistic for each block is:

    T_k = S * sum_c ||mean_s L_sc,k - grand_mean||_F^2

Condition labels are permuted within subject, with max-T correction
across blocks. A6.W0 is fixed (estimated from label-blind pooled data).

Dependencies: numpy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Helper: omnibus tangent statistic
# ============================================================================

def _tangent_stat(Lsc: list[list[NDArray]]) -> float:
    """
    Omnibus statistic for a set of tangent components.

    Parameters
    ----------
    Lsc : list of list of arrays, shape (S, C)
        Lsc[s][c] is a matrix of shape (d, dim_block).

    Returns
    -------
    T : float
        S * Σ_c ||mean_s L_sc - grand_mean||_F²
    """
    S = len(Lsc)
    C = len(Lsc[0])
    template_shape = Lsc[0][0].shape

    cond_means = []
    grand = np.zeros(template_shape, dtype=np.float64)

    for c in range(C):
        M = np.zeros(template_shape, dtype=np.float64)
        for s in range(S):
            M += Lsc[s][c]
        M /= S
        cond_means.append(M)
        grand += M / C

    T = 0.0
    for c in range(C):
        D = cond_means[c] - grand
        T += S * np.sum(D ** 2)

    return float(T)


# ============================================================================
# Main: tangent geometry test
# ============================================================================

def cdhsa_tangent_geometry_test(
    R: dict,
    A6: dict,
    blocks: list[NDArray[np.integer]] | None = None,
    opts: dict | None = None,
) -> dict:
    """
    Direction-sensitive subspace geometry test.

    For block W_k and local projector P_sc = U_sc U_sc^T, define::

        L_sc,k = (I - W_k W_k^T) P_sc W_k

    L_sc,k is the off-block leakage / tangent component. Its Frobenius
    norm is invariant to internal orthogonal rotation of W_k. Unlike
    scalar alignment w'Pw, its SIGN/DIRECTION is retained, so a balanced
    rotation around a pooled midpoint is detectable.

    The omnibus statistic for each block is::

        T_k = S * Σ_c ||mean_s L_sc,k - grand_mean L||_F²

    Condition labels are permuted within subject, with max-T correction
    across blocks. A6.W0 is fixed because it was estimated from
    label-blind pooled data.

    Parameters
    ----------
    R : dict
        Output of ``cdhsa_A1_A5``. Must contain 'U', 'S', 'C'.
    A6 : dict
        Output of ``cdhsa_A6_common_rank``. Must contain 'W0', 'r0'.
    blocks : list of arrays, optional
        Each array contains 1-based indices into A6.W0.
        Default: all modes as one block {1, 2, ..., r0}.
    opts : dict, optional
        n_perm (int, default 5000)
        seed (int, default 1234)
        alpha (float, default 0.05)

    Returns
    -------
    G : dict with keys:
        blocks : list of arrays
        T_obs : ndarray (Kb,) — observed tangent statistic.
        p_uncorrected : ndarray (Kb,)
        p_maxT : ndarray (Kb,) — max-T corrected p-value.
        sig_maxT : ndarray (Kb,) bool
        null_T : ndarray (n_perm, Kb)
        null_maxT : ndarray (n_perm,)
        alpha, n_perm, seed, S, C, d
        mean_difference_norm, effect_ratio : ndarray (Kb,)  [only if C=2]

    CRITICAL ANALYSIS vs MATLAB
    ---------------------------
    Faithful translation of ``cdhsa_tangent_geometry_test.m``.

    Key mathematical properties:
    1. **Invariance**: ||L_sc,k||_F² = ||P_sc W_k||_F² - ||W_k^T P_sc W_k||_F².
       The tangent norm equals the total projected norm minus the in-block
       norm. This is invariant to right-multiplying W_k by any orthogonal matrix.
    2. **Direction sensitivity**: Unlike a_{sc,j} = ||U^T w_j||² which is
       always non-negative, L_sc,k is a MATRIX that can point in any direction.
       This means two conditions with similar alignment magnitude but different
       alignment DIRECTIONS produce a large T_k.
    3. **Effect size (C=2)**: The mean_difference_norm and effect_ratio are
       descriptive (not used for inference). They quantify the magnitude of
       the mean tangent difference relative to within-subject variability.
    """
    if opts is None:
        opts = {}
    n_perm = opts.get("n_perm", 5000)
    seed = opts.get("seed", 1234)
    alpha_val = opts.get("alpha", 0.05)

    if A6["r0"] < 1:
        raise ValueError("No supported common subspace (A6.r0 = 0).")

    # Default: all modes as one block
    if blocks is None:
        blocks = [np.arange(1, A6["r0"] + 1, dtype=int)]

    if not isinstance(blocks, list) or not all(
        isinstance(b, np.ndarray) for b in blocks
    ):
        raise ValueError("blocks must be a list of numpy arrays.")

    S = R["S"]
    C = R["C"]
    Kb = len(blocks)
    d = A6["W0"].shape[0]

    # ---- Compute tangent components L{s,c,k} ----
    L: list[list[list[NDArray]]] = [
        [[None] * Kb for _ in range(C)] for _ in range(S)
    ]

    for k in range(Kb):
        idx = np.atleast_1d(np.asarray(blocks[k], dtype=int))
        idx = idx - 1  # Convert 1-based to 0-based
        if np.any(idx < 0) or np.any(idx >= A6["r0"]):
            raise ValueError(f"Block {k} contains invalid indices.")
        Wk = A6["W0"][:, idx]  # (d, dim_k)
        WWt = Wk @ Wk.T  # (d, d)

        for s in range(S):
            for c in range(C):
                Ui = R["U"][s][c]  # (d, r_sc)
                # PW = U (U^T W_k)
                PW = Ui @ (Ui.T @ Wk)  # (d, dim_k)
                # L = PW - W_k W_k^T PW = (I - WW^T) PW
                Lsc = PW - WWt @ PW
                L[s][c][k] = Lsc

    # ---- Observed statistic ----
    T_obs = np.zeros(Kb, dtype=np.float64)
    for k in range(Kb):
        L_k = [[L[s][c][k] for c in range(C)] for s in range(S)]
        T_obs[k] = _tangent_stat(L_k)

    # ---- Null distribution ----
    rng = np.random.default_rng(seed)
    null_T = np.zeros((n_perm, Kb), dtype=np.float64)
    null_maxT = np.zeros(n_perm, dtype=np.float64)

    for b in range(n_perm):
        for k in range(Kb):
            # Permute condition labels within each subject
            L_k = [[L[s][rng.permutation(C)[c]][k] for c in range(C)] for s in range(S)]
            null_T[b, k] = _tangent_stat(L_k)
        null_maxT[b] = np.max(null_T[b, :])

    # ---- p-values ----
    p_unc = np.array([
        (1 + np.sum(null_T[:, k] >= T_obs[k])) / (n_perm + 1)
        for k in range(Kb)
    ])
    p_max = np.array([
        (1 + np.sum(null_maxT >= T_obs[k])) / (n_perm + 1)
        for k in range(Kb)
    ])

    result = {
        "blocks": blocks,
        "T_obs": T_obs,
        "p_uncorrected": p_unc,
        "p_maxT": p_max,
        "sig_maxT": p_max < alpha_val,
        "null_T": null_T,
        "null_maxT": null_maxT,
        "alpha": alpha_val,
        "n_perm": n_perm,
        "seed": seed,
        "S": S,
        "C": C,
        "d": d,
    }

    # ---- Effect sizes for C=2 (descriptive only) ----
    if C == 2:
        mean_diff_norm = np.zeros(Kb, dtype=np.float64)
        effect_ratio = np.zeros(Kb, dtype=np.float64)

        for k in range(Kb):
            diffs = [L[s][1][k] - L[s][0][k] for s in range(S)]
            M = np.mean(diffs, axis=0)
            mean_diff_norm[k] = np.linalg.norm(M, "fro")

            dn_sq = [
                np.linalg.norm(diffs[s] - M, "fro") ** 2 for s in range(S)
            ]
            denom = np.sqrt(np.mean(dn_sq))
            effect_ratio[k] = mean_diff_norm[k] / max(denom, np.finfo(np.float64).eps)

        result["mean_difference_norm"] = mean_diff_norm
        result["effect_ratio"] = effect_ratio

    return result

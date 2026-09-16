"""
cdhsa/d_condition_specific.py - Step D: Condition-specific residual modes
========================================================================

After extracting the common subspace W0 (steps A1-A6), Step D identifies
modes that are reliably present within a condition but NOT captured by
the common subspace.

Algorithm (JIVE-inspired):
  1. For each (s,c), compute the residual basis after removing W0:
     U_res_sc = orthogonalize((I - W0 W0^T) @ U_sc)
  2. Pool residual bases within each condition across subjects:
     B_c = [U_res_1c, ..., U_res_Sc] / sqrt(S)
  3. SVD of B_c to obtain condition-specific directions W_c
  4. Quantify per-subject alignment with condition-specific modes
  5. Test for condition-specific structure via prevalence contrast

Dependencies: numpy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _orthogonalize_residual(
    U: NDArray[np.floating],
    W0: NDArray[np.floating],
    tol: float = 1e-10,
) -> NDArray[np.floating]:
    """
    Remove the projection of U onto W0 and re-orthogonalize.

    Parameters
    ----------
    U : array, shape (d, r_sc)
    W0 : array, shape (d, r0)

    Returns
    -------
    U_res : array, shape (d, r_res)
        Orthonormal basis for the component of U orthogonal to W0.
        r_res <= r_sc - rank(U^T W0) in exact arithmetic.
    """
    # Remove common component: U_res = (I - W0 W0^T) U
    proj = W0 @ (W0.T @ U)  # (d, r_sc)
    U_res = U - proj

    # Re-orthogonalize via thin SVD (numerical stability)
    U_res, s_res, _ = np.linalg.svd(U_res, full_matrices=False)

    # Keep columns with non-negligible singular values
    r_res = int(np.sum(s_res > tol))
    if r_res == 0:
        return np.zeros((U.shape[0], 0), dtype=np.float64)
    return U_res[:, :r_res]


def cdhsa_D_condition_specific_modes(
    X: list[list[NDArray[np.floating]]],
    L: int,
    R: dict,
    A6: dict,
    opts: dict | None = None,
) -> dict:
    """
    Step D: Extract condition-specific residual Hankel modes.

    After the common subspace W0 has been removed, this function:

      1. Computes residual Hankel bases for each recording.
      2. Pools residuals within each condition to find condition-specific
         directions (analogous to JIVE individual structure).
      3. Quantifies alignment of each recording with its condition's
         specific modes.
      4. Reports prevalence contrast: how much more aligned subjects are
         to their OWN condition's modes vs. the other condition's modes.

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] shape (p, T) - original EEG data.
    L : int
        Hankel embedding depth.
    R : dict
        Output of cdhsa_A1_A5. Must contain 'U', 'S', 'C', 'p', 'rank'.
    A6 : dict
        Output of cdhsa_A6_common_rank. Must contain 'W0', 'r0'.
    opts : dict, optional
        max_specific : int (default 10)
            Maximum number of condition-specific modes per condition.
        residual_rank_method : str (default 'gap')
            How to determine the rank of each residual recording:
            - 'local_gap': use a gap in the singular values of (I-W0W0^T)U_sc
            - 'local_threshold': keep SVs > threshold * max_sv
            - 'fixed': use fixed_residual_rank
        residual_rank_threshold : float (default 0.1)
            Threshold for 'local_threshold' method.
        fixed_residual_rank : int (default 5)
            Rank for 'fixed' method.
        prevalence_quantile : float (default 0.10)
            Lower quantile for prevalence computation.

    Returns
    -------
    D : dict with keys:
        W_specific : list of arrays
            W_specific[c] shape (d, r_c) - condition-specific modes.
        lambda_specific : list of arrays
            lambda_specific[c] shape (r_c,) - condition-specific singular values.
        r_specific : ndarray (C,) - number of modes per condition.
        U_residual : list of list of arrays
            U_residual[s][c] shape (d, r_res_sc) - residual basis per recording.
        residual_rank : ndarray (S, C) - residual rank per recording.
        alignment_specific : ndarray (S, C)
            Alignment with OWN condition's specific modes.
        alignment_cross : ndarray (S, C)
            Alignment with the OTHER condition's specific modes.
        prevalence_contrast : ndarray (C,)
            Mean(own alignment) - Mean(cross alignment) per condition.
        S, C, p, d, r0

    DESIGN NOTES
    ------------
    This Step is inspired by JIVE's individual structure estimation.
    The key idea: after removing what is COMMON (W0), what remains may
    contain structure that is reliable WITHIN a condition but not shared
    across conditions. This is precisely the condition-specific signal.

    The prevalence_contrast metric quantifies discriminability:
    - A positive value for condition c means subjects in condition c
      are more aligned with their own specific modes than with the
      other condition's modes.
    - This is descriptive, not inferential. For formal testing,
      use the B/C permutation tests on the specific-mode alignment.
    """
    from src.cdhsa.a_common_subspace import build_block_hankel, truncated_left_svd

    if opts is None:
        opts = {}
    max_specific = opts.get("max_specific", 10)
    res_rank_method = opts.get("residual_rank_method", "local_gap")
    res_rank_threshold = opts.get("residual_rank_threshold", 0.1)
    fixed_res_rank = opts.get("fixed_residual_rank", 5)
    prev_q = opts.get("prevalence_quantile", 0.10)

    S = R["S"]
    C = R["C"]
    p = R["p"]
    d = R["d"]
    r0 = A6["r0"]

    if r0 < 1:
        raise ValueError(
            "Cannot compute condition-specific modes when A6.r0 = 0."
        )

    W0 = A6["W0"]  # (d, r0)

    # ---- Step D1: Compute residual bases ----
    U_residual: list[list[NDArray]] = [[None] * C for _ in range(S)]
    residual_ranks = np.zeros((S, C), dtype=int)

    for s in range(S):
        for c in range(C):
            Ui = R["U"][s][c]  # (d, r_sc)
            U_res = _orthogonalize_residual(Ui, W0)

            # Determine residual rank
            if res_rank_method == "fixed":
                r_res = min(fixed_res_rank, U_res.shape[1])
            elif res_rank_method == "local_threshold":
                if U_res.shape[1] == 0:
                    r_res = 0
                else:
                    _, sv, _ = np.linalg.svd(U_res, full_matrices=False)
                    r_res = int(np.sum(sv > res_rank_threshold * sv[0]))
            elif res_rank_method == "local_gap":
                if U_res.shape[1] <= 1:
                    r_res = U_res.shape[1]
                else:
                    _, sv, _ = np.linalg.svd(U_res, full_matrices=False)
                    ratios = sv[:-1] / np.maximum(sv[1:], 1e-15)
                    # Find first large gap (ratio > 2)
                    gaps = np.where(ratios > 2.0)[0]
                    if len(gaps) > 0 and gaps[0] > 0:
                        r_res = gaps[0] + 1
                    else:
                        # Fallback: keep SVs above 5% of max
                        r_res = int(np.sum(sv > 0.05 * sv[0]))
            else:
                raise ValueError(
                    f"Unknown residual_rank_method: '{res_rank_method}'"
                )

            r_res = min(r_res, U_res.shape[1])
            U_residual[s][c] = U_res[:, :r_res]
            residual_ranks[s, c] = r_res

    # ---- Step D2: Pool residuals within condition, SVD -> W_c ----
    W_specific: list[NDArray] = []
    lambda_specific: list[NDArray] = []
    r_specific = np.zeros(C, dtype=int)

    for c in range(C):
        Ucat_c = np.concatenate(
            [U_residual[s][c] for s in range(S) if U_residual[s][c].shape[1] > 0],
            axis=1,
        )
        if Ucat_c.shape[1] == 0:
            W_specific.append(np.zeros((d, 0), dtype=np.float64))
            lambda_specific.append(np.array([]))
            continue

        B_c = Ucat_c / np.sqrt(S)
        q_c = min(max_specific, d, Ucat_c.shape[1])
        Wc, sv_c, _ = np.linalg.svd(B_c, full_matrices=False)
        Wc = Wc[:, :q_c]
        sv_c = sv_c[:q_c]

        W_specific.append(Wc)
        lambda_specific.append(sv_c)
        r_specific[c] = q_c

    # ---- Step D3: Alignment with own and cross condition modes ----
    alignment_specific = np.zeros((S, C), dtype=np.float64)
    alignment_cross = np.zeros((S, C), dtype=np.float64)

    for s in range(S):
        for c in range(C):
            Ures = U_residual[s][c]  # (d, r_res_sc)
            if Ures.shape[1] == 0 or W_specific[c].shape[1] == 0:
                alignment_specific[s, c] = 0.0
            else:
                G = Ures.T @ W_specific[c]
                alignment_specific[s, c] = np.linalg.norm(G, "fro") ** 2 / max(
                    r_specific[c], 1
                )

            # Cross-condition alignment (align with the OTHER condition's modes)
            c_other = 1 - c if C == 2 else -1
            if c_other >= 0 and W_specific[c_other].shape[1] > 0:
                if Ures.shape[1] > 0:
                    G_cross = Ures.T @ W_specific[c_other]
                    alignment_cross[s, c] = np.linalg.norm(G_cross, "fro") ** 2 / max(
                        r_specific[c_other], 1
                    )

    # ---- Step D4: Prevalence contrast ----
    prevalence_own = np.array([
        np.quantile(alignment_specific[:, c], prev_q) for c in range(C)
    ])
    mean_own = np.mean(alignment_specific, axis=0)
    mean_cross = np.mean(alignment_cross, axis=0) if C == 2 else np.zeros(C)
    prevalence_contrast = mean_own - mean_cross

    return {
        "W_specific": W_specific,
        "lambda_specific": lambda_specific,
        "r_specific": r_specific,
        "U_residual": U_residual,
        "residual_rank": residual_ranks,
        "alignment_specific": alignment_specific,
        "alignment_cross": alignment_cross,
        "mean_own_alignment": mean_own,
        "mean_cross_alignment": mean_cross,
        "prevalence_own": prevalence_own,
        "prevalence_contrast": prevalence_contrast,
        "S": S,
        "C": C,
        "p": p,
        "d": d,
        "r0": r0,
    }

"""
cdhsa/b_energy.py — Steps B/C of CD-HSA: Energy and geometry condition tests
================================================================================

For each fixed common Hankel mode/block (from A6.W0):

  **Step B** — Energy in the ORIGINAL (unnormalized) Hankel matrix.
    E_{sc,k} = ||H_sc^T W_k||_F²  measures how much variance each common
    direction captures in each recording, using the original scale.

  **Step C** — Geometric alignment with the local reliable Hankel subspace.
    a_{sc,k} = ||U_sc^T W_k||_F² / r_sc  (rank-adjusted) measures how much
    of each common direction is geometrically present, SEPARATE from amplitude.

Within-subject permutation tests with max-F correction determine which
blocks show significant condition effects.

CRITICAL DESIGN PRINCIPLE
--------------------------
A6.W0 is held FIXED. Since W0 was estimated from the pooled subject×condition
projectors WITHOUT using condition labels, relabeling conditions within a
subject does NOT change the pooled W0. This justifies the permutation test.

Dependencies: numpy, cdhsa.a_common_subspace, cdhsa.permutation_tests
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.cdhsa.a_common_subspace import build_block_hankel
from src.cdhsa.permutation_tests import within_subject_permutation_rm


# ============================================================================
# Helper: compute all energy and alignment metrics per recording and block
# ============================================================================

def compute_common_mode_metrics(
    X: list[list[NDArray[np.floating]]],
    L: int,
    R: dict,
    A6: dict,
    blocks: list[NDArray[np.integer]],
) -> dict:
    """
    Compute energy and alignment metrics for each recording and common-mode block.

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] shape (p, T) — original (unnormalized) EEG data.
    L : int
        Hankel embedding depth.
    R : dict
        Output of ``cdhsa_A1_A5``. Must contain 'U', 'rank', 'S', 'C', 'p', 'L_used'.
    A6 : dict
        Output of ``cdhsa_A6_common_rank``. Must contain 'W0', 'r0'.
    blocks : list of arrays
        Each element is an array of 1-based indices into A6.W0.
        Example: [np.array([1,2]), np.array([3]), np.array([4,5])].

    Returns
    -------
    M : dict with keys:
        blocks : list of arrays (Kb elements)
        energy_abs : ndarray (S, C, Kb)
            ||H_sc^T W_k||_F² for each recording and block.
        energy_rel : ndarray (S, C, Kb)
            energy_abs / ||H_sc||_F².
        align_raw : ndarray (S, C, Kb)
            ||U_sc^T W_k||_F² (unadjusted).
        align_adj : ndarray (S, C, Kb)
            ||U_sc^T W_k||_F² / r_sc (rank-adjusted).
        local_rank : ndarray (S, C)
        S, C, Kb, d, p : int

    Notes
    -----
    - energy_abs uses the ORIGINAL (unnormalized) Hankel, so it captures
      both geometry AND amplitude. This is the Step B metric.
    - align_adj divides by r_sc to remove the mechanical dependence of
      raw alignment on local rank. A basis of rank r in d dimensions has
      expected alignment r/d with any fixed direction.
    - For a singleton block {j}, energy_abs = ||H^T w_j||² = w_j^T (H H^T) w_j.
    """
    S = R["S"]
    C = R["C"]
    p = R["p"]
    L_used = R["L_used"]
    Kb = len(blocks)
    d = p * L_used

    energy_abs = np.zeros((S, C, Kb), dtype=np.float64)
    energy_rel = np.zeros((S, C, Kb), dtype=np.float64)
    align_raw = np.zeros((S, C, Kb), dtype=np.float64)
    align_adj = np.zeros((S, C, Kb), dtype=np.float64)

    for s in range(S):
        for c in range(C):
            Xi = np.asarray(X[s][c], dtype=np.float64)
            H = build_block_hankel(Xi, L)  # (d, K_hankel)
            Ht = H.T  # (K_hankel, d) — precompute transpose
            total_energy = np.linalg.norm(H, "fro") ** 2

            Ui = R["U"][s][c]  # (d, r_sc)
            r_sc = R["rank"][s, c]

            for k, idx in enumerate(blocks):
                idx = np.atleast_1d(np.asarray(idx, dtype=int)) - 1  # 0-based
                Wk = A6["W0"][:, idx]  # (d, dim_k)

                # Energy: ||H^T W_k||_F²
                proj = Ht @ Wk  # (K_hankel, dim_k)
                energy_abs[s, c, k] = np.linalg.norm(proj, "fro") ** 2
                energy_rel[s, c, k] = energy_abs[s, c, k] / max(
                    total_energy, 1e-300
                )

                # Alignment: ||U_sc^T W_k||_F²
                G = Ui.T @ Wk  # (r_sc, dim_k)
                raw = np.linalg.norm(G, "fro") ** 2
                align_raw[s, c, k] = raw
                align_adj[s, c, k] = raw / max(r_sc, 1)

    return {
        "blocks": blocks,
        "energy_abs": energy_abs,
        "energy_rel": energy_rel,
        "align_raw": align_raw,
        "align_adj": align_adj,
        "local_rank": R["rank"].copy(),
        "S": S,
        "C": C,
        "Kb": Kb,
        "d": d,
        "p": p,
    }


# ============================================================================
# Main: Steps B/C
# ============================================================================

def cdhsa_BC_condition_tests(
    X: list[list[NDArray[np.floating]]],
    L: int,
    R: dict,
    A6: dict,
    opts: dict | None = None,
) -> dict:
    """
    Steps B/C of CD-HSA: condition effects on energy and geometry.

    For each fixed common Hankel mode/block:
      B) energy in the ORIGINAL (unnormalized) Hankel matrix.
      C) geometrical alignment with the local reliable Hankel subspace.
    Within-subject permutation tests with max-F correction across blocks.

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] shape (p, T).
    L : int
        Hankel embedding depth.
    R : dict
        Output of ``cdhsa_A1_A5``.
    A6 : dict
        Output of ``cdhsa_A6_common_rank``. Must have r0 >= 1.
    opts : dict, optional
        blocks : list of arrays — indices into A6.W0 (1-based).
            Default: singleton modes [1], [2], ..., [r0].
        energy_metric : str
            'log_absolute' (default), 'absolute', 'relative', 'log_relative'.
        geometry_metric : str
            'adjusted' (default) or 'raw'.
        n_perm : int (default 5000)
        seed : int (default 1234)
        alpha : float (default 0.05)
        condition_names : list of str — optional names for conditions.

    Returns
    -------
    BC : dict with keys:
        metrics : dict — output from compute_common_mode_metrics.
        energy_data : ndarray (S, C, Kb) — transformed energy matrix.
        geometry_data : ndarray (S, C, Kb) — transformed geometry matrix.
        energy_test : dict — permutation test results for energy.
        geometry_test : dict — permutation test results for geometry.
        rank_test : dict — permutation test results for local rank.
        sig_energy_maxF : ndarray (Kb,) bool
        sig_geometry_maxF : ndarray (Kb,) bool
        block_names : list of str
        condition_names : list of str
        summary : dict — compact numerical summary.

    CRITICAL ANALYSIS vs MATLAB
    ---------------------------
    Faithful translation of ``cdhsa_BC_condition_tests.m``.

    Key design decisions:
    1. **A6.W0 is fixed** — it was estimated from label-blind pooled data,
       so condition permutation within subjects doesn't change it.
    2. **Energy uses unnormalized H** — the A1-A5 pipeline normalizes H/||H||
       before SVD, but energy must use the original scale.
    3. **Rank-adjusted alignment** — divides by r_sc to remove the mechanical
       dependence on local rank.
    4. **Rank is also tested** — treated as a single (K=1) repeated-measures
       outcome, since raw alignment depends on rank.
    5. **Block indices are 1-based** in the API (matching MATLAB convention)
       and converted to 0-based internally.

    The MATLAB ``compute_common_mode_metrics`` function body was not provided
    but its interface and output fields are fully determined from the calling
    code in ``cdhsa_BC_condition_tests.m``.
    """
    if opts is None:
        opts = {}
    if A6["r0"] < 1:
        raise ValueError(
            "A6.r0 = 0. Do not run B/C without a supported common subspace."
        )

    # Default blocks: singleton modes
    if "blocks" not in opts or not opts["blocks"]:
        blocks = [np.array([j], dtype=int) for j in range(1, A6["r0"] + 1)]
    else:
        blocks = [
            np.atleast_1d(np.asarray(b, dtype=int)) for b in opts["blocks"]
        ]

    energy_metric = opts.get("energy_metric", "log_absolute")
    geometry_metric = opts.get("geometry_metric", "adjusted")
    n_perm = opts.get("n_perm", 5000)
    seed = opts.get("seed", 1234)
    alpha = opts.get("alpha", 0.05)
    condition_names = opts.get("condition_names", [])

    # ---- Compute raw metrics ----
    M = compute_common_mode_metrics(X, L, R, A6, blocks)
    Kb = M["Kb"]

    # ---- Transform energy metric ----
    tiny = np.finfo(np.float64).tiny
    if energy_metric == "log_absolute":
        Yenergy = np.log(np.maximum(M["energy_abs"], tiny))
        energy_label = "log absolute Hankel energy"
    elif energy_metric == "absolute":
        Yenergy = M["energy_abs"].copy()
        energy_label = "absolute Hankel energy"
    elif energy_metric == "relative":
        Yenergy = M["energy_rel"].copy()
        energy_label = "relative Hankel energy"
    elif energy_metric == "log_relative":
        Yenergy = np.log(np.maximum(M["energy_rel"], tiny))
        energy_label = "log relative Hankel energy"
    else:
        raise ValueError(f"Unknown energy_metric: '{energy_metric}'")

    # ---- Transform geometry metric ----
    if geometry_metric == "adjusted":
        Ygeom = M["align_adj"].copy()
        geometry_label = "rank-adjusted subspace alignment"
    elif geometry_metric == "raw":
        Ygeom = M["align_raw"].copy()
        geometry_label = "raw subspace alignment"
    else:
        raise ValueError(f"Unknown geometry_metric: '{geometry_metric}'")

    # ---- Permutation tests ----
    energy_test = within_subject_permutation_rm(
        Yenergy, {"n_perm": n_perm, "seed": seed}
    )
    geometry_test = within_subject_permutation_rm(
        Ygeom, {"n_perm": n_perm, "seed": seed + 1}
    )

    # Rank as a repeated-measures outcome (single variable, K=1)
    Yrank = M["local_rank"][:, :, np.newaxis].astype(np.float64)
    rank_test = within_subject_permutation_rm(
        Yrank, {"n_perm": n_perm, "seed": seed + 2}
    )

    # ---- Names ----
    if not condition_names:
        condition_names = [f"Condition {c + 1}" for c in range(M["C"])]
    elif len(condition_names) != M["C"]:
        raise ValueError(
            f"condition_names must have {M['C']} entries, got {len(condition_names)}"
        )

    block_names = []
    for k in range(Kb):
        idx = blocks[k]
        if len(idx) == 1:
            block_names.append(f"W{int(idx[0])}")
        else:
            block_names.append(f"W[{','.join(str(int(i)) for i in idx)}]")

    # ---- Significance masks ----
    sig_energy = energy_test["p_maxF"] < alpha
    sig_geom = geometry_test["p_maxF"] < alpha

    # ---- Assemble output ----
    BC = {
        "metrics": M,
        "energy_metric": energy_metric,
        "geometry_metric": geometry_metric,
        "energy_label": energy_label,
        "geometry_label": geometry_label,
        "energy_data": Yenergy,
        "geometry_data": Ygeom,
        "energy_test": energy_test,
        "geometry_test": geometry_test,
        "rank_test": rank_test,
        "alpha": alpha,
        "sig_energy_maxF": sig_energy,
        "sig_geometry_maxF": sig_geom,
        "block_names": block_names,
        "condition_names": condition_names,
        "summary": {
            "block": block_names,
            "energy_F": energy_test["F_obs"],
            "energy_p_unc": energy_test["p_uncorrected"],
            "energy_p_maxF": energy_test["p_maxF"],
            "geometry_F": geometry_test["F_obs"],
            "geometry_p_unc": geometry_test["p_uncorrected"],
            "geometry_p_maxF": geometry_test["p_maxF"],
        },
    }

    # C=2 effect sizes
    if M["C"] == 2:
        BC["summary"]["energy_difference_C2_minus_C1"] = energy_test[
            "mean_difference_C2_minus_C1"
        ]
        BC["summary"]["energy_cohen_dz"] = energy_test["cohen_dz"]
        BC["summary"]["geometry_difference_C2_minus_C1"] = geometry_test[
            "mean_difference_C2_minus_C1"
        ]
        BC["summary"]["geometry_cohen_dz"] = geometry_test["cohen_dz"]

    return BC

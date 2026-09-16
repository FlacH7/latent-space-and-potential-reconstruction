"""
cdhsa/permutation_tests.py — Within-subject permutation tests for CD-HSA
=====================================================================

Implements repeated-measures permutation inference with max-statistic
correction across multiple outcome blocks (common Hankel modes).

The key design principle: condition labels are permuted WITHIN subject
(never between subjects), preserving the subject-level exchangeability
under H0 that conditions are interchangeable.

Dependencies: numpy
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def within_subject_permutation_rm(
    Y: NDArray[np.floating],
    opts: dict | None = None,
) -> dict:
    """
    Within-subject permutation test for repeated-measures data with max correction.

    For each outcome variable (block) k, computes a one-way repeated-measures
    F statistic. Permutations shuffle condition labels *within* each subject.
    p-values use max-F correction across blocks to control the family-wise
    error rate.

    Parameters
    ----------
    Y : array, shape (S, C, K)
        S subjects, C conditions, K outcome variables (blocks).
    opts : dict, optional
        n_perm (int, default 5000): number of permutation replicates.
        seed (int, default 1234): RNG seed for reproducibility.

    Returns
    -------
    result : dict with keys:
        F_obs : ndarray, shape (K,)
            Observed F statistic for each block.
        p_uncorrected : ndarray, shape (K,)
            Permutation p-value per block (no correction).
        p_maxF : ndarray, shape (K,)
            Max-F corrected p-value: P(max F_null >= F_obs(k)).
        null_F : ndarray, shape (n_perm, K)
            Full null distribution of F statistics.
        null_maxF : ndarray, shape (n_perm,)
            Max across blocks for each null replicate.
        n_perm : int
        seed : int
        mean_difference_C2_minus_C1 : ndarray, shape (K,)  [only if C=2]
            Mean difference condition 2 minus condition 1.
        cohen_dz : ndarray, shape (K,)  [only if C=2]
            Cohen's dz for paired samples.

    CRITICAL ANALYSIS vs MATLAB
    ---------------------------
    The MATLAB ``within_subject_permutation_rm`` is not provided as a
    separate file, but its interface and expected output are fully
    determined from ``cdhsa_BC_condition_tests.m``. We implement:

    1. **F statistic**: One-way repeated-measures ANOVA. For C=2 this
       reduces to the squared paired t-statistic (F = t²).
    2. **Permutation scheme**: ``randperm(C)`` within each subject.
       MATLAB uses ``rng(seed,'twister')`` = MT19937. NumPy's
       ``default_rng(seed)`` also uses MT19937, but the actual
       permutation sequences differ due to different internal shuffle
       algorithms. This is statistically irrelevant.
    3. **Max correction**: p_maxF(k) = (1 + #{null_max >= F_obs(k)}) / (n_perm+1).
       This controls the family-wise error rate under the closed testing
       principle.

    Edge cases handled:
    - C < 2: raises ValueError (no condition effect to test).
    - S < 2: raises ValueError (no between-subject variability).
    - SS_error ≈ 0: returns F = 0 (no evidence for condition effect).
    """
    if opts is None:
        opts = {}
    n_perm = opts.get("n_perm", 5000)
    seed = opts.get("seed", 1234)

    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 2:
        Y = Y[:, :, np.newaxis]
    S, C, K = Y.shape

    if C < 2:
        raise ValueError(
            f"Need at least 2 conditions for permutation test, got C={C}"
        )
    if S < 2:
        raise ValueError(
            f"Need at least 2 subjects for permutation test, got S={S}"
        )

    # --- Observed F statistics ---
    F_obs = np.zeros(K, dtype=np.float64)
    for k in range(K):
        F_obs[k] = _rm_anova_f(Y[:, :, k])

    # --- Null distribution via within-subject permutation ---
    rng = np.random.default_rng(seed)
    null_F = np.zeros((n_perm, K), dtype=np.float64)
    null_maxF = np.zeros(n_perm, dtype=np.float64)

    for b in range(n_perm):
        # Shuffle condition labels within each subject independently
        Y_perm = np.empty_like(Y)
        for s in range(S):
            perm = rng.permutation(C)
            Y_perm[s, :, :] = Y[s, perm, :]

        for k in range(K):
            null_F[b, k] = _rm_anova_f(Y_perm[:, :, k])
        null_maxF[b] = np.max(null_F[b, :])

    # --- p-values ---
    p_unc = np.array([
        (1 + np.sum(null_F[:, k] >= F_obs[k])) / (n_perm + 1)
        for k in range(K)
    ])
    p_max = np.array([
        (1 + np.sum(null_maxF >= F_obs[k])) / (n_perm + 1)
        for k in range(K)
    ])

    result = {
        "F_obs": F_obs,
        "p_uncorrected": p_unc,
        "p_maxF": p_max,
        "null_F": null_F,
        "null_maxF": null_maxF,
        "n_perm": n_perm,
        "seed": seed,
    }

    # Effect sizes for C=2 (paired samples)
    if C == 2:
        diffs = Y[:, 1, :] - Y[:, 0, :]  # (S, K)
        mean_diff = np.mean(diffs, axis=0)  # (K,)
        std_diff = np.std(diffs, axis=0, ddof=1)
        cohen_dz = mean_diff / np.maximum(std_diff, 1e-15)
        result["mean_difference_C2_minus_C1"] = mean_diff
        result["cohen_dz"] = cohen_dz

    return result


def _rm_anova_f(Y_k: NDArray[np.floating]) -> float:
    """
    One-way repeated-measures F statistic.

    Parameters
    ----------
    Y_k : array, shape (S, C)
        Repeated-measures data: S subjects × C conditions.

    Returns
    -------
    F : float
        F statistic for the condition effect.

    Notes
    -----
    Decomposition:

    SS_condition = S * sum_c (Ybar_c - Ybar)^2,  df1 = C - 1
    SS_error    = sum_{s,c} (Y_sc - Ybar_s - Ybar_c + Ybar)^2,  df2 = (S-1)(C-1)
    F = (SS_condition / df1) / (SS_error / df2)

    For C=2 this equals t² where t is the paired t-statistic.
    Returns 0.0 when SS_error is near zero (perfect within-subject consistency
    leaves no degrees of freedom for error).
    """
    S, C = Y_k.shape

    if C < 2:
        raise ValueError(f"Need at least 2 conditions, got C={C}")

    grand_mean = np.mean(Y_k)
    subject_means = np.mean(Y_k, axis=1)  # (S,)
    condition_means = np.mean(Y_k, axis=0)  # (C,)

    SS_condition = S * np.sum((condition_means - grand_mean) ** 2)

    residuals = (
        Y_k - subject_means[:, None] - condition_means[None, :] + grand_mean
    )
    SS_error = np.sum(residuals ** 2)

    df1 = C - 1
    df2 = (S - 1) * (C - 1)

    if df1 == 0 or df2 == 0 or SS_error < 100 * np.finfo(np.float64).eps:
        return 0.0

    F = (SS_condition / df1) / (SS_error / df2)
    return float(F)

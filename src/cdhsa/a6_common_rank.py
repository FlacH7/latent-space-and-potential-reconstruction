"""
cdhsa/a6_common_rank.py — Step A6: Common rank selection with CV and null calibration
=====================================================================================

Determines the common rank r0 — the number of population Hankel directions
that are reliably shared across subjects — via two complementary criteria:

  (i)   Population commonality: observed λ_j exceeds a position-matched
        random-subspace null quantile.
  (ii)  Cross-validation: held-out CV_min(r) exceeds the same null's
        CV quantile (subjects left out, never conditions).

The final r0 is the largest CONSECUTIVE r from 1 satisfying BOTH criteria.
This avoids an arbitrary fixed CV threshold.

Dependencies: numpy, cdhsa.null_distributions
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Helper: population common basis from a cell of local bases
# ============================================================================

def common_basis_from_U(
    U_cell: list[list[NDArray[np.floating]]],
    max_r: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Population common basis from local Hankel bases via SVD of the
    concatenated matrix.

    Equivalent to eigendecomposing the pooled scatter
    M₀ = (1/N) Σ_{s,c} U_{sc} U_{sc}^T.

    Parameters
    ----------
    U_cell : list of list of arrays
        U_cell[s][c] shape (d, r_sc) — local orthonormal Hankel bases.
    max_r : int
        Maximum number of common directions to extract.

    Returns
    -------
    W : ndarray, shape (d, q)
        Population common basis (columns orthonormal), q = min(max_r, d, Σr_sc).
    lambda_ : ndarray, shape (q,)
        Commonality eigenvalues λ_j = σ_j²(B) where B = [U₁...U_N]/√N.

    Notes
    -----
    This is the same computation as in ``a_common_subspace.cdhsa_A1_A5`` step
    A3. It is factored out here because both ``crossvalidate_common_rank``
    and ``random_subspace_null`` need to re-estimate it on subsets of subjects.
    """
    S = len(U_cell)
    C = len(U_cell[0])
    N = S * C

    Ucat = np.concatenate(
        [U_cell[s][c] for s in range(S) for c in range(C)],
        axis=1,
    )
    B = Ucat / np.sqrt(N)

    d = B.shape[0]
    q = min(max_r, d, B.shape[1])

    U_B, s_B, _ = np.linalg.svd(B, full_matrices=False)
    W = U_B[:, :q]
    lambda_ = s_B[:q] ** 2
    lambda_ = np.clip(lambda_, 0.0, 1.0)

    return W, lambda_


# ============================================================================
# Subject-level cross-validation
# ============================================================================

def crossvalidate_common_rank(
    U_cell: list[list[NDArray[np.floating]]],
    r_values: NDArray[np.integer],
    opts: dict | None = None,
) -> dict:
    """
    Subject-level cross-validation of the common Hankel subspace.

    Folds are made over SUBJECTS, never over subject-condition rows.
    For each training fold the population common basis is estimated only
    from training subjects. It is then scored in held-out subjects using::

        score_i(r) = ||U_i^T W_r||_F² / r

    The output includes condition-specific held-out scores and::

        CV_min(r) = min_c mean_f score_{f,c}(r)

    which is deliberately strict: a direction is not called common if it
    generalizes only to one condition.

    Parameters
    ----------
    U_cell : list of list of arrays
        U_cell[s][c] shape (d, r_sc).
    r_values : array of int
        Candidate rank values to evaluate.
    opts : dict, optional
        n_folds (int, default 5): number of CV folds.
        seed (int, default 1): for reproducible fold assignment.

    Returns
    -------
    cv : dict with keys:
        r_values : ndarray (nr,)
        fold_id : ndarray (S,) — fold assignment per subject
        fold_score : ndarray (n_folds, nr)
        condition_score : ndarray (n_folds, C, nr)
        mean : ndarray (nr,) — overall mean CV score
        mean_condition : ndarray (C, nr) or (1, nr) if C=1
        min_condition : ndarray (nr,) — CV_min(r)

    CRITICAL ANALYSIS vs MATLAB
    ---------------------------
    Faithful translation of ``crossvalidate_common_rank.m``.

    Key design decisions preserved:
    1. **Subject-level folds** — never split conditions within subject.
    2. **CV_min = min over conditions** — strict criterion requiring
       generalization to ALL conditions.
    3. **Score normalization by r** — ||U_i^T W_r||_F² / r so that the
       score is bounded by 1 and comparable across different r values.
    4. **Separate RNG state**: MATLAB saves/restores ``rng`` state to
       avoid polluting the caller's stream. Python's
       ``np.random.default_rng(seed)`` creates an independent generator,
       achieving the same isolation automatically.
    """
    if opts is None:
        opts = {}
    n_folds = opts.get("n_folds", 5)
    seed = opts.get("seed", 1)

    S = len(U_cell)
    C = len(U_cell[0])

    if S < 2:
        raise ValueError(
            "At least two subjects are required for subject-level CV."
        )

    r_values = np.atleast_1d(np.asarray(r_values, dtype=int))
    nr = len(r_values)
    actual_folds = min(n_folds, S)

    # Reproducible fold assignment (isolated RNG)
    fold_rng = np.random.default_rng(seed)
    order = fold_rng.permutation(S)
    fold_id = np.zeros(S, dtype=int)
    for i in range(S):
        fold_id[order[i]] = i % actual_folds

    fold_score = np.full((actual_folds, nr), np.nan)
    condition_score = np.full((actual_folds, C, nr), np.nan)

    for f in range(actual_folds):
        train_idx = np.where(fold_id != f)[0]
        test_idx = np.where(fold_id == f)[0]

        # Training common basis from training subjects only
        Utrain = [[U_cell[s][c] for c in range(C)] for s in train_idx]
        maxr = int(np.max(r_values))
        Wtrain, _ = common_basis_from_U(Utrain, maxr)

        for ir in range(nr):
            r = int(r_values[ir])
            if r > Wtrain.shape[1]:
                continue
            Wr = Wtrain[:, :r]

            all_scores: list[float] = []
            for c in range(C):
                sc = np.zeros(len(test_idx))
                for ii, s in enumerate(test_idx):
                    Ui = U_cell[s][c]
                    G = Ui.T @ Wr  # (r_sc, r)
                    sc[ii] = np.linalg.norm(G, "fro") ** 2 / r
                condition_score[f, c, ir] = float(np.mean(sc))
                all_scores.extend(sc.tolist())
            fold_score[f, ir] = float(np.mean(all_scores))

    mean_cv = np.nanmean(fold_score, axis=0)

    if C == 1:
        mean_condition = np.nanmean(condition_score, axis=0).reshape(1, -1)
    else:
        mean_condition = np.nanmean(condition_score, axis=0)

    cv_min = np.nanmin(mean_condition, axis=0)

    return {
        "r_values": r_values,
        "fold_id": fold_id,
        "fold_score": fold_score,
        "condition_score": condition_score,
        "mean": mean_cv,
        "mean_condition": mean_condition,
        "min_condition": cv_min,
    }


# ============================================================================
# Main: Step A6
# ============================================================================

def cdhsa_A6_common_rank(
    R: dict,
    opts: dict | None = None,
) -> dict:
    """
    Step A6: Cross-validated and null-calibrated common rank.

    The recommended common rank r₀ is the largest consecutive r for which
    BOTH criteria hold simultaneously:

      (i)  observed λ_j exceeds its position-matched random-subspace
           (1-α) null quantile for every j ≤ r.
      (ii) held-out CV_min(r) exceeds the random-subspace null quantile.

    This avoids an arbitrary fixed CV threshold by calibrating against
    a null that preserves the rank structure of each local basis.

    Parameters
    ----------
    R : dict
        Output of ``cdhsa_A1_A5``. Must contain keys: 'U', 'lambda_', 'W',
        'rank', 'S', 'C'.
    opts : dict, optional
        max_common (int, default min(20, len(R['lambda_'])))
        n_folds (int, default 5)
        n_null (int, default 100)
        alpha (float, default 0.05)
        seed (int, default 1)

    Returns
    -------
    A6 : dict with keys:
        r_values : ndarray (qmax,) — candidate ranks tested
        cv : dict — output from crossvalidate_common_rank
        null : dict — output from random_subspace_null
        lambda_observed : ndarray (qmax,) — λ_j for j=1..qmax
        lambda_significant : ndarray (qmax,) bool — λ exceeds null quantile
        cv_significant : ndarray (qmax,) bool — CV_min exceeds null quantile
        pass : ndarray (qmax,) bool — BOTH criteria pass
        r0 : int — recommended common rank (0 if none pass)
        W0 : ndarray (d, r0) — fixed common basis columns 1..r0
        lambda0 : ndarray (r0,) — commonality of retained directions

    CRITICAL ANALYSIS vs MATLAB
    ---------------------------
    Faithful translation of ``cdhsa_A6_common_rank.m``.

    Design note on the null model:
    The MATLAB uses Haar-random bases that preserve each local rank r_sc.
    The tutor has noted this null is **too liberal** because it destroys
    the Hankel temporal correlation structure, making the null commonality
    spectrum systematically too low. The consequence is anti-conservative
    inference: more components declared significant than truly are.

    The recommended fix is the Hankel-preserving null (channel rotation
    I_L ⊗ Q_sc) implemented in ``null_distributions.hankel_preserving_null``.
    For development and validation, the random subspace null is sufficient.
    For final inference, switch to the Hankel-preserving null.
    """
    from src.cdhsa.null_distributions import random_subspace_null

    # Required fields from A1-A5 (validate BEFORE using them)
    for key in ("U", "lambda_", "W"):
        if key not in R:
            raise ValueError(f"R must contain '{key}' from cdhsa_A1_A5.")

    if opts is None:
        opts = {}
    max_common = opts.get("max_common", min(20, len(R["lambda_"])))
    n_folds = opts.get("n_folds", 5)
    n_null = opts.get("n_null", 100)
    alpha = opts.get("alpha", 0.05)
    seed = opts.get("seed", 1)

    qmax = min(max_common, len(R["lambda_"]), R["W"].shape[1])
    r_values = np.arange(1, qmax + 1, dtype=int)

    # (1) Cross-validation on OBSERVED data
    cv_opts = {"n_folds": n_folds, "seed": seed}
    cv = crossvalidate_common_rank(R["U"], r_values, cv_opts)

    # (2) Null distribution (random subspace null)
    null_opts = {
        "n_null": n_null,
        "alpha": alpha,
        "n_folds": n_folds,
        "seed": seed + 10,
    }
    null = random_subspace_null(R["U"], r_values, null_opts)

    # (3) Compare observed vs null
    obs_lambda = R["lambda_"][:qmax]
    lambda_sig = obs_lambda > null["lambda_q"][:qmax]
    cv_sig = cv["min_condition"] > null["cv_min_q"]
    pass_both = lambda_sig & cv_sig

    # (4) Consecutive-prefix criterion
    r0 = 0
    for r in range(qmax):
        if np.all(pass_both[: r + 1]):
            r0 = r + 1
        else:
            break

    # Assemble output
    result = {
        "r_values": r_values,
        "cv": cv,
        "null": null,
        "lambda_observed": obs_lambda,
        "lambda_significant": lambda_sig,
        "cv_significant": cv_sig,
        "pass": pass_both,
        "r0": r0,
        "lambda0": R["lambda_"][:r0].copy() if r0 > 0 else np.array([]),
    }

    if r0 > 0:
        result["W0"] = R["W"][:, :r0].copy()
    else:
        result["W0"] = np.zeros((R["W"].shape[0], 0))

    return result

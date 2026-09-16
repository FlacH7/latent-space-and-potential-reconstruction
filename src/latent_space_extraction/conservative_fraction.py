"""
Conservative Fraction Subspace Selection
========================================

Stage II of the pipeline: selects an N-dimensional subspace of ICA
components that maximises variance retention.

Three metrics are implemented:
    * ``'variance_sum'``   — sum of individual variances (Equation 7)
    * ``'first_pc_var'``   — variance of the first principal axis (Equation 8)
    * ``'total_variance'`` — total projected variance via pseudoinverse (Equation 9)

For each metric, an exhaustive combinatorial search is performed over
all ``C(D, N)`` combinations. This is feasible for ``N = 2`` or ``3``
with ``D ~ 60`` (1,770 and 34,220 combinations, respectively). For
larger ``N``, a greedy or approximate strategy should be used.

References
----------
    Equations (5)–(10) of the pipeline report.
"""

from __future__ import annotations

import time
from itertools import combinations
from multiprocessing import Pool
from typing import Callable

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_conservative_fraction(
    combination: tuple[int, ...],
    data: np.ndarray,
    *,
    metric: str = "variance_sum",
    total_var: float | None = None,
) -> float:
    """
    Compute the conservative fraction :math:`f_c(S)` for a candidate subset.

    Parameters
    ----------
    combination : tuple of int
        Row indices (component indices) that define the candidate subspace
        ``S = {i_1, ..., i_N}``.
    data : np.ndarray, shape (D, T)
        Clean component matrix ``Y`` (zero-mean rows). ``D`` is the total
        number of available components and ``T`` the number of time samples.
    metric : str, default ``'variance_sum'``
        Variance measure to use. One of:

        * ``'variance_sum'`` — sum of marginal variances of the selected
          components (Equation 7). Fast but ignores correlations.
        * ``'first_pc_var'`` — largest eigenvalue of the subspace covariance
          (Equation 8). Invariant to intra-subspace rotations.
        * ``'total_variance'`` — total variance retained when the full data
          is orthogonally projected onto the subspace (Equation 9). Most
          rigorous; reduces to PCA explained variance for top-N PCs.
    total_var : float or None, optional
        Pre-computed total variance ``Tr(C)`` of the full data. If None, it
        is computed on the fly.

    Returns
    -------
    fc : float
        Conservative fraction in ``[0, 1]``.

    Raises
    ------
    ValueError
        If *metric* is not one of the supported options.
    """
    D, T = data.shape
    sub_data = data[list(combination), :]  # (N, T)
    N = sub_data.shape[0]

    # Total variance (denominator) — cache if provided
    if total_var is None:
        total_var = float(np.trace(np.cov(data)))
    if total_var == 0:
        return 0.0

    if metric == "variance_sum":
        # Eq. (7): V_sum(S) = sum_{k in S} Var(y_k)
        var_sub = float(np.var(sub_data, axis=1, ddof=1).sum())
        return var_sub / total_var

    elif metric == "first_pc_var":
        # Eq. (8): V_pc1(S) = lambda_max(C_S)
        cov_sub = np.cov(sub_data)  # (N, N)
        eigvals = np.linalg.eigvalsh(cov_sub)
        first_pc_var = float(np.max(eigvals))
        return first_pc_var / total_var

    elif metric == "total_variance":
        # Eq. (9): V_proj(S) = ||P_S Y||_F^2 / T
        # Implemented via trace(inv(G) @ M) for numerical stability
        G = sub_data @ sub_data.T  # Gram matrix (N, N)
        cross = sub_data @ data.T  # (N, D)
        M = (cross @ cross.T) / (T - 1)  # (N, N)

        try:
            inv_G = np.linalg.pinv(G)
        except np.linalg.LinAlgError:
            inv_G = np.linalg.pinv(G)

        var_ret = float(np.trace(inv_G @ M))
        return var_ret / total_var

    else:
        raise ValueError(
            f"Metric '{metric}' not recognised. "
            f"Choose from: 'variance_sum', 'first_pc_var', 'total_variance'."
        )


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _evaluate_fc_worker(args: tuple) -> tuple[tuple[int, ...], float]:
    """Unpacker for multiprocessing.Pool."""
    combination, data, metric, total_var = args
    fc = compute_conservative_fraction(
        combination, data, metric=metric, total_var=total_var
    )
    return combination, fc


# ---------------------------------------------------------------------------
# Exhaustive search
# ---------------------------------------------------------------------------

def find_best_subspace_fc(
    data: np.ndarray,
    N: int,
    *,
    metric: str = "variance_sum",
    n_workers: int | None = None,
    show_progress: bool = True,
) -> tuple[tuple[int, ...] | None, float]:
    """
    Exhaustively search for the N-component subspace with the largest
    conservative fraction.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix (zero-mean rows).
    N : int
        Target subspace dimensionality (2 or 3 recommended for exhaustive
        search; 4 and above become prohibitively expensive).
    metric : str, default ``'variance_sum'``
        Variance measure — see :func:`compute_conservative_fraction`.
    n_workers : int or None, optional
        Number of parallel processes. ``None`` uses all available cores.
        Set to ``1`` to disable parallelism.
    show_progress : bool, default True
        Show a tqdm progress bar.

    Returns
    -------
    best_combination : tuple[int, ...] or None
        Indices of the optimal subspace, or None if *data* is empty.
    best_fc : float
        Maximum conservative fraction achieved.

    Raises
    ------
    ValueError
        If ``N > D``.
    """
    D, T = data.shape
    if N > D:
        raise ValueError(f"N={N} cannot exceed D={D}")

    # Pre-compute total variance to avoid redundant work in workers
    total_var = float(np.trace(np.cov(data)))

    all_combs = list(combinations(range(D), N))
    n_combs = len(all_combs)

    if n_combs == 0:
        return None, 0.0

    print(
        f"[ConservativeFraction] Searching best subspace of {N} components "
        f"out of {D} total."
    )
    print(f"[ConservativeFraction] Metric: {metric}")
    print(f"[ConservativeFraction] Evaluating {n_combs:,} combinations...")

    args_list = [
        (comb, data, metric, total_var) for comb in all_combs
    ]

    start = time.time()
    results: list[tuple[tuple[int, ...], float]] = []

    if n_workers == 1:
        # Serial execution
        it = (
            tqdm(args_list, desc="ConservativeFraction")
            if show_progress
            else args_list
        )
        for a in it:
            results.append(_evaluate_fc_worker(a))
    else:
        # Parallel execution
        with Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(_evaluate_fc_worker, args_list)
            if show_progress:
                it = tqdm(it, total=n_combs, desc="ConservativeFraction")
            results = list(it)

    elapsed = time.time() - start
    print(f"[ConservativeFraction] Done in {elapsed:.2f} s.")

    # Find maximum
    best_comb: tuple[int, ...] | None = None
    best_fc = -np.inf
    for comb, fc in results:
        if fc > best_fc:
            best_fc = fc
            best_comb = comb

    return best_comb, float(best_fc)


# ---------------------------------------------------------------------------
# Greedy forward selection (fallback for larger N)
# ---------------------------------------------------------------------------

def greedy_forward_selection_fc(
    data: np.ndarray,
    N: int,
    *,
    metric: str = "variance_sum",
) -> tuple[list[int], float]:
    """
    Greedy forward selection for the conservative-fraction criterion.

    Starting from an empty set, iteratively add the component that gives
    the largest marginal increase in the conservative fraction until *N*
components are selected. Runs in ``O(N * D)`` metric evaluations,
    making it suitable for ``N >= 4`` where exhaustive search is
    infeasible.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix (zero-mean rows).
    N : int
        Target subspace dimensionality.
    metric : str, default ``'variance_sum'``
        Variance measure — see :func:`compute_conservative_fraction`.

    Returns
    -------
    selected : list[int]
        Indices of the greedily selected components.
    final_fc : float
        Conservative fraction of the final subspace.
    """
    D, T = data.shape
    if N > D:
        raise ValueError(f"N={N} cannot exceed D={D}")

    total_var = float(np.trace(np.cov(data)))
    selected: list[int] = []
    remaining = set(range(D))

    for _ in range(N):
        best_idx = -1
        best_fc = -np.inf
        for idx in remaining:
            trial = tuple(selected + [idx])
            fc = compute_conservative_fraction(
                trial, data, metric=metric, total_var=total_var
            )
            if fc > best_fc:
                best_fc = fc
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    final_fc = compute_conservative_fraction(
        tuple(selected), data, metric=metric, total_var=total_var
    )
    return selected, float(final_fc)


# ---------------------------------------------------------------------------
# Batch evaluation helper (score many combinations at once)
# ---------------------------------------------------------------------------

def score_many_combinations_fc(
    data: np.ndarray,
    combinations_list: list[tuple[int, ...]],
    *,
    metric: str = "variance_sum",
    n_workers: int | None = None,
    show_progress: bool = True,
) -> list[tuple[tuple[int, ...], float]]:
    """
    Evaluate the conservative fraction for an arbitrary list of combinations.

    Useful for Pareto-frontier construction or custom search strategies.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix.
    combinations_list : list of tuple
        Candidate subspace index sets.
    metric : str, default ``'variance_sum'``
        Variance measure.
    n_workers : int or None, optional
        Number of parallel processes.
    show_progress : bool, default True

    Returns
    -------
    scored : list of (tuple, float)
        Each entry is ``(combination, fc_value)``.
    """
    total_var = float(np.trace(np.cov(data)))
    args_list = [
        (comb, data, metric, total_var) for comb in combinations_list
    ]

    results: list[tuple[tuple[int, ...], float]] = []

    if n_workers == 1 or len(args_list) < 100:
        it = (
            tqdm(args_list, desc="ScoreFC")
            if show_progress
            else args_list
        )
        for a in it:
            results.append(_evaluate_fc_worker(a))
    else:
        with Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(_evaluate_fc_worker, args_list)
            if show_progress:
                it = tqdm(it, total=len(args_list), desc="ScoreFC")
            results = list(it)

    return results

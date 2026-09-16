"""
Markov-Time Subspace Selection
==============================

Stage III of the pipeline: selects an N-dimensional subspace of ICA
components that minimises the Markov relaxation time.

The procedure is:
    1. Discretise each component into ``n_bins`` quantile bins.
    2. Enumerate joint states via mixed-radix indexing.
    3. Estimate the Markov transition matrix by counting transitions.
    4. Compute the spectral gap and relaxation time ``tau``.
    5. Search combinatorially for the subset with the smallest ``tau``.

A small relaxation time means the discretised dynamics explore the state
space rapidly, indicating rich intrinsic stochastic dynamics.

References
----------
    Equations (11)–(15) of the pipeline report.
"""

from __future__ import annotations

import sys
import time
from itertools import combinations
from multiprocessing import Pool
from typing import Sequence

import numpy as np
from scipy.linalg import eig
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Discretisation
# ---------------------------------------------------------------------------

def discretize_series(
    data: np.ndarray,
    n_bins: int = 5,
) -> np.ndarray:
    """
    Discretise each row of *data* into *n_bins* quantile-based bins.

    For component ``k``, the bin edges ``{q_0, q_1, ..., q_{n_b}}`` are the
    empirical quantiles (Equation 11). Every bin contains (asymptotically)
    the same number of samples, which avoids sparsity in the distribution
    tails.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix (each row is a univariate time series).
    n_bins : int, default 5
        Number of quantile bins per component.

    Returns
    -------
    bins_idx : np.ndarray, shape (D, T), dtype int
        Bin indices in ``{0, ..., n_bins - 1}`` for each sample.
    """
    D, T = data.shape
    bins_idx = np.empty((D, T), dtype=int)

    for i in range(D):
        # Inner quantile edges (exclude 0th and 100th percentiles)
        quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
        edges = np.percentile(data[i, :], quantiles)
        # Extend to [-inf, +inf] to cover all values
        edges = np.concatenate([[-np.inf], edges, [np.inf]])
        bins_idx[i, :] = np.digitize(data[i, :], edges) - 1  # 0-indexed

    return bins_idx


# ---------------------------------------------------------------------------
# Markov transition matrix and relaxation time
# ---------------------------------------------------------------------------

def compute_markov_time(
    state_sequence: np.ndarray,
    n_states: int,
    *,
    ergodicity_tol: float = 1e-6,
    periodicity_tol: float = 1e-6,
) -> float:
    """
    Estimate the relaxation time (Markov time) of a discrete-state sequence.

    The transition matrix ``P`` is built by counting observed transitions
    (Equation 12). Its eigenvalues are ordered by modulus (Equation 13) and
    the relaxation time follows Equation 14::

        tau = -1 / ln(|lambda_2|)

    Parameters
    ----------
    state_sequence : np.ndarray, 1-D
        Integer array of states in ``{0, ..., n_states - 1}``.
    n_states : int
        Total number of possible states.
    ergodicity_tol : float, default 1e-6
        Numerical tolerance for the dominant eigenvalue.
    periodicity_tol : float, default 1e-6
        If ``|lambda_2|`` is within this distance of 1.0, the chain is
        considered periodic / non-ergodic and ``inf`` is returned.

    Returns
    -------
    tau : float
        Relaxation time in discrete steps, or ``np.inf`` if the chain is
        non-ergodic or periodic.
    """
    T = len(state_sequence)
    P = np.zeros((n_states, n_states))

    # Count transitions (including self-transitions)
    for t in range(T - 1):
        i = int(state_sequence[t])
        j = int(state_sequence[t + 1])
        P[i, j] += 1

    row_sums = P.sum(axis=1, keepdims=True)
    zero_rows = (row_sums.ravel() == 0)

    if zero_rows.any():
        # Unvisited states → chain is not irreducible
        return np.inf

    P = P / row_sums

    # Spectral decomposition
    eigenvalues = eig(P, left=False, right=False)
    eig_mod = np.abs(eigenvalues)
    eig_mod_sorted = np.sort(eig_mod)[::-1]

    # Force lambda_1 = 1 (numerical safety)
    if not np.isclose(eig_mod_sorted[0], 1.0, atol=ergodicity_tol):
        eig_mod_sorted[0] = 1.0

    lambda2 = eig_mod_sorted[1]

    if lambda2 >= 1.0 - periodicity_tol:
        return np.inf

    tau = -1.0 / np.log(lambda2)
    return float(tau)


# ---------------------------------------------------------------------------
# Single-combination evaluator
# ---------------------------------------------------------------------------

def evaluate_markov_combination(
    combination: tuple[int, ...],
    bins_idx: np.ndarray,
    n_bins: int,
) -> tuple[tuple[int, ...], float]:
    """
    Compute the Markov relaxation time for a single candidate subspace.

    Parameters
    ----------
    combination : tuple of int
        Row indices of the selected components.
    bins_idx : np.ndarray, shape (D, T)
        Pre-computed bin indices from :func:`discretize_series`.
    n_bins : int
        Number of bins per component.

    Returns
    -------
    combination : tuple[int, ...]
        The same index set (for convenience in parallel maps).
    tau : float
        Relaxation time (``np.inf`` if non-ergodic).
    """
    sub_bins = bins_idx[list(combination), :]  # (N, T)
    N = sub_bins.shape[0]

    # Mixed-radix enumeration: each joint state → single integer
    powers = n_bins ** np.arange(N)
    state_sequence = np.dot(powers, sub_bins)  # (T,)
    n_states = n_bins ** N

    tau = compute_markov_time(state_sequence, n_states)
    return combination, tau


def _evaluate_markov_worker(args: tuple) -> tuple[tuple[int, ...], float]:
    """Unpacker for multiprocessing.Pool."""
    combination, bins_idx, n_bins = args
    return evaluate_markov_combination(combination, bins_idx, n_bins)


# ---------------------------------------------------------------------------
# Exhaustive combinatorial search
# ---------------------------------------------------------------------------

def find_best_subspace_markov(
    data: np.ndarray,
    N: int,
    *,
    n_bins: int = 5,
    n_workers: int | None = None,
    show_progress: bool = True,
    maximize: bool = False,
    return_all_taus: bool = False,
) -> tuple:
    """
    Exhaustively search for the N-component subspace with the *smallest*
    Markov relaxation time.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix (zero-mean rows).
    N : int
        Target subspace dimensionality (2 or 3 recommended).
    n_bins : int, default 5
        Number of quantile bins per component.
    n_workers : int or None, optional
        Number of parallel processes. ``None`` uses all cores. Set to ``1``
        for serial execution.
    show_progress : bool, default True
        Show a tqdm progress bar.
    return_all_taus : bool, default False
        If True, also return the list of ``(combination, tau)`` for every
        evaluated combination.  Used by the plotters module to generate
        Stage-3 diagnostic plots (heatmap, ranked, distribution).

    Returns
    -------
    best_combination : tuple[int, ...] or None
        Indices of the optimal subspace, or None if no valid subspace was
        found (all returned ``inf``).
    best_tau : float
        Minimum relaxation time achieved (``np.inf`` if no valid subspace).
    all_taus : list of (tuple, float), optional
        Only returned if ``return_all_taus=True``.  Each entry is
        ``(combination, tau)`` for every combination evaluated.

    Raises
    ------
    ValueError
        If ``N > D``.
    """
    D, T = data.shape
    if N > D:
        raise ValueError(f"N={N} cannot exceed D={D}")

    print(f"[MarkovTime] Discretising {D} components into {n_bins} bins...")
    sys.stdout.flush()
    bins_idx = discretize_series(data, n_bins)
    print("[MarkovTime] Discretisation done.")
    sys.stdout.flush()

    all_combs = list(combinations(range(D), N))
    n_combs = len(all_combs)

    print(
        f"[MarkovTime] Evaluating {n_combs:,} combinations of {N} components..."
    )
    sys.stdout.flush()

    args_list = [(comb, bins_idx, n_bins) for comb in all_combs]

    start = time.time()
    results: list[tuple[tuple[int, ...], float]] = []

    if n_workers == 1 or n_combs < 500:
        it = (
            tqdm(args_list, desc="MarkovTime")
            if show_progress
            else args_list
        )
        for a in it:
            results.append(_evaluate_markov_worker(a))
        sys.stdout.flush()
    else:
        print(f"[MarkovTime] Using {n_workers} parallel workers...")
        sys.stdout.flush()
        with Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(_evaluate_markov_worker, args_list)
            if show_progress:
                it = tqdm(it, total=n_combs, desc="MarkovTime")
            results = list(it)
            # FIX: tqdm 100% no significa que list() terminó — el pool
            # puede estar haciendo join/cleanup de worker processes.
            print("[MarkovTime] All workers finished. Collecting results...")
            sys.stdout.flush()

    elapsed = time.time() - start
    print(f"[MarkovTime] Done in {elapsed:.2f} s.")
    sys.stdout.flush()

    best_comb: tuple[int, ...] | None = None
    best_tau = -np.inf if maximize else np.inf

    for comb, tau in results:
        if maximize:
            if np.isfinite(tau) and tau > best_tau:
                best_tau = tau
                best_comb = comb
        else:
            if tau < best_tau:
                best_tau = tau
                best_comb = comb
    if maximize and best_comb is None:
        best_tau = np.inf  # No valid subspace found

    if return_all_taus:
        return best_comb, float(best_tau), results
    return best_comb, float(best_tau)


# ---------------------------------------------------------------------------
# Greedy forward selection (fallback for larger N)
# ---------------------------------------------------------------------------

def greedy_forward_selection_markov(
    data: np.ndarray,
    N: int,
    *,
    n_bins: int = 5,
    maximize: bool = False,
) -> tuple[list[int], float]:
    """
    Greedy forward selection for the Markov-time criterion.

    Iteratively adds the component that yields the smallest relaxation time
    until *N* components are selected.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix.
    N : int
        Target subspace dimensionality.
    n_bins : int, default 5
        Number of quantile bins per component.
    maximize : bool, default False
    Returns
    -------
    selected : list[int]
        Greedily selected component indices.
    final_tau : float
        Relaxation time of the final subspace.
    """
    D, T = data.shape
    if N > D:
        raise ValueError(f"N={N} cannot exceed D={D}")

    bins_idx = discretize_series(data, n_bins)
    selected: list[int] = []
    remaining = set(range(D))

    for _ in range(N):
        best_idx = -1
        best_tau = -np.inf if maximize else np.inf
        for idx in remaining:
            trial = tuple(selected + [idx])
            _, tau = evaluate_markov_combination(trial, bins_idx, n_bins)
            if maximize:
                if np.isfinite(tau) and tau > best_tau:
                    best_tau = tau
                    best_idx = idx
            else:
                if tau < best_tau:
                    best_tau = tau
                    best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    _, final_tau = evaluate_markov_combination(
        tuple(selected), bins_idx, n_bins
    )
    return selected, float(final_tau)


# ---------------------------------------------------------------------------
# Batch evaluation helper (score many combinations at once)
# ---------------------------------------------------------------------------

def score_many_combinations_markov(
    data: np.ndarray,
    combinations_list: list[tuple[int, ...]],
    *,
    n_bins: int = 5,
    n_workers: int | None = None,
    show_progress: bool = True,
) -> list[tuple[tuple[int, ...], float]]:
    """
    Evaluate the Markov relaxation time for an arbitrary list of combinations.

    Useful for Pareto-frontier construction or custom search strategies.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix.
    combinations_list : list of tuple
        Candidate subspace index sets.
    n_bins : int, default 5
        Number of quantile bins per component.
    n_workers : int or None, optional
        Number of parallel processes.
    show_progress : bool, default True

    Returns
    -------
    scored : list of (tuple, float)
        Each entry is ``(combination, tau_value)``.
    """
    bins_idx = discretize_series(data, n_bins)
    args_list = [
        (comb, bins_idx, n_bins) for comb in combinations_list
    ]

    results: list[tuple[tuple[int, ...], float]] = []

    if n_workers == 1 or len(args_list) < 500:
        it = (
            tqdm(args_list, desc="ScoreMarkov")
            if show_progress
            else args_list
        )
        for a in it:
            results.append(_evaluate_markov_worker(a))
    else:
        with Pool(processes=n_workers) as pool:
            it = pool.imap_unordered(_evaluate_markov_worker, args_list)
            if show_progress:
                it = tqdm(it, total=len(args_list), desc="ScoreMarkov")
            results = list(it)

    return results

"""
Unified Subspace Selection Strategies
=====================================

Orchestration layer that combines the conservative-fraction (Stage II) and
Markov-time (Stage III) criteria into cohesive selection strategies.

The two criteria may disagree: a high-variance component can evolve slowly
(large ``tau``), while a component with fast dynamics may carry little
amplitude. This module provides four strategies to reconcile them:

    1. **Weighted scalarisation** — convex combination of normalised scores.
    2. **Sequential filtering** — filter top-K under one criterion, re-rank
       under the other.
    3. **Pareto frontier** — non-dominated subspaces in the
       ``(f_c, 1/tau)`` plane.
    4. **Independent execution** — simply run both criteria and return both
       results (useful when the downstream analysis can handle two
       candidate subspaces).

All strategies operate on the clean component matrix ``Y`` and produce
one or more candidate subspaces of dimension ``N``.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Literal, Sequence

import numpy as np

from src.latent_space_extraction.conservative_fraction import (
    compute_conservative_fraction,
    find_best_subspace_fc,
    greedy_forward_selection_fc,
    score_many_combinations_fc,
)
from src.latent_space_extraction.markov_subspace import (
    compute_markov_time,
    discretize_series,
    evaluate_markov_combination,
    find_best_subspace_markov,
    greedy_forward_selection_markov,
    score_many_combinations_markov,
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _min_max_scale(values: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1] using min-max normalisation."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-12:
        return np.ones_like(values)
    return (values - vmin) / (vmax - vmin)


def _compute_fc_and_inv_tau(
    data: np.ndarray,
    combinations_list: list[tuple[int, ...]],
    *,
    metric: str = "variance_sum",
    n_bins: int = 5,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate both objectives for a list of combinations.

    Returns
    -------
    fc_vals : np.ndarray
        Conservative fractions.
    inv_tau_vals : np.ndarray
        Inverse relaxation times (``0`` where ``tau == inf``).
    """
    # Conservative fraction
    fc_results = score_many_combinations_fc(
        data, combinations_list, metric=metric, n_workers=n_workers,
        show_progress=False,
    )
    fc_vals = np.array([fc for _, fc in fc_results])

    # Markov time
    tau_results = score_many_combinations_markov(
        data, combinations_list, n_bins=n_bins, n_workers=n_workers,
        show_progress=False,
    )
    tau_vals = np.array([tau for _, tau in tau_results])
    inv_tau_vals = np.where(np.isinf(tau_vals), 0.0, 1.0 / tau_vals)

    return fc_vals, inv_tau_vals


# ---------------------------------------------------------------------------
# Strategy 1: Weighted scalarisation
# ---------------------------------------------------------------------------

def weighted_score_selection(
    data: np.ndarray,
    N: int,
    *,
    alphas: Sequence[float] | None = None,
    metric: str = "variance_sum",
    n_bins: int = 5,
    n_workers: int | None = None,
    search_strategy: Literal["exhaustive", "greedy"] = "exhaustive",
) -> dict:
    """
    Scalarised weighted selection.

    Normalises both ``f_c`` and ``1/tau`` to ``[0, 1]`` and forms a convex
    combination::

        S_alpha = alpha * f_c_tilde + (1 - alpha) * (1/tau)_tilde

    The best subspace for each *alpha* is returned.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix.
    N : int
        Target subspace dimensionality.
    alphas : sequence of float, optional
        Grid of trade-off parameters in ``[0, 1]``. Defaults to
        ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
    metric : str, default ``'variance_sum'``
        Variance measure for the conservative fraction.
    n_bins : int, default 5
        Quantile bins for Markov discretisation.
    n_workers : int or None, optional
        Parallel workers.
    search_strategy : ``'exhaustive'`` | ``'greedy'``, default ``'exhaustive'``
        Search method. Use ``'greedy'`` for ``N >= 4``.

    Returns
    -------
    result : dict
        Keys:

        * ``'alphas'``            — array of alpha values tested.
        * ``'best_combinations'`` — list of optimal subspaces (one per alpha).
        * ``'best_scores'``       — list of scalarised scores.
        * ``'fc_values'``         — conservative fractions of selected subspaces.
        * ``'tau_values'``        — relaxation times of selected subspaces.
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Enumerate (or greedily approximate) candidate combinations
    if search_strategy == "exhaustive":
        D = data.shape[0]
        all_combs = list(combinations(range(D), N))
    else:
        # For greedy, we need a different approach: generate candidates
        # by running greedy for several random starts or heuristics.
        # Here we simply collect a moderate-sized candidate set.
        all_combs = _generate_candidate_combinations(data, N, n_candidates=500)

    fc_vals, inv_tau_vals = _compute_fc_and_inv_tau(
        data, all_combs, metric=metric, n_bins=n_bins, n_workers=n_workers,
    )

    fc_tilde = _min_max_scale(fc_vals)
    inv_tau_tilde = _min_max_scale(inv_tau_vals)

    best_combs: list[tuple[int, ...]] = []
    best_scores: list[float] = []
    best_fc: list[float] = []
    best_tau: list[float] = []

    for alpha in alphas:
        scores = alpha * fc_tilde + (1.0 - alpha) * inv_tau_tilde
        idx_best = int(np.argmax(scores))
        best_combs.append(all_combs[idx_best])
        best_scores.append(float(scores[idx_best]))
        best_fc.append(float(fc_vals[idx_best]))
        tau_val = (
            np.inf if inv_tau_vals[idx_best] == 0
            else 1.0 / inv_tau_vals[idx_best]
        )
        best_tau.append(float(tau_val))

    return {
        "alphas": np.array(alphas),
        "best_combinations": best_combs,
        "best_scores": np.array(best_scores),
        "fc_values": np.array(best_fc),
        "tau_values": np.array(best_tau),
    }


# ---------------------------------------------------------------------------
# Strategy 2: Sequential filtering
# ---------------------------------------------------------------------------

def sequential_filtering_selection(
    data: np.ndarray,
    N: int,
    K: int,
    *,
    primary: Literal["fc", "markov"] = "fc",
    metric: str = "variance_sum",
    n_bins: int = 5,
    n_workers: int | None = None,
) -> dict:
    """
    Sequential filtering: retain the top-K candidates under one criterion
    and re-rank them under the other.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix.
    N : int
        Target subspace dimensionality.
    K : int
        Number of candidates to retain after the primary filter.
    primary : ``'fc'`` | ``'markov'``, default ``'fc'``
        Which criterion to use for the first filtering stage.
        ``'fc'``   → filter by conservative fraction, re-rank by Markov time.
        ``'markov'`` → filter by Markov time, re-rank by conservative fraction.
    metric : str, default ``'variance_sum'``
        Variance measure for the conservative fraction.
    n_bins : int, default 5
        Quantile bins for Markov discretisation.
    n_workers : int or None, optional
        Parallel workers.

    Returns
    -------
    result : dict
        Keys:

        * ``'primary_criterion'``   — ``'fc'`` or ``'markov'``.
        * ``'primary_topK'``        — top-K under primary criterion.
        * ``'reordered'``           — same candidates reordered under
          secondary criterion.
        * ``'best_combination'``    — best subspace after re-ranking.
        * ``'best_fc'``             — its conservative fraction.
        * ``'best_tau'``            — its relaxation time.
    """
    D, T = data.shape
    all_combs = list(combinations(range(D), N))

    fc_vals, inv_tau_vals = _compute_fc_and_inv_tau(
        data, all_combs, metric=metric, n_bins=n_bins, n_workers=n_workers,
    )
    tau_vals = np.where(inv_tau_vals == 0, np.inf, 1.0 / inv_tau_vals)

    if primary == "fc":
        # Stage 1: keep top-K by fc
        top_k_idx = np.argsort(fc_vals)[-K:][::-1]
        primary_top_k = [all_combs[i] for i in top_k_idx]

        # Stage 2: re-rank by tau (ascending)
        tau_of_top = np.array([tau_vals[i] for i in top_k_idx])
        reorder_idx = np.argsort(tau_of_top)
        reordered = [primary_top_k[i] for i in reorder_idx]

    elif primary == "markov":
        # Stage 1: keep top-K by 1/tau (i.e. smallest tau)
        inv_tau_finite = np.where(inv_tau_vals == 0, -1.0, inv_tau_vals)
        top_k_idx = np.argsort(inv_tau_finite)[-K:][::-1]
        primary_top_k = [all_combs[i] for i in top_k_idx]

        # Stage 2: re-rank by fc (descending)
        fc_of_top = np.array([fc_vals[i] for i in top_k_idx])
        reorder_idx = np.argsort(fc_of_top)[::-1]
        reordered = [primary_top_k[i] for i in reorder_idx]

    else:
        raise ValueError(f"primary must be 'fc' or 'markov', got {primary!r}")

    best_comb = reordered[0]
    best_fc = float(compute_conservative_fraction(best_comb, data, metric=metric))
    _, best_tau = evaluate_markov_combination(
        best_comb, discretize_series(data, n_bins), n_bins
    )

    return {
        "primary_criterion": primary,
        "primary_topK": primary_top_k,
        "reordered": reordered,
        "best_combination": best_comb,
        "best_fc": best_fc,
        "best_tau": float(best_tau),
    }


# ---------------------------------------------------------------------------
# Strategy 3: Pareto frontier
# ---------------------------------------------------------------------------

def pareto_frontier_selection(
    data: np.ndarray,
    N: int,
    *,
    metric: str = "variance_sum",
    n_bins: int = 5,
    n_workers: int | None = None,
) -> dict:
    """
    Compute the Pareto frontier in the ``(f_c, 1/tau)`` plane.

    A subspace is *non-dominated* if no other subspace has both a higher
    conservative fraction *and* a smaller relaxation time.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix.
    N : int
        Target subspace dimensionality.
    metric : str, default ``'variance_sum'``
        Variance measure for the conservative fraction.
    n_bins : int, default 5
        Quantile bins for Markov discretisation.
    n_workers : int or None, optional
        Parallel workers.

    Returns
    -------
    result : dict
        Keys:

        * ``'frontier_combinations'`` — list of non-dominated subspaces.
        * ``'frontier_fc'``           — their conservative fractions.
        * ``'frontier_tau'``          — their relaxation times.
        * ``'all_combinations'``      — all evaluated combinations.
        * ``'all_fc'``                — all conservative fractions.
        * ``'all_tau'``               — all relaxation times.
    """
    D, T = data.shape
    all_combs = list(combinations(range(D), N))

    fc_vals, inv_tau_vals = _compute_fc_and_inv_tau(
        data, all_combs, metric=metric, n_bins=n_bins, n_workers=n_workers,
    )
    tau_vals = np.where(inv_tau_vals == 0, np.inf, 1.0 / inv_tau_vals)

    # Identify non-dominated points
    # We want to MAXIMISE fc and MINIMISE tau  →  MAXIMISE fc and MAXIMISE 1/tau
    is_non_dominated = np.ones(len(all_combs), dtype=bool)

    for i in range(len(all_combs)):
        if not is_non_dominated[i]:
            continue
        for j in range(len(all_combs)):
            if i == j:
                continue
            # j dominates i if: fc_j >= fc_i AND tau_j <= tau_i, with at least one strict
            if (
                fc_vals[j] >= fc_vals[i]
                and tau_vals[j] <= tau_vals[i]
                and (fc_vals[j] > fc_vals[i] or tau_vals[j] < tau_vals[i])
            ):
                is_non_dominated[i] = False
                break

    frontier_idx = np.where(is_non_dominated)[0]
    frontier_idx = frontier_idx[np.argsort(fc_vals[frontier_idx])]

    return {
        "frontier_size": len(frontier_idx),
        "frontier_combinations": [all_combs[i] for i in frontier_idx],
        "frontier_fc": fc_vals[frontier_idx],
        "frontier_tau": tau_vals[frontier_idx],
        "all_combinations": all_combs,
        "all_fc": fc_vals,
        "all_tau": tau_vals,
    }


# ---------------------------------------------------------------------------
# Strategy 4: Independent execution (run both, return both)
# ---------------------------------------------------------------------------

def independent_selection(
    data: np.ndarray,
    N: int,
    *,
    metric: str = "variance_sum",
    n_bins: int = 5,
    n_workers: int | None = None,
    fc_search: Literal["exhaustive", "greedy"] = "exhaustive",
    markov_search: Literal["exhaustive", "greedy"] = "exhaustive",
) -> dict:
    """
    Run both selection criteria independently and return both optimal
    subspaces.

    This is the simplest unification strategy and is useful when the
    downstream analysis can handle two candidate subspaces, or when a
    human expert will make the final choice.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Clean component matrix.
    N : int
        Target subspace dimensionality.
    metric : str, default ``'variance_sum'``
        Variance measure for the conservative fraction.
    n_bins : int, default 5
        Quantile bins for Markov discretisation.
    n_workers : int or None, optional
        Parallel workers.
    fc_search : ``'exhaustive'`` | ``'greedy'``, default ``'exhaustive'``
    markov_search : ``'exhaustive'`` | ``'greedy'``, default ``'exhaustive'``

    Returns
    -------
    result : dict
        Keys:

        * ``'fc_combination'``   — best subspace under conservative fraction.
        * ``'fc_value'``         — its conservative fraction.
        * ``'markov_combination'`` — best subspace under Markov time.
        * ``'markov_tau'``       — its relaxation time.
    """
    # Conservative fraction
    if fc_search == "exhaustive":
        fc_comb, fc_val = find_best_subspace_fc(
            data, N, metric=metric, n_workers=n_workers,
        )
    else:
        fc_comb_tuple, fc_val = greedy_forward_selection_fc(
            data, N, metric=metric,
        )
        fc_comb = tuple(fc_comb_tuple)

    # Markov time
    if markov_search == "exhaustive":
        markov_comb, markov_tau = find_best_subspace_markov(
            data, N, n_bins=n_bins, n_workers=n_workers,
        )
    else:
        markov_comb_tuple, markov_tau = greedy_forward_selection_markov(
            data, N, n_bins=n_bins,
        )
        markov_comb = tuple(markov_comb_tuple)

    return {
        "fc_combination": fc_comb,
        "fc_value": float(fc_val),
        "markov_combination": markov_comb,
        "markov_tau": float(markov_tau),
    }


# ---------------------------------------------------------------------------
# Helper: generate candidate combinations for greedy/approximate modes
# ---------------------------------------------------------------------------

def _generate_candidate_combinations(
    data: np.ndarray,
    N: int,
    *,
    n_candidates: int = 500,
) -> list[tuple[int, ...]]:
    """
    Generate a diverse set of candidate combinations for approximate
    search strategies.

    Strategy:
        1. Run greedy forward selection for fc (gives 1 candidate).
        2. Run greedy forward selection for Markov (gives 1 candidate).
        3. Fill the rest with random combinations.
    """
    import random

    D, T = data.shape

    candidates: list[tuple[int, ...]] = []

    # Greedy candidates
    greedy_fc, _ = greedy_forward_selection_fc(data, N)
    candidates.append(tuple(greedy_fc))

    greedy_markov, _ = greedy_forward_selection_markov(data, N)
    candidates.append(tuple(greedy_markov))

    # Random candidates
    rng = random.Random(42)
    remaining = n_candidates - len(candidates)
    seen = set(candidates)
    for _ in range(remaining * 5):  # oversample to handle collisions
        if len(candidates) >= n_candidates:
            break
        comb = tuple(sorted(rng.sample(range(D), N)))
        if comb not in seen:
            seen.add(comb)
            candidates.append(comb)

    return candidates[:n_candidates]

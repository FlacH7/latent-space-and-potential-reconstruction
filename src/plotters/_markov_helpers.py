"""
_markov_helpers.py — Helpers para plots de Stage 3 (Markov)
============================================================

Funciones de apoyo para los plots de la etapa de seleccion Markov.
Principalmente, recalcula la matriz de transicion ``P`` para una
combinacion de componentes seleccionada, replicando la logica de
``src.latent_space_extraction.markov_subspace``.
"""
from __future__ import annotations

import numpy as np


def compute_transition_matrix(
    data: np.ndarray,
    combination: tuple[int, ...],
    n_bins: int,
) -> np.ndarray:
    """
    Recalcula la matriz de transicion ``P`` para una combinacion dada.

    Replica la logica de
    :func:`src.latent_space_extraction.markov_subspace.discretize_series`
    + :func:`compute_markov_time`, pero retorna ``P`` en lugar de ``tau``.

    Parameters
    ----------
    data : np.ndarray, shape (D, T)
        Matriz de salida de Stage 2 (D componentes x T muestras).
    combination : tuple of int
        Indices de los componentes seleccionados.
    n_bins : int
        Numero de bins por componente (cuantiles).

    Returns
    -------
    P : np.ndarray, shape (n_states, n_states)
        Matriz de transicion estocastica por filas.
    n_states : int
        Numero de estados (= ``n_bins ** len(combination)``).
    state_sequence : np.ndarray, 1-D
        Secuencia de estados observada (longitud T).
    """
    from src.latent_space_extraction.markov_subspace import (
        discretize_series,
    )

    combination = tuple(int(i) for i in combination)
    bins_idx = discretize_series(data, n_bins)
    sub_bins = bins_idx[list(combination), :]   # (N, T)
    N = sub_bins.shape[0]

    powers = n_bins ** np.arange(N)
    state_sequence = np.dot(powers, sub_bins).astype(int)  # (T,)
    n_states = n_bins ** N

    T = len(state_sequence)
    P = np.zeros((n_states, n_states), dtype=float)
    for t in range(T - 1):
        i = int(state_sequence[t])
        j = int(state_sequence[t + 1])
        P[i, j] += 1.0

    row_sums = P.sum(axis=1, keepdims=True)
    # Evitar division por cero: filas no visitadas se dejan en cero
    nonzero = row_sums.ravel() > 0
    P[nonzero] = P[nonzero] / row_sums[nonzero]

    return P, n_states, state_sequence

"""
Hankel-DMD Latent Space Extractor
===================================
Extracts a low-dimensional latent subspace from EEG data via **Hankel
(delay) embedding** followed by **Dynamic Mode Decomposition (DMD)**.

This module is designed to plug into the existing pipeline as an
alternative to the ICA-based Stage-I + scoring path.  When
``scoring_method="hankel_dmd"`` is used in
:func:`extract_latent_subspace.extract_latent_space`, ICA/ICLabel are
bypassed: only band-pass filtering is applied, and this module is
called directly to produce the latent coordinates.

Typical usage (standalone)::

    from hankel_dmd_extractor import extract_hankel_dmd_latent_space

    # X: np.ndarray of shape (n_channels, n_times) — filtered EEG
    latent, meta = extract_hankel_dmd_latent_space(
        X,
        n_dim=4,                # latent rank P
        embedding_depth=60,     # Hankel delay T
        dt=1.0 / 250.0,         # sampling interval [s]
    )
    # latent.shape == (n_times - T + 1, n_dim)

Typical usage (via orchestrator)::

    from extract_latent_subspace import extract_latent_space

    latent, meta = extract_latent_space(
        "/path/to/raw.fif",
        n_dim=4,
        scoring_method="hankel_dmd",
        hankel_embedding_depth=60,
    )

Reference
---------
The Hankel+DMD pipeline follows the delay-embedding DMD approach
(Schmid 2010; Tu et al. 2014) adapted for multichannel EEG:

1. Average-reference and drop last channel (to avoid rank deficit).
2. Build a multivariate block-Hankel matrix from delay embeddings.
3. Truncated SVD to obtain the POD basis (left singular vectors).
4. Reduced DMD propagator and eigendecomposition in the POD basis.
5. Latent time courses = projection of Hankel matrix onto POD basis.

The returned latent coordinates are the **dominant temporal modes**
captured by the DMD, ordered by singular value magnitude.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.linalg import hankel as scipy_hankel
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_DEPTH: int = 60  # default delay snapshots (T)


def _auto_embedding_depth(sfreq: float, n_times: int) -> int:
    """
    Choose a sensible embedding depth *T* from the sampling rate and
    signal length.

    Rule of thumb: T is roughly 0.25--0.5 s worth of samples, bounded
    between ``max(50, min(200, n_times // 10))``.
    """
    t_samples = int(sfreq * 0.25)  # ~250 ms at sfreq
    t_samples = max(50, min(t_samples, 200))
    t_samples = min(t_samples, n_times // 10)
    return max(t_samples, 10)  # absolute floor


def _build_multivariate_hankel(
    X: np.ndarray,
    T: int,
) -> np.ndarray:
    """
    Build the multivariate block-Hankel matrix from a channel-time
    array.

    Parameters
    ----------
    X : ndarray, shape (Ne, Nt)
        Zero-mean channel data.
    T : int
        Embedding depth.

    Returns
    -------
    H : ndarray, shape (Ne * T, Nt - T + 1)
        Block-Hankel matrix.
    """
    Ne, Nt = X.shape
    if T > Nt:
        raise ValueError(
            f"Embedding depth T={T} cannot exceed number of time samples Nt={Nt}"
        )
    # 0-based Hankel index matrix
    idx = scipy_hankel(np.arange(T), np.arange(T - 1, Nt))
    H = X[:, idx].reshape(Ne * T, -1)
    return H


def _compute_dmd_spectra(
    A: np.ndarray,
    dt: float,
) -> dict[str, np.ndarray]:
    """
    Eigendecomposition of the reduced propagator → oscillatory spectra.

    Parameters
    ----------
    A : ndarray, shape (P, P)
        Reduced DMD operator.
    dt : float
        Sampling interval [s].

    Returns
    -------
    osc : dict
        Dictionary with keys ``lambda``, ``omega``, ``freq``, ``damp``.
    """
    lam, _ = np.linalg.eig(A)
    omega = np.log(lam) / dt
    freq = np.imag(omega) / (2 * np.pi)
    damp = np.real(omega)
    return {"lambda": lam, "omega": omega, "freq": freq, "damp": damp}


def eeg_hankel_dmd_core(
    V: np.ndarray,
    T: int,
    P: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Core Hankel+DMD pipeline.

    Parameters
    ----------
    V : ndarray, shape (Ne+1, Nt)
        Raw EEG matrix (channels x time).  Last row is discarded *after*
        average referencing.
    T : int
        Embedding depth.
    P : int
        Latent rank to retain (number of DMD modes).
    dt : float
        Sampling interval [s].

    Returns
    -------
    L : ndarray, shape (Ne * T, P)
        Dominant left singular vectors (POD basis).
    s : ndarray, shape (P,)
        Singular values (descending).
    H : ndarray, shape (Ne * T, Nt - T + 1)
        Multivariate block-Hankel matrix.
    osc : dict
        Oscillatory decomposition with keys:
        - ``lambda``: DMD eigenvalues (discrete-time)
        - ``omega`` : Continuous eigenvalues ``log(lambda)/dt``
        - ``freq``  : Oscillation frequencies [Hz]
        - ``damp``  : Growth/damping rates [1/s]
        - ``modes`` : DMD modes in delay-embedded space
    scores : ndarray, shape (P, Nt - T + 1)
        Latent time courses = projection of *H* onto the POD basis.
    """
    # a) Average-reference over ALL electrodes, THEN discard last channel
    V = V - V.mean(axis=0, keepdims=True)
    X = V[:-1, :]  # drop last channel to avoid rank deficit
    Ne, Nt = X.shape

    if P > Ne * T:
        raise ValueError(
            f"Requested rank P={P} exceeds available modes Ne*T={Ne * T} "
            f"(Ne={Ne}, T={T}). Reduce n_dim or increase embedding_depth."
        )

    # b) Build multivariate block-Hankel matrix
    H = _build_multivariate_hankel(X, T)

    # c) Truncated SVD via svds (memory-friendly for large H)
    U, s_vals, Vh = svds(H, k=P)

    # Ensure descending order
    ord_idx = np.argsort(s_vals)[::-1]
    s_vals = s_vals[ord_idx]
    U = U[:, ord_idx]
    Vh = Vh[ord_idx, :]

    L = U             # (Ne*T) x P   left singular vectors
    R = Vh.T          # (Nt-T+1) x P  right singular vectors

    # d) Reduced DMD propagator in the POD basis
    H1 = H[:, :-1]    # first Nt-T columns
    H2 = H[:, 1:]     # one-step forward shift
    R1 = R[:-1, :]    # right singular vectors of H1

    # P x P reduced operator
    A = (L.T @ H2 @ R1) @ np.diag(1.0 / s_vals)

    # e) Eigendecomposition → oscillatory components
    lam, W = np.linalg.eig(A)
    omega = np.log(lam) / dt
    freq = np.imag(omega) / (2 * np.pi)
    damp = np.real(omega)

    # DMD modes in full delay-embedded space
    modes = H2 @ R1 @ np.diag(1.0 / s_vals) @ W

    # Sort by |frequency| for readability
    isort = np.argsort(np.abs(freq))
    osc = {
        "lambda": lam[isort],
        "omega": omega[isort],
        "freq": freq[isort],
        "damp": damp[isort],
        "modes": modes[:, isort],
    }

    # Latent time courses — P x (Nt-T+1)
    scores = L.T @ H

    return L, s_vals, H, osc, scores


def extract_hankel_dmd_latent_space(
    X: np.ndarray,
    *,
    n_dim: int = 2,
    embedding_depth: int | None = None,
    dt: float | None = None,
    sfreq: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Extract a low-dimensional latent subspace from filtered EEG using
    Hankel delay embedding + DMD.

    This is the **entry-point** called by
    :func:`extract_latent_subspace.extract_latent_space` when
    ``scoring_method="hankel_dmd"``.

    Parameters
    ----------
    X : np.ndarray, shape (n_channels, n_times)
        Filtered EEG data matrix (channels x time samples).
    n_dim : int, default 2
        Target dimensionality of the latent subspace (latent rank *P*
        in DMD terminology).  This is the number of dominant DMD modes
        to retain.
    embedding_depth : int or None, optional
        Hankel embedding depth (*T* — number of past snapshots in the
        delay vector).

        * ``None`` → auto-computed from *sfreq* and signal length.
          Rule of thumb: ``T = clip(sfreq * 0.25, 50, 200)`` capped at
          ``n_times // 10``.
        * ``int`` → explicit value.  Must be ``< n_times``.

    dt : float or None, optional
        Sampling interval in seconds (``1 / sfreq``).  If *sfreq* is
        also provided, ``dt`` takes precedence.
    sfreq : float or None, optional
        Sampling frequency in Hz.  Used to auto-compute *dt* and
        *embedding_depth* when those are not provided explicitly.

    Returns
    -------
    latent : np.ndarray, shape (n_samples, n_dim)
        The extracted latent subspace.  Each column is one latent
        dimension (POD mode time series).  Because of the delay
        embedding, ``n_samples = n_times - T + 1``.
    meta : dict
        Metadata including:

        * ``selected_indices`` — mode indices ``[0, 1, ..., n_dim-1]``
          (kept for API consistency with other scoring methods).
        * ``scoring_method`` — ``"hankel_dmd"``.
        * ``latent_scores`` — DMD spectral info: frequencies [Hz],
          damping rates [1/s], eigenvalues.
        * ``preprocessing`` — filter params, channel count, embedding
          depth, singular values, etc.
        * ``elapsed_time`` — wall-clock time in seconds.
    """
    t0 = time.time()

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (channels x time), got shape {X.shape}")

    n_channels, n_times = X.shape

    # Resolve dt
    if dt is None:
        if sfreq is None:
            raise ValueError(
                "Either dt or sfreq must be provided for Hankel+DMD."
            )
        dt = 1.0 / float(sfreq)
    else:
        dt = float(dt)
        if sfreq is None:
            sfreq = 1.0 / dt

    # Resolve embedding depth
    if embedding_depth is None:
        if sfreq is None:
            raise ValueError(
                "Cannot auto-compute embedding_depth without sfreq or dt."
            )
        T = _auto_embedding_depth(sfreq, n_times)
    else:
        T = int(embedding_depth)

    if T >= n_times:
        raise ValueError(
            f"embedding_depth ({T}) must be < n_times ({n_times}). "
            f"Reduce T or use a longer recording segment."
        )

    if n_dim < 1:
        raise ValueError(f"n_dim must be >= 1, got {n_dim}")

    # We need Ne+1 channels for the avg-ref + drop-last step.
    # If we have exactly 1 channel, duplicate it so the avg-ref works.
    if n_channels == 1:
        V = np.vstack([X, X.copy()])  # shape (2, n_times)
    else:
        V = X  # shape (Ne+1, n_times) or more

    # ------------------------------------------------------------------
    # Run core Hankel+DMD
    # ------------------------------------------------------------------
    L, s_vals, H, osc, scores = eeg_hankel_dmd_core(V, T, n_dim, dt)

    # scores: (n_dim, n_samples_latent)  → transpose to (n_samples, n_dim)
    latent = scores.T

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Build metadata — same top-level keys as the ICA-based path
    # ------------------------------------------------------------------
    n_samples_latent = latent.shape[0]
    n_channels_after_ref = V.shape[0] - 1  # after dropping last

    meta: dict[str, Any] = {
        "selected_indices": list(range(n_dim)),
        "scoring_method": "hankel_dmd",
        "latent_scores": {
            "frequencies_hz": osc["freq"].real.tolist(),
            "damping_rates": osc["damp"].real.tolist(),
            "eigenvalues": osc["lambda"].tolist(),
            "continuous_eigenvalues": osc["omega"].tolist(),
        },
        "preprocessing": {
            "D": n_channels_after_ref,
            "T": T,
            "sfreq": float(sfreq),
            "n_channels": n_channels,
            "n_times": n_times,
            "n_samples_latent": n_samples_latent,
            "excluded_indices": [],
            "kept_indices": list(range(n_channels)),
            "embedding_depth": T,
            "latent_rank": n_dim,
            "dt": dt,
            "singular_values": s_vals.tolist(),
            "hankel_shape": H.shape,
        },
        "Y": X,  # filtered data matrix for downstream reuse
        "Y_shape": (n_channels, n_times),
        "elapsed_time": elapsed,
        # Extra DMD-specific info
        "dmd": {
            "left_singular_vectors_shape": L.shape,
            "singular_values": s_vals.tolist(),
            "hankel_shape": H.shape,
            "oscillatory": osc,
        },
    }

    return latent, meta

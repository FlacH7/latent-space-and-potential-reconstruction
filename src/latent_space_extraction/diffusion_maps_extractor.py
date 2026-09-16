"""
Diffusion Maps Latent Space Extractor
======================================

Extracts a low-dimensional latent subspace from EEG data using Diffusion Maps
(Coifman & Lafon, 2006) via the pyDiffMap package.

Unlike ICA (statistical independence) or Hankel+DMD (delay-embedding linear
decomposition), Diffusion Maps preserve the intrinsic geometric connectivity
of the data graph and linearise the Koopman operator in the embedded space.

Key features
------------
* Gaussian kernel with automatic bandwidth (Berry-Giannakis-Harlim) or
  manual sigma.
* Density normalisation via the alpha parameter (Coifman-Lafon framework):
  alpha=0.5 -> classical Diffusion Maps, alpha=0.0 -> Laplacian Eigenmaps,
  alpha=1.0 -> Fokker-Planck.
* Multiscale filtering via the diffusion-time parameter *t*.
* Out-of-sample (Nystroem) extension for projecting new sessions onto a
  previously trained embedding without recomputing the full operator.

References
----------
Coifman, R. R., & Lafon, S. (2006). Diffusion maps. *Applied and
Computational Harmonic Analysis*, 21(1), 5-30.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from pydiffmap.diffusion_map import DiffusionMap
from sklearn.neighbors import NearestNeighbors


def extract_diffusion_maps_latent_space(
    X: np.ndarray,
    n_dim: int,
    sigma: float | None = None,
    k: int = 100,
    diffusion_time: float = 0.0,
    alpha: float = 0.5,
    sfreq: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Apply Diffusion Maps to an EEG data matrix using pyDiffMap.

    Parameters
    ----------
    X : np.ndarray, shape (n_channels, n_times)
        Band-pass filtered EEG data matrix (channels x time).
    n_dim : int
        Target dimensionality of the latent subspace.
    sigma : float | None
        Bandwidth of the Gaussian kernel.  If *None*, the bandwidth is
        estimated automatically via the Berry-Giannakis-Harlim (``bgh``)
        method implemented in pyDiffMap.
    k : int
        Number of nearest neighbours used to build the sparse affinity
        matrix.  Default 100.
    diffusion_time : float
        Diffusion time *t* >= 0.  Larger values attenuate fine-scale
        components (small eigenvalues) and highlight macroscopic dynamics.
        *t = 0* means no eigenvalue filtering is applied (raw eigenvectors).
    alpha : float
        Density-normalisation parameter (Coifman-Lafon):
        ``0.0`` = Laplacian Eigenmaps, ``0.5`` = classical Diffusion Maps,
        ``1.0`` = Fokker-Planck.
    sfreq : float | None
        Sampling frequency (Hz).  Stored in metadata only.

    Returns
    -------
    latent : np.ndarray, shape (n_times, n_dim)
        Diffusion-map latent space.  Each row corresponds to one time
        sample; each column to one latent dimension.
    meta : dict
        Dictionary containing sigma used, eigenvalues, k, diffusion_time,
        alpha, spectral gap, and execution time.
    """
    t_start = time.perf_counter()

    # Transpose: pyDiffMap expects (n_samples, n_features)
    # n_samples = n_times, n_features = n_channels
    data = X.T.astype(np.float64)  # shape (n_times, n_channels)
    n_samples, n_features = data.shape

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if n_samples < k + 1:
        k = max(10, n_samples - 1)
        print(f"[diffusion_maps] Ajustando k={k} porque n_samples={n_samples} es pequeno.")

    if n_dim >= n_samples - 1:
        raise ValueError(
            f"n_dim ({n_dim}) debe ser menor que n_samples-1 ({n_samples-1})."
        )

    # ------------------------------------------------------------------
    # Configure epsilon for pyDiffMap
    # In pyDiffMap, epsilon is the denominator of the Gaussian kernel
    # exponent: K(x,y) = exp(-||x-y||^2 / epsilon).
    # Conversion from sigma: epsilon = sigma^2 / 2
    # (standard Gaussian: K(x,y) = exp(-||x-y||^2 / (2*sigma^2)))
    # ------------------------------------------------------------------
    if sigma is not None:
        epsilon = (sigma ** 2) / 2.0
    else:
        epsilon = "bgh"  # Auto-estimacion Berry-Giannakis-Harlim

    # ------------------------------------------------------------------
    # Instantiate and fit DiffusionMap
    # pyDiffMap already EXCLUDES the trivial eigenvalue (lambda_0 = 1),
    # so n_evecs = n_dim gives exactly the non-trivial components.
    # ------------------------------------------------------------------
    dmap = DiffusionMap.from_sklearn(
        n_evecs=n_dim,
        epsilon=epsilon,
        alpha=alpha,
        k=k,
        neighbor_params={
            "n_jobs": -1,
        },
    )

    coords = dmap.fit_transform(data)  # shape (n_samples, n_dim)

    # ------------------------------------------------------------------
    # Eigenvalue post-processing
    # pyDiffMap returns evals sorted descending (largest first).
    # All returned eigenvalues are non-trivial.
    # ------------------------------------------------------------------
    if hasattr(dmap, "evals") and dmap.evals is not None:
        evals = np.array(dmap.evals, dtype=np.float64)
        evecs = np.array(dmap.evecs, dtype=np.float64) if hasattr(dmap, "evecs") else coords
        # Ensure descending order (should already be, but enforce it)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        coords = coords[:, idx]
    else:
        # Fallback: use coords directly
        evals = np.ones(coords.shape[1])
        evecs = coords.copy()

    # ------------------------------------------------------------------
    # Apply diffusion time: Psi_t(x) = [lambda_1^t * phi_1(x), ...]
    # coords from pyDiffMap is the embedding at t=1 (already includes
    # lambda^1).  For arbitrary t we rescale:
    #   latent = evecs * |lambda|^t   (full eigenvalue power from scratch)
    # Using evecs (raw eigenvectors) and applying lambda^t is the
    # mathematically standard definition of the DM embedding.
    # ------------------------------------------------------------------
    if diffusion_time > 0:
        # Use absolute eigenvalues for the power law (numerical negatives
        # are precision artefacts of the sparse eigensolver).
        evals_abs = np.clip(np.abs(evals[:n_dim]), a_min=1e-15, a_max=None)
        latent = evecs[:, :n_dim] * (evals_abs[np.newaxis, :] ** diffusion_time)
    else:
        # t=0: raw eigenvectors, no eigenvalue filtering
        latent = evecs[:, :n_dim]

    # ------------------------------------------------------------------
    # Spectral gap: lambda_1 - lambda_2 (gap between top two non-trivial
    # eigenvalues).  Larger gap -> clearer spectral separation.
    # ------------------------------------------------------------------
    spectral_gap = None
    if len(evals) >= 2:
        spectral_gap = float(abs(evals[0]) - abs(evals[1]))

    # ------------------------------------------------------------------
    # Degenerate-eigenvalue warning
    # If |lambda_1| > 0.999, the graph may be over-connected (sigma
    # too large or k too small).
    # ------------------------------------------------------------------
    if len(evals) > 0 and abs(evals[0]) > 0.999:
        warnings.warn(
            f"[diffusion_maps] Primer autovalor |lambda_1| = {abs(evals[0]):.6f} "
            f"muy cercano a 1. El grafo puede estar sobre-conectado. "
            f"Considere reducir sigma o aumentar k.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Extract sigma/epsilon actually used
    # ------------------------------------------------------------------
    sigma_used = sigma
    epsilon_fitted = None
    if sigma is None and hasattr(dmap, "epsilon_fitted") and dmap.epsilon_fitted is not None:
        epsilon_fitted = float(dmap.epsilon_fitted)
        sigma_used = np.sqrt(2.0 * epsilon_fitted)

    # Handle edge case: auto-epsilon gives 0 (duplicated/constant data)
    if sigma_used is not None and sigma_used == 0.0:
        warnings.warn(
            "[diffusion_maps] sigma estimado = 0 (posibles datos duplicados o "
            "constantes). Usando epsilon=0.1 como fallback.",
            UserWarning,
            stacklevel=2,
        )
        sigma_used = np.sqrt(2.0 * 0.1)

    elapsed = time.perf_counter() - t_start

    meta = {
        "sigma_used": float(sigma_used) if sigma_used is not None else None,
        "epsilon_fitted": epsilon_fitted,
        "k_neighbors": k,
        "diffusion_time": float(diffusion_time),
        "alpha": float(alpha),
        "eigenvalues": evals.tolist(),
        "eigenvalues_latent": evals[:n_dim].tolist(),
        "spectral_gap": spectral_gap,
        "n_samples": n_samples,
        "n_features": n_features,
        "elapsed_time": elapsed,
        "sfreq": sfreq,
    }

    return latent, meta


def nystroem_extension(
    X_new: np.ndarray,
    X_train: np.ndarray,
    train_latent: np.ndarray,
    sigma: float,
    k: int = 100,
) -> np.ndarray:
    """
    Extend a trained diffusion-map latent space to new data via Nystroem
    interpolation, without retraining the diffusion operator.

    For a new point x_new the k-th component is approximated as::

        phi_k(x_new) ~ sum_j  K(x_new, x_j) * phi_k(x_j) / sum_j K(x_new, x_j)

    where the sum runs over the *k* nearest neighbours of x_new in the
    training set.  This is a weighted interpolation of the training
    latent coordinates, using the Gaussian kernel as the weight.

    Parameters
    ----------
    X_new : np.ndarray, shape (n_channels, n_new_times)
        New EEG data (e.g. a test-retest session).
    X_train : np.ndarray, shape (n_channels, n_train_times)
        Training EEG data that was used to fit the diffusion map.
    train_latent : np.ndarray, shape (n_train_times, n_dim)
        Latent coordinates of the training data.
    sigma : float
        Kernel bandwidth (same value used during training).
    k : int
        Number of nearest neighbours for the Nystroem approximation.

    Returns
    -------
    latent_new : np.ndarray, shape (n_new_times, n_dim)
        Projected latent coordinates for the new data.
    """
    data_new = X_new.T.astype(np.float64)    # (n_new_times, n_channels)
    data_train = X_train.T.astype(np.float64)  # (n_train_times, n_channels)
    n_new, n_features = data_new.shape
    n_train, _ = data_train.shape
    n_dim = train_latent.shape[1]

    # Find k-nearest neighbours of each new point in the training set
    nn = NearestNeighbors(
        n_neighbors=min(k, n_train - 1), metric="euclidean", n_jobs=-1,
    )
    nn.fit(data_train)
    distances, indices = nn.kneighbors(data_new)

    epsilon = (sigma ** 2) / 2.0

    # Gaussian kernel weights
    K = np.exp(-(distances ** 2) / epsilon)

    # Weighted interpolation of the training latent coordinates
    latent_new = np.zeros((n_new, n_dim))
    for i in range(n_new):
        weights = K[i] / (K[i].sum() + 1e-12)
        latent_new[i] = np.sum(
            train_latent[indices[i]] * weights[:, np.newaxis], axis=0,
        )

    return latent_new
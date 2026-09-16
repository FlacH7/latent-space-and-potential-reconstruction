"""
Stage 1 — Embedding (optional)
==============================

First stage of the modular latent-space pipeline.  It takes the filtered
channel matrix ``X`` of shape ``(n_channels, n_times)`` and optionally
builds a multivariate delay-embedding (block-Hankel) matrix.

Variants
--------
* ``None`` / ``IdentityEmbedding`` — the signal passes through unchanged:
  ``data`` stays ``(N_c, N_t)``.
* ``"hankel"`` / ``HankelEmbedding`` — builds the block-Hankel matrix

      H ∈ R^{(N_c·T) × (N_t − T + 1)}

  where ``T`` is the embedding depth.  When ``depth=None`` it is
  auto-computed as ``clip(sfreq * 0.25, 50, 200)`` (capped at
  ``n_times // 10``), following ``hankel_dmd_extractor.py``.

Shape convention (unbreakable rule)
-----------------------------------
All internal matrices flow as ``(n_features, n_samples)`` — rows are
variables/modes/channels, columns are time.

The embedding stage keeps a reference to the *original* channel matrix in
its metadata (``meta["input_data"]``).  Some Stage-2 dynamics (e.g. DMD
with Hankel) need it to reproduce the numerics of the legacy pipeline
exactly (average-reference + drop-last-channel happen inside
``eeg_hankel_dmd_core``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.latent_space_extraction.hankel_dmd_extractor import (
    _auto_embedding_depth,
    _build_multivariate_hankel,
)


# ---------------------------------------------------------------------------
# Stage context shared across stages
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """
    Shared, read-only context propagated through the three stages.

    Attributes
    ----------
    raw : mne.io.Raw | None
        The (filtered or unfiltered) MNE Raw object, when available.
        Required by Stage-2 ``pca_ica`` *without* Hankel (MNE ICA needs a
        real Raw object with channel names / positions).  Not used by the
        sklearn-ICA Hankel branch.
    sfreq : float
        Sampling frequency in Hz.
    dt : float
        Sampling interval in seconds (``1 / sfreq``).
    l_freq, h_freq : float
        Band-pass filter cut-offs applied upstream.
    n_workers : int | None
        Parallel processes for combinatorial searches (Stage 3).
    verbose : bool | str | None
        Verbosity level.
    stage1_name : str | None
        Name of the embedding stage actually executed (``None`` or
        ``"hankel"``).  Filled in by the orchestrator so that Stage 2 can
        adapt its numerics (see the critical handoffs in the refactor
        specification).
    stage1_meta : dict
        Metadata emitted by Stage 1 (includes ``input_data`` — the
        original channel matrix — and ``depth`` for the Hankel variant).
    channel_intersection : list[str] | None
        Global channel intersection across all super-subjects (set by
        the batch runner).  Used by CD-HSA strategies to validate that
        the Hankel dimensionality matches the trained modes.
    """

    raw: Any = None
    sfreq: float | None = None
    dt: float | None = None
    l_freq: float = 1.0
    h_freq: float = 40.0
    n_workers: int | None = None
    verbose: Any = None
    stage1_name: str | None = None
    stage1_meta: dict = field(default_factory=dict)
    channel_intersection: list[str] | None = None

    @property
    def has_hankel(self) -> bool:
        """True when Stage 1 built a Hankel delay-embedding matrix."""
        return self.stage1_name == "hankel"


# ---------------------------------------------------------------------------
# Stage 1 implementations
# ---------------------------------------------------------------------------

class IdentityEmbedding:
    """
    Pass-through embedding: the filtered channel matrix flows unchanged.

    ``fit_transform(X)`` returns ``X`` itself with shape
    ``(n_channels, n_times)``.
    """

    name = None  # canonical name in the API is ``None``

    def fit_transform(
        self,
        data: np.ndarray,
        *,
        ctx: PipelineContext | None = None,
        **params,
    ) -> tuple[np.ndarray, dict]:
        if data.ndim != 2:
            raise ValueError(
                f"[Stage1/identity] Expected a 2-D matrix (features, samples), "
                f"got shape {data.shape}"
            )
        meta = {
            "embedding": None,
            "input_shape": tuple(data.shape),
            "output_shape": tuple(data.shape),
            "n_time_lost": 0,
            # Kept for interface uniformity with HankelEmbedding; Stage-2
            # branches that need the original channels read this entry.
            "input_data": data,
        }
        return data, meta


class HankelEmbedding:
    """
    Multivariate block-Hankel delay embedding.

    Parameters (via ``stage1_params``)
    ----------------------------------
    depth : int | None
        Embedding depth *T* (number of delayed snapshots stacked per
        column).  ``None`` → auto-computed as
        ``clip(sfreq * 0.25, 50, 200)`` capped at ``n_times // 10``
        (delegated to :func:`hankel_dmd_extractor._auto_embedding_depth`).

    Notes
    -----
    * No average-referencing or channel dropping is applied here: those
      operations are specific to the DMD numerics and remain inside
      ``eeg_hankel_dmd_core`` (Empate 4 of the specification).
    * The output ``H`` has shape ``(n_channels * T, n_times - T + 1)``.
    """

    name = "hankel"

    def __init__(self, depth: int | None = None):
        self.depth = depth

    def fit_transform(
        self,
        data: np.ndarray,
        *,
        ctx: PipelineContext | None = None,
        **params,
    ) -> tuple[np.ndarray, dict]:
        if data.ndim != 2:
            raise ValueError(
                f"[Stage1/hankel] Expected a 2-D matrix (channels, times), "
                f"got shape {data.shape}"
            )

        n_ch, n_times = data.shape

        depth = self.depth
        if depth is None:
            sfreq = getattr(ctx, "sfreq", None) if ctx is not None else None
            if sfreq is None:
                raise ValueError(
                    "[Stage1/hankel] depth=None requires ctx.sfreq to "
                    "auto-compute the embedding depth."
                )
            depth = _auto_embedding_depth(sfreq, n_times)
        depth = int(depth)

        if depth >= n_times:
            raise ValueError(
                f"[Stage1/hankel] embedding depth ({depth}) must be < "
                f"n_times ({n_times}). Reduce depth or use a longer segment."
            )

        H = _build_multivariate_hankel(data, depth)

        meta = {
            "embedding": "hankel",
            "depth": depth,
            "input_shape": (n_ch, n_times),
            "output_shape": tuple(H.shape),
            "n_time_lost": depth - 1,
            "input_data": data,  # needed by Stage-2 DMD (legacy numerics)
        }
        return H, meta


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

STAGE1_REGISTRY: dict[str | None, type] = {
    None: IdentityEmbedding,
    "hankel": HankelEmbedding,
}


def build_stage1(
    name: str | None,
    params: dict | None = None,
):
    """
    Instantiate a Stage-1 embedding by name.

    Parameters
    ----------
    name : None | "hankel"
    params : dict | None
        Forwarded to the class constructor (e.g. ``{"depth": 250}``).

    Raises
    ------
    ValueError
        If the embedding name is unknown.
    """
    params = dict(params or {})
    if name not in STAGE1_REGISTRY:
        valid = [repr(k) for k in STAGE1_REGISTRY]
        raise ValueError(
            f"[Stage1] Unknown embedding {name!r}. Valid options: "
            f"{', '.join(valid)}"
        )
    cls = STAGE1_REGISTRY[name]
    if cls is IdentityEmbedding:
        if params:
            raise ValueError(
                f"[Stage1] embedding=None takes no parameters, got {params}"
            )
        return cls()
    return cls(**params)

"""
EEG Preprocessing Module
========================

Handles the complete Stage I pipeline:
    1. Data loading and channel selection (EEG + EOG)
    2. Band-pass filtering (1 Hz - 40 Hz)
    3. PCA whitening + ICA decomposition (PICARD)
    4. ICLabel-based artifact rejection

Output:
    Clean component matrix Y of shape (D, T) where each row is a
    zero-mean independent component time series.

All functions operate on mne.Raw objects or NumPy arrays and are
fully reusable for batch processing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import mne
import numpy as np
from mne.preprocessing import ICA


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default filter cutoffs (Hz) — see paper, Section "Filtering and channel selection"
DEFAULT_L_FREQ: float = 1.0
DEFAULT_H_FREQ: float = 40.0

# ICA parameters
DEFAULT_ICA_METHOD: str = "picard"
DEFAULT_ICA_RANDOM_STATE: int = 42

# ICLabel classes to retain (strict rule: only 'brain' or 'other')
DEFAULT_RETAINED_LABELS: frozenset[str] = frozenset({"brain", "other"})


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_raw_eeg(
    file_path: str | Path,
    *,
    preload: bool = True,
    pick_eeg: bool = True,
    pick_eog: bool = True,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Load a raw EEG recording and select only EEG (and optionally EOG) channels.

    Non-EEG channels (MEG, stimulus, misc) are dropped as described in Stage I
    of the pipeline.

    Parameters
    ----------
    file_path : str or Path
        Path to the raw data file (``.fif``, ``.edf``, ``.bdf``, etc.).
    preload : bool, default True
        If True, data are loaded into memory.
    pick_eeg : bool, default True
        Retain EEG channels.
    pick_eog : bool, default True
        Retain EOG channels (useful for ICA to help identify ocular artifacts).
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        Filtered (in the channel-type sense) raw object.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    raw = mne.io.read_raw(file_path, preload=preload, verbose=verbose)
    raw.pick_types(meg=False, eeg=pick_eeg, eog=pick_eog, verbose=verbose)
    return raw


def load_sample_mne_data(
    dataset: str = "sample",
    *,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Load one of MNE's built-in sample datasets for quick testing.

    Parameters
    ----------
    dataset : str, default ``"sample"``
        Which built-in dataset to fetch. Currently only ``"sample"`` is
        supported.
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        Raw object with only EEG + EOG channels selected.
    """
    if dataset == "sample":
        sample_dir = mne.datasets.sample.data_path()
        raw_fname = sample_dir / "MEG" / "sample" / "sample_audvis_raw.fif"
    else:
        raise ValueError(f"Unsupported built-in dataset: {dataset!r}")

    return load_raw_eeg(raw_fname, preload=True, verbose=verbose)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_bandpass_filter(
    raw: mne.io.Raw,
    *,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Apply a band-pass FIR filter to the raw data.

    Default cutoffs (1 Hz, 40 Hz) follow Stage I of the pipeline:
        * High-pass at 1 Hz removes slow drifts and DC offsets that distort
          the covariance estimate used for whitening.
        * Low-pass at 40 Hz is above the band of interest for most cortical
          rhythms and provides anti-aliasing guard.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object to filter (modified in-place).
    l_freq : float, default 1.0
        Lower cutoff frequency in Hz.
    h_freq : float, default 40.0
        Upper cutoff frequency in Hz.
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        The same object, filtered in-place.
    """
    # Skip filtering if both frequencies are None (non-EEG data)
    if l_freq is None and h_freq is None:
        return raw
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=verbose)
    return raw


# ---------------------------------------------------------------------------
# ICA Decomposition
# ---------------------------------------------------------------------------

def fit_ica(
    raw: mne.io.Raw,
    *,
    n_components: int | float | None = None,
    method: str = DEFAULT_ICA_METHOD,
    random_state: int | None = DEFAULT_ICA_RANDOM_STATE,
    max_iter: int | str = "auto",
    verbose: bool | str | None = None,
) -> ICA:
    """
    Fit ICA on the (pre-filtered) raw data.

    MNE handles PCA whitening internally as part of the ICA fitting process.
    The whitening matrix ``W = Lambda^{-1/2} E^T`` (see Equation 2 in the
    paper) is not exposed explicitly, but the resulting unmixing matrix
    ``W_ica = M * Lambda^{-1/2} * E^T`` (Equation 3) can be accessed via
    ``ica.unmixing_matrix_``.

    Parameters
    ----------
    raw : mne.io.Raw
        Filtered raw object.
    n_components : int | float | None, optional
        Number of ICA components to compute.

        * ``None``  ->  all channels minus one.
        * ``int``   ->  exact number of components.
        * ``float`` ->  fraction of explained variance (0.0 < x <= 1.0).

        The paper recommends using all available channels (minus one) to
        avoid premature dimensionality reduction before artifact rejection.
    method : str, default ``"picard"``
        ICA algorithm. ``"picard"`` is recommended (see paper, Section
        "ICA decomposition") as it converges faster than FastICA or
        Infomax on high-dimensional EEG data.
    random_state : int | None, default 42
        Random seed for reproducibility.
    max_iter : int or str, default ``"auto"``
        Maximum number of iterations.
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    """
    n_ch = len(raw.ch_names)

    if n_components is None:
        n_components = n_ch
    elif isinstance(n_components, float):
        if not 0.0 < n_components <= 1.0:
            raise ValueError("n_components as float must be in (0.0, 1.0]")

    ica = ICA(
        n_components=n_components,
        method=method,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose,
    )
    ica.fit(raw, verbose=verbose)
    return ica


# ---------------------------------------------------------------------------
# Artifact Rejection (ICLabel)
# ---------------------------------------------------------------------------

def classify_components(
    raw: mne.io.Raw,
    ica: ICA,
    method: str = "iclabel",
) -> dict:
    """
    Automatically classify ICA components using ICLabel.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object used for fitting ICA.
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    method : str, default ``"iclabel"``
        Labeling method. Only ``"iclabel"`` is supported.

    Returns
    -------
    labels : dict
        Dictionary with keys ``'labels'`` (list of predicted label strings)
        and ``'y_pred_proba'`` (array of class probabilities).

    Raises
    ------
    ImportError
        If ``mne_icalabel`` is not installed.
    """
    try:
        from mne_icalabel import label_components
    except ImportError as exc:
        raise ImportError(
            "mne_icalabel is required for automatic component classification. "
            "Install it with:  pip install mne-icalabel"
        ) from exc

    return label_components(raw, ica, method=method)

def print_label_counts(labels: dict) -> None:
    """
    Print the count of each predicted ICLabel class.
    
    Parameters
    ----------
    labels : dict
        Output of :func:`classify_components`.
    """
    from collections import Counter
    predicted = labels["labels"]
    counts = Counter(predicted)
    print("\n  ICLabel component counts:")
    for label, count in sorted(counts.items()):
        print(f"    {label:15s}: {count}")


def get_artifact_indices(
    labels: dict,
    retained_labels: Sequence[str] | None = None,
) -> list[int]:
    """
    Determine which component indices to *exclude* based on ICLabel output.

    The default strict rule (see paper, Section "Artifact rejection with
    ICLabel") retains only components labeled ``'brain'`` or ``'other'``.

    Parameters
    ----------
    labels : dict
        Output of :func:`classify_components`.
    retained_labels : sequence of str, optional
        Labels to *keep*. All others are marked as artifacts. Defaults to
        ``{'brain', 'other'}``.

    Returns
    -------
    exclude_indices : list[int]
        List of component indices to exclude.
    """
    if retained_labels is None:
        retained_labels = DEFAULT_RETAINED_LABELS

    predicted = labels["labels"]
    exclude = [idx for idx, label in enumerate(predicted) if label not in retained_labels]
    return exclude


def reject_artifacts(
    ica: ICA,
    exclude_indices: list[int],
) -> ICA:
    """
    Mark artifact components for exclusion *in-place*.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    exclude_indices : list[int]
        Component indices to exclude (output of :func:`get_artifact_indices`).

    Returns
    -------
    ica : mne.preprocessing.ICA
        The same object with ``ica.exclude`` populated.
    """
    ica.exclude = list(exclude_indices)
    return ica


# ---------------------------------------------------------------------------
# Clean Component Extraction
# ---------------------------------------------------------------------------

def extract_clean_components(
    raw: mne.io.Raw,
    ica: ICA,
    *,
    apply_exclude: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """
    Extract the time courses of the surviving (non-artifact) ICA components.

    This produces the clean data matrix ``Y`` described in the paper
    (Equation 4), of shape ``(D, T)`` where ``D`` is the number of retained
    components and ``T`` is the number of time samples.

    Each row is automatically centered to zero mean, as required by the
    downstream conservative-fraction and Markov-time analyses.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object.
    ica : mne.preprocessing.ICA
        Fitted ICA object with ``ica.exclude`` already set.
    apply_exclude : bool, default True
        If True, omit excluded components from the output.

    Returns
    -------
    Y : np.ndarray, shape (D, T)
        Clean, zero-mean component matrix.
    kept_indices : list[int]
        Indices of the retained components (useful for tracing back to ICA).
    """
    sources = ica.get_sources(raw)
    sources_data = sources.get_data()  # shape (n_ica_components, T)

    if apply_exclude:
        excluded = set(ica.exclude or [])
        kept_indices = [i for i in range(sources_data.shape[0]) if i not in excluded]
        Y = sources_data[kept_indices, :]
    else:
        kept_indices = list(range(sources_data.shape[0]))
        Y = sources_data

    # Center each component (zero mean per row) — required by downstream stages
    Y = Y - Y.mean(axis=1, keepdims=True)

    return Y, kept_indices


# ---------------------------------------------------------------------------
# Filtered Data Matrix Extraction (Hankel+DMD bypass — no ICA)
# ---------------------------------------------------------------------------

def extract_filtered_data_matrix(
    raw: mne.io.Raw,
    *,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    verbose: bool | str | None = None,
) -> tuple[np.ndarray, mne.io.Raw, float]:
    """
    Apply band-pass filtering and return the data as a NumPy matrix.

    This is a **lightweight Stage-I** used when the downstream latent-space
    extractor (e.g. Hankel+DMD) does **not** require ICA decomposition or
    ICLabel artifact rejection.  Only filtering is applied.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG recording (channels should already be restricted to EEG+EOG).
    l_freq : float, default 1.0
        High-pass cutoff (Hz).
    h_freq : float, default 40.0
        Low-pass cutoff (Hz).
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    X : np.ndarray, shape (n_channels, n_times)
        Filtered data matrix (channels x time samples).  Each row is
        centered to zero mean.
    raw_filtered : mne.io.Raw
        The filtered MNE Raw object (useful for accessing ``raw.info``).
    sfreq : float
        Sampling frequency in Hz.
    """
    raw_filtered = apply_bandpass_filter(
        raw, l_freq=l_freq, h_freq=h_freq, verbose=verbose
    )

    X = raw_filtered.get_data()  # shape (n_channels, n_times)

    # Center each channel (zero mean per row) — consistent with ICA path
    X = X - X.mean(axis=1, keepdims=True)

    sfreq = raw_filtered.info["sfreq"]

    return X, raw_filtered, sfreq


# ---------------------------------------------------------------------------
# Full Stage-I Pipeline (convenience)
# ---------------------------------------------------------------------------

def run_full_preprocessing(
    raw: mne.io.Raw,
    *,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    n_components: int | float | None = None,
    ica_method: str = DEFAULT_ICA_METHOD,
    ica_random_state: int | None = DEFAULT_ICA_RANDOM_STATE,
    retained_labels: Sequence[str] | None = None,
    verbose: bool | str | None = None,
) -> dict:
    """
    Run the complete Stage I preprocessing pipeline.

    This is a convenience wrapper that chains filtering, ICA fitting,
    ICLabel classification, artifact rejection, and clean-component
    extraction into a single call.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG recording (channels should already be restricted to EEG+EOG).
    l_freq : float, default 1.0
        High-pass cutoff (Hz).
    h_freq : float, default 40.0
        Low-pass cutoff (Hz).
    n_components : int | float | None, optional
        Number of ICA components. ``None`` means ``n_channels - 1``.
    ica_method : str, default ``"picard"``
        ICA algorithm passed to MNE.
    ica_random_state : int | None, default 42
        Random seed.
    retained_labels : sequence of str, optional
        ICLabel classes to keep. Defaults to ``['brain', 'other']``.
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    result : dict
        Dictionary with the following keys:

        * ``'Y'``                — clean component matrix (D, T)
        * ``'ica'``              — fitted ICA object
        * ``'kept_indices'``     — indices of retained components
        * ``'excluded_indices'`` — indices of rejected (artifact) components
        * ``'labels'``           — full ICLabel output dictionary
        * ``'raw_filtered'``     — the filtered Raw object
        * ``'D'``                — number of clean components
        * ``'T'``                — number of time samples
        * ``'n_channels'``       — original channel count
        * ``'sfreq'``            — sampling frequency
    """
    # 1. Filtering
    raw_filtered = apply_bandpass_filter(
        raw, l_freq=l_freq, h_freq=h_freq, verbose=verbose
    )

    # 2. ICA decomposition (PCA whitening happens internally)
    ica = fit_ica(
        raw_filtered,
        n_components=n_components,
        method=ica_method,
        random_state=ica_random_state,
        verbose=verbose,
    )

    # 3. ICLabel classification
    labels = classify_components(raw_filtered, ica)
    
    print_label_counts(labels)

    # 4. Artifact rejection
    excluded = get_artifact_indices(labels, retained_labels=retained_labels)
    reject_artifacts(ica, excluded)

    # 5. Extract clean components
    Y, kept_indices = extract_clean_components(raw_filtered, ica)

    D, T = Y.shape

    return {
        "Y": Y,
        "ica": ica,
        "kept_indices": kept_indices,
        "excluded_indices": excluded,
        "labels": labels,
        "raw_filtered": raw_filtered,
        "D": D,
        "T": T,
        "n_channels": len(raw.ch_names),
        "sfreq": raw.info["sfreq"],
    }


def run_full_preprocessing_from_file(
    file_path: str | Path,
    **kwargs,
) -> dict:
    """
    Load a file and run the complete Stage I preprocessing pipeline.

    Parameters
    ----------
    file_path : str or Path
        Path to the raw EEG data file.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`run_full_preprocessing`.

    Returns
    -------
    result : dict
        Same structure as :func:`run_full_preprocessing`.
    """
    raw = load_raw_eeg(file_path)
    return run_full_preprocessing(raw, **kwargs)

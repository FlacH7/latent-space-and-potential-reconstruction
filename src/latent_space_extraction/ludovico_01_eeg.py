"""
ludovico_01_eeg.py
=================
Loader for the **Ludovico_01** dataset.

Each CSV file contains a multivariate time series where:
  - Columns = channels (e.g. 59 columns)
  - Rows    = observations / time samples (e.g. 10 000 rows)

The data does **not** come from EEG, so no EEG-specific preprocessing
(filtering, ICA, ICLabel) is applied.  The CSV is read as-is, centered
to zero mean per channel, and wrapped in an ``mne.io.RawArray`` so that
it is compatible with the existing 3-stage latent-space pipeline.

Sampling frequency is assumed to be 1000 Hz (1 ms per sample), giving a
total recording duration of 10 s for 10 000 samples.

Usage::

    from src.latent_space_extraction.ludovico_01_eeg import load_ludovico_01_from_ids

    raw = load_ludovico_01_from_ids(
        subject="data_01_23_18",
        db_path="/path/to/Ludovico_01",
        t_start=0.0,
        t_stop=10.0,
        verbose=True,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

logger = logging.getLogger("ludovico_01_eeg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Assumed sampling frequency: 1 ms per sample → 1000 Hz
DEFAULT_SFREQ: float = 1000.0

# Channel type used for all columns (generic "data" — not EEG-specific)
MNE_CHANNEL_TYPE: str = "misc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_csv_path(subject: str, db_path: str | Path) -> Path:
    """Build the CSV path from the subject name.

    The subject name is expected to be the CSV filename *without* extension,
    e.g. ``"data_01_23_18"`` → ``<db_path>/data_01_23_18.csv``.
    """
    db_path = Path(db_path)
    csv_file = db_path / f"{subject}.csv"
    return csv_file


def _sanitize_column_names(col_names: list[str]) -> list[str]:
    """Replace numeric-looking column names with 'Ch 0', 'Ch 1', ..."""
    if not col_names:
        return col_names
    numeric_count = 0
    for name in col_names:
        try:
            float(name)
            numeric_count += 1
        except (ValueError, TypeError):
            pass
    if numeric_count == len(col_names):
        return [f"Ch {i}" for i in range(len(col_names))]
    return col_names


def _read_csv_as_matrix(csv_path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Read a CSV file and return the data matrix and column names.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.  Columns = channels, Rows = time samples.

    Returns
    -------
    data : np.ndarray, shape (n_samples, n_channels)
        The time-series data transposed to (samples, channels) for MNE.
    col_names : list[str]
        Column names from the CSV header.  If all column names look
        like numeric values (no real header), they are replaced with
        ``"Ch 0"``, ``"Ch 1"``, etc.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    col_names = list(df.columns)
    col_names = _sanitize_column_names(col_names)
    # MNE expects (n_channels, n_times), so transpose
    data = df.values.T  # shape: (n_channels, n_samples)

    return data, col_names


def _build_raw_array(
    data: np.ndarray,
    ch_names: list[str],
    sfreq: float = DEFAULT_SFREQ,
) -> mne.io.RawArray:
    """Wrap a (n_channels, n_times) matrix into an MNE RawArray.

    Each channel is centered to zero mean.  Channel type is set to
    ``MNE_CHANNEL_TYPE`` ("misc") since this is not real EEG.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_times)
    ch_names : list[str]
    sfreq : float

    Returns
    -------
    raw : mne.io.RawArray
    """
    # Center each channel
    data = data - data.mean(axis=1, keepdims=True)

    n_channels = data.shape[0]
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=[MNE_CHANNEL_TYPE] * n_channels,
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------


def load_ludovico_01_csv(
    csv_file: str | Path,
    *,
    sfreq: float = DEFAULT_SFREQ,
    t_start: float | None = None,
    t_stop: float | None = None,
    preload: bool = True,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """Load a single CSV file from the Ludovico_01 dataset.

    Parameters
    ----------
    csv_file : str or Path
        Path to the CSV file.
    sfreq : float, default 1000.0
        Sampling frequency in Hz.
    t_start, t_stop : float | None
        Optional time window in seconds to crop the recording.
    preload : bool, default True
        Whether to preload data into memory.
    verbose : bool | str | None
        Verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        RawArray with the CSV data.
    """
    csv_file = Path(csv_file)
    data, ch_names = _read_csv_as_matrix(csv_file)
    raw = _build_raw_array(data, ch_names, sfreq=sfreq)

    if verbose:
        n_ch = len(ch_names)
        n_samples = raw.n_times
        duration = n_samples / sfreq
        print(f"  [Ludovico01] Loaded: {csv_file.name}")
        print(f"  [Ludovico01] Channels  : {n_ch}")
        print(f"  [Ludovico01] Samples   : {n_samples}")
        print(f"  [Ludovico01] Duration  : {duration:.2f} s (sfreq={sfreq:.0f} Hz)")

    # Crop to time window if requested
    if t_start is not None or t_stop is not None:
        start = t_start if t_start is not None else 0.0
        stop = t_stop if t_stop is not None else raw.times[-1]
        start = max(0.0, start)
        stop = min(stop, raw.times[-1])

        if start >= stop:
            raise ValueError(
                f"Invalid crop window: t_start={start:.1f}s >= t_stop={stop:.1f}s"
            )

        raw.crop(tmin=start, tmax=stop)

        if verbose:
            print(f"  [Ludovico01] Cropped to {start:.1f}s - {stop:.1f}s")

    return raw


# ---------------------------------------------------------------------------
# High-level loader from IDs
# ---------------------------------------------------------------------------


def load_ludovico_01_from_ids(
    subject: str,
    *,
    db_path: str | Path | None = None,
    sfreq: float = DEFAULT_SFREQ,
    t_start: float | None = None,
    t_stop: float | None = None,
    preload: bool = True,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """Load a CSV from Ludovico_01 by subject name.

    Parameters
    ----------
    subject : str
        Subject/filename identifier, e.g. ``"data_01_23_18"``.
        The ``.csv`` extension is appended automatically.
    db_path : str | Path | None
        Root directory of the Ludovico_01 dataset.
        If ``None``, reads from config ``DB_LUDOVICO_01_PATH``.
    sfreq : float, default 1000.0
        Sampling frequency in Hz.
    t_start, t_stop : float | None
        Time window in seconds.
    preload, verbose
        Forwarded to :func:`load_ludovico_01_csv`.

    Returns
    -------
    raw : mne.io.Raw
    """
    if db_path is None:
        try:
            from src.utils.config import DB_LUDOVICO_01_PATH as _cfg_path
            db_path = _cfg_path
        except (ImportError, AttributeError):
            raise FileNotFoundError(
                "No db_path provided and DB_LUDOVICO_01_PATH not found in config."
            )

    csv_file = _resolve_csv_path(subject, db_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file}\n"
            f"  (subject={subject}, db_path={db_path})"
        )

    if verbose:
        print(f"  [Ludovico01] Resolved: {csv_file}")

    return load_ludovico_01_csv(
        csv_file,
        sfreq=sfreq,
        t_start=t_start,
        t_stop=t_stop,
        preload=preload,
        verbose=verbose,
    )


def list_ludovico_01_subjects(db_path: str | Path) -> list[str]:
    """List all available CSV subjects in the Ludovico_01 directory.

    Excludes ``structural.csv`` and any non-CSV files.

    Parameters
    ----------
    db_path : str or Path
        Root directory of the Ludovico_01 dataset.

    Returns
    -------
    subjects : list[str]
        Sorted list of subject names (filename stems).
    """
    db_path = Path(db_path)
    if not db_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {db_path}")

    subjects = []
    for f in sorted(db_path.iterdir()):
        if f.suffix.lower() == ".csv" and f.stem != "structural":
            subjects.append(f.stem)

    return subjects

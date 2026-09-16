"""
anphy_eeg.py
============
Loader and preprocessor for the **ANPHY-Sleep** dataset.

Mirrors the API of ``siena_eeg.py`` so that
``test_iga_from_eeg_latent_anphy.py`` can offload all raw-loading
logic to a single call.

Typical workflow::

    raw = load_anphy_eeg(
        "/path/to/EPCTL01.edf",
        t_start=450.0,
        t_stop=480.0,
        preload=False,
        verbose=True,
    )
"""

from __future__ import annotations

from pathlib import Path

import mne


# ---------------------------------------------------------------------------
# Channel whitelist (same montage used by the ANPHY pipeline)
# ---------------------------------------------------------------------------

_ANPHY_VALID_EEG_CHANNELS: frozenset[str] = frozenset({
     "FP1", "F3", "C3", "P3", "O1", "F7", "T3", "T5",
     "FC1","FC5","CP1","CP5","F9",
     "FZ", "CZ", "PZ",
     "FP2", "F4", "C4", "P4","O2", "F8", "T4", "T6",
     "FC2","FC6","CP2","CP6","F10"
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_channel_name(ch_name: str) -> str:
    """
    Strip type prefix (EEG, EOG, EKG, EMG, etc.) and return the bare
    channel name in upper-case.
    """
    stripped = ch_name.strip()
    for prefix in ("EEG ", "EOG ", "EKG ", "EMG ", "ECG "):
        if stripped.upper().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped.strip().upper()


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------

def load_anphy_eeg(
    edf_file: str | Path,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    t_minutes: tuple[float, float] | None = None,
    first_n_minutes: float | None = None,
    preload: bool = False,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Load an ANPHY-Sleep ``.edf`` file, keep only valid EEG/EOG channels,
    rename them to standard 10-20 names, set the montage, and optionally
    crop to a time window.

    Time selection is resolved in this priority order:

    1. ``t_minutes`` – ``(start_min, stop_min)``.
    2. ``first_n_minutes`` – keep only the first *N* minutes.
    3. ``t_start`` / ``t_stop`` – absolute seconds.

    Parameters
    ----------
    edf_file : str or Path
        Path to the ``.edf`` file.
    t_start : float | None, optional
        Start time in seconds.
    t_stop : float | None, optional
        Stop time in seconds.
    t_minutes : tuple[float, float] | None, optional
        ``(start_min, stop_min)`` convenience shortcut.
    first_n_minutes : float | None, optional
        If given, ``t_start`` is set to 0 and ``t_stop`` to ``N*60``.
    preload : bool, default False
        Passed to MNE.  When ``False``, the file is *not* fully loaded
        into RAM until after cropping and channel selection, which keeps
        memory usage low.
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        Cropped raw object containing only valid EEG/EOG channels with
        standard 10-20 names and montage set.

    Raises
    ------
    FileNotFoundError
        If ``edf_file`` does not exist.
    ValueError
        If no valid EEG channels are found or the crop window is invalid.
    """
    edf_file = Path(edf_file)
    if not edf_file.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_file}")

    # ------------------------------------------------------------------
    # Resolve time window (minutes take precedence)
    # ------------------------------------------------------------------
    _t_start: float | None = t_start
    _t_stop: float | None = t_stop

    if t_minutes is not None:
        if len(t_minutes) != 2:
            raise ValueError("t_minutes must be a tuple of (start, stop)")
        _t_start, _t_stop = t_minutes[0] * 60.0, t_minutes[1] * 60.0
    elif first_n_minutes is not None:
        _t_start, _t_stop = 0.0, first_n_minutes * 60.0

    # ------------------------------------------------------------------
    # Load raw EDF (preload=False keeps memory low until we crop)
    # ------------------------------------------------------------------
    raw = mne.io.read_raw_edf(edf_file, preload=preload, verbose=verbose)

    # ------------------------------------------------------------------
    # Crop to requested time window
    # ------------------------------------------------------------------
    total_dur = raw.times[-1]

    if _t_start is not None or _t_stop is not None:
        start = _t_start if _t_start is not None else 0.0
        stop = _t_stop if _t_stop is not None else total_dur

        start = max(0.0, start)
        stop = min(stop, total_dur)

        if start >= stop:
            raise ValueError(
                f"Invalid crop window: t_start={start:.1f}s >= t_stop={stop:.1f}s "
                f"(recording duration: {total_dur:.1f}s)"
            )

        raw.crop(tmin=start, tmax=stop)

        if verbose:
            print(f"  [ANPHY] Cropped to {start:.1f}s - {stop:.1f}s "
                  f"({(stop - start) / 60:.1f} min of "
                  f"{total_dur / 60:.1f} min total)")
    else:
        if verbose:
            print(f"  [ANPHY] Full recording loaded: {total_dur / 60:.1f} min")

    # ------------------------------------------------------------------
    # Select valid channels
    # ------------------------------------------------------------------
    valid_chs = [
        ch for ch in raw.ch_names
        if _clean_channel_name(ch) in _ANPHY_VALID_EEG_CHANNELS
    ]

    if not valid_chs:
        all_chs = ", ".join(raw.ch_names)
        raise ValueError(
            f"No valid EEG channels found in {edf_file.name}. "
            f"Channels present: {all_chs}"
        )

    n_dropped = len(raw.ch_names) - len(valid_chs)
    raw.pick(valid_chs)

    if verbose:
        print(f"  [ANPHY] Kept {len(valid_chs)} EEG/EOG channels, "
              f"dropped {n_dropped} non-EEG channels.")

    # ------------------------------------------------------------------
    # Rename channels to standard 10-20 names and set montage
    # ------------------------------------------------------------------
    rename_map = {}
    for ch in raw.ch_names:
        stripped = ch.strip()
        if stripped.upper().startswith("EEG "):
            rename_map[ch] = stripped[4:].strip()
    if rename_map:
        raw.rename_channels(rename_map)

    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="warn")
        if verbose:
            print(f"  [ANPHY] Montage 'standard_1020' set successfully.")
    except Exception as exc:
        if verbose:
            print(f"  [ANPHY] Warning: could not set montage: {exc}")

    # Force load only the cropped / picked segment into RAM
    raw.load_data()
    return raw
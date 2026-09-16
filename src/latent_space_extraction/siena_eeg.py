"""
siena_eeg.py
============
Loader and annotation parser for the **Siena Scalp EEG** dataset
(PhysioNet).  The module provides three levels of convenience:

1. **Low-level** – ``load_siena_eeg()`` loads a single ``.edf`` file,
   keeps only valid EEG/EOG channels, and optionally crops to a time
   window.
2. **Metadata** – ``parse_siena_seizure_list()`` parses the
   ``Seizures-list-PNxx.txt`` annotation file shipped with each
   patient folder.
3. **High-level** – ``load_siena_with_info()`` combines the two above:
   point it to a patient directory and it returns the ``mne.Raw``
   object together with structured seizure metadata.

Typical workflow::

    # Load first hour of PN09's first recording
    raw, info = load_siena_with_info(
        "./PN09",
        record_index=0,
        t_minutes=(0, 60),
        verbose=True,
    )

    # info now contains:
    #   info["patient_id"]  -> "PN09"
    #   info["sfreq"]       -> 512
    #   info["seizures"]    -> [(6054.0, 6134.0)]  # (start, end) in seconds
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mne


# ---------------------------------------------------------------------------
# Channel whitelist
# ---------------------------------------------------------------------------

# Valid 10-20 / 10-10 channel names expected in Siena EDF files.
# Any channel whose uppercase name is not in this set is dropped (EKG,
# placeholders like "1", aux channels, etc.).
_SIENA_VALID_EEG_CHANNELS: frozenset[str] = frozenset({
    # --- 10-20 core ---
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ",
    # --- 10-10 extensions ---
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    "FT7", "FT8", "TP7", "TP8", "PO7", "PO8",
    "AF3", "AF4", "AF7", "AF8",
    "F1", "F2", "F5", "F6",
    "C1", "C2", "C5", "C6",
    "P1", "P2", "P5", "P6",
    "PO3", "PO4", "POZ", "OZ",
    "FPZ", "AFZ", "CPZ",
    "F9", "F10", "T9", "T10", "P9", "P10",
    # --- EOG channels (kept for ICA) ---
    "VEOG", "HEOG", "EOG",
    # --- Reference / misc (sometimes present) ---
    "A1", "A2", "M1", "M2", "M3", "M4", "NAS", "REF",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_channel_name(ch_name: str) -> str:
    """
    Strip type prefix (EEG, EOG, EKG, EMG, etc.) and return the bare
    channel name in upper-case.

    Examples
    --------
    >>> _clean_channel_name("EEG Fp1")
    'FP1'
    >>> _clean_channel_name("EKG EKG")
    'EKG'
    >>> _clean_channel_name("Fp1")
    'FP1'
    """
    stripped = ch_name.strip()
    # Remove common type prefixes
    for prefix in ("EEG ", "EOG ", "EKG ", "EMG ", "ECG "):
        if stripped.upper().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped.strip().upper()


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------

def load_siena_eeg(
    edf_file: str | Path,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    t_minutes: tuple[float, float] | None = None,
    first_n_minutes: float | None = None,
    preload: bool = True,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Load a Siena scalp-EEG ``.edf`` file, keep only valid EEG channels,
    and optionally crop to a time window.

    Siena files contain ~29 EEG channels (10-20 / 10-10) plus EKG and
    occasionally invalid / placeholder channels (e.g. a channel named
    ``"1"``).  This loader drops every channel whose uppercase name is
    *not* in the standard EEG/EOG montage, so only neural data are
    retained.

    Time selection can be performed in three mutually exclusive ways
    (checked in the order listed):

    * ``t_minutes``   – start / stop expressed in **minutes** from the
      beginning of the recording (e.g. ``(0, 60)`` = first hour).
    * ``first_n_minutes`` – keep only the first *N* minutes.
    * ``t_start`` / ``t_stop`` – absolute seconds (MNE convention).

    Parameters
    ----------
    edf_file : str or Path
        Path to the ``.edf`` file (e.g. ``"PN09-1.edf"``).
    t_start : float | None, optional
        Start time in seconds.  Only used when none of the
        higher-priority options are given.
    t_stop : float | None, optional
        Stop time in seconds.
    t_minutes : tuple[float, float] | None, optional
        ``(start_min, stop_min)`` – convenience shortcut.  ``(0, 60)``
        extracts the first hour.
    first_n_minutes : float | None, optional
        If given, ``t_start`` is set to 0 and ``t_stop`` to
        ``N * 60`` seconds.
    preload : bool, default True
        If True, data are loaded into RAM.
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        Raw object cropped to the requested window and containing only
        valid EEG/EOG channels.

    Raises
    ------
    FileNotFoundError
        If ``edf_file`` does not exist.
    ValueError
        If no valid EEG channels are found after filtering.

    Examples
    --------
    >>> # Full recording, only valid EEG channels
    >>> raw = load_siena_eeg("PN09/PN09-1.edf")

    >>> # First hour (minutes 0 to 60)
    >>> raw = load_siena_eeg("PN09/PN09-1.edf", t_minutes=(0, 60))

    >>> # From minute 30 to minute 90
    >>> raw = load_siena_eeg("PN09/PN09-1.edf", t_minutes=(30, 90))

    >>> # Only the first 20 minutes
    >>> raw = load_siena_eeg("PN09/PN09-1.edf", first_n_minutes=20)

    >>> # Explicit seconds (MNE style)
    >>> raw = load_siena_eeg("PN09/PN09-1.edf", t_start=0, t_stop=3600)
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
    # Load raw EDF and keep only standard EEG/EOG channels
    # ------------------------------------------------------------------
    raw = mne.io.read_raw_edf(edf_file, preload=preload, verbose=verbose)
    
    # ------------------------------------------------------------------
    # Crop to requested time window
    # ------------------------------------------------------------------
    total_dur = raw.times[-1]

    if _t_start is not None or _t_stop is not None:
        start = _t_start if _t_start is not None else 0.0
        stop = _t_stop if _t_stop is not None else total_dur

        # Clamp to actual recording bounds
        start = max(0.0, start)
        stop = min(stop, total_dur)

        if start >= stop:
            raise ValueError(
                f"Invalid crop window: t_start={start:.1f}s >= t_stop={stop:.1f}s "
                f"(recording duration: {total_dur:.1f}s)"
            )

        raw.crop(tmin=start, tmax=stop)

        if verbose:
            print(f"  [Siena] Cropped to {start:.1f}s - {stop:.1f}s "
                  f"({(stop - start) / 60:.1f} min of "
                  f"{total_dur / 60:.1f} min total)")
    else:
        if verbose:
            print(f"  [Siena] Full recording loaded: {total_dur / 60:.1f} min")

    # ------------------------------------------------------------------
    # Select valid Channels
    # ------------------------------------------------------------------
    valid_chs = [
        ch for ch in raw.ch_names
        if _clean_channel_name(ch) in _SIENA_VALID_EEG_CHANNELS
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
        print(f"  [Siena] Kept {len(valid_chs)} EEG/EOG channels, "
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
            print(f"  [Siena] Montage 'standard_1020' set successfully.")
    except Exception as exc:
        if verbose:
            print(f"  [Siena] Warning: could not set montage: {exc}")
    raw.load_data()
    return raw


# ---------------------------------------------------------------------------
# Seizure-list parser
# ---------------------------------------------------------------------------

def parse_siena_seizure_list(txt_file: str | Path) -> dict:
    """
    Parse a ``Seizures-list-PNxx.txt`` annotation file from the Siena
    dataset and return structured metadata.

    Parameters
    ----------
    txt_file : str or Path
        Path to the ``Seizures-list-PNxx.txt`` file.

    Returns
    -------
    info : dict
        Dictionary with keys:

        * ``'patient_id'``      – e.g. ``"PN09"``
        * ``'sfreq'``           – sampling frequency in Hz (int)
        * ``'channels'``        – ordered list of valid channel names
        * ``'records'``         – list of record dicts, each with:
          ``'file'``, ``'reg_start'``, ``'reg_end'``, ``'seizures'``
          where ``'seizures'`` is a list of ``(start_sec, end_sec)``
          tuples relative to the record start.

    Examples
    --------
    >>> info = parse_siena_seizure_list("PN09/Seizures-list-PN09.txt")
    >>> print(info["records"][0]["seizures"])
    [(6054.0, 6134.0)]
    """
    txt_file = Path(txt_file)
    if not txt_file.exists():
        raise FileNotFoundError(f"Seizure list file not found: {txt_file}")

    lines = txt_file.read_text(encoding="utf-8").splitlines()

    patient_id = txt_file.stem.replace("Seizures-list-", "")

    sfreq: int | None = None
    channels: list[str] = []
    in_channel_section = False
    records: list[dict] = []
    current_record: dict | None = None
    current_seizures: list[list[float | None]] = []

    def _parse_hms(val: str) -> float:
        parts = val.strip().split(".")
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        return h * 3600.0 + m * 60.0 + s

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if "Data Sampling Rate:" in stripped:
            sfreq = int(stripped.split(":")[-1].strip().split()[0])
            continue

        if "Channels in EDF files:" in stripped:
            in_channel_section = True
            continue

        if in_channel_section and stripped.lower().startswith("seizure"):
            in_channel_section = False
            continue

        if in_channel_section and ":" in stripped:
            ch_name = stripped.split(":", 1)[-1].strip()
            if ch_name:
                channels.append(ch_name)
            continue

        if stripped.lower().startswith("seizure n"):
            if current_record is not None:
                current_record["seizures"] = current_seizures
                records.append(current_record)
            current_seizures = []
            current_record = {
                "file": None,
                "reg_start": None,
                "reg_end": None,
                "seizures": [],
            }
            continue

        if current_record is None:
            continue

        if stripped.startswith("File name:"):
            current_record["file"] = stripped.split(":", 1)[-1].strip()
        elif stripped.startswith("Registration start time:"):
            current_record["reg_start"] = stripped.split(":", 1)[-1].strip()
        elif stripped.startswith("Registration end time:"):
            current_record["reg_end"] = stripped.split(":", 1)[-1].strip()
        elif stripped.startswith("Seizure start time:"):
            val = stripped.split(":", 1)[-1].strip()
            try:
                t0_sec = _parse_hms(current_record["reg_start"])
                t1_sec = _parse_hms(val)
                seizure_start = t1_sec - t0_sec
            except Exception:
                seizure_start = None
            current_seizures.append([seizure_start, None])
        elif stripped.startswith("Seizure end time:"):
            val = stripped.split(":", 1)[-1].strip()
            try:
                t0_sec = _parse_hms(current_record["reg_start"])
                t1_sec = _parse_hms(val)
                seizure_end = t1_sec - t0_sec
            except Exception:
                seizure_end = None
            if current_seizures and current_seizures[-1][1] is None:
                current_seizures[-1][1] = seizure_end

    if current_record is not None:
        current_record["seizures"] = current_seizures
        records.append(current_record)

    for rec in records:
        rec["seizures"] = [
            (s, e) for s, e in rec["seizures"]
            if s is not None and e is not None
        ]

    return {
        "patient_id": patient_id,
        "sfreq": sfreq,
        "channels": channels,
        "records": records,
    }


# ---------------------------------------------------------------------------
# High-level convenience loader
# ---------------------------------------------------------------------------

def load_siena_with_info(
    patient_dir: str | Path,
    record_index: int = 0,
    *,
    t_minutes: tuple[float, float] | None = None,
    first_n_minutes: float | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    preload: bool = True,
    verbose: bool | str | None = None,
) -> tuple[mne.io.Raw, dict]:
    """
    Convenience wrapper that parses the ``Seizures-list-PNxx.txt`` file
    inside *patient_dir* and loads the requested EDF record.

    Parameters
    ----------
    patient_dir : str or Path
        Path to the patient folder (e.g. ``"./PN09/"``).
    record_index : int, default 0
        Which EDF record to load (0-based, matching the order in the
        seizure-list file).
    t_minutes, first_n_minutes, t_start, t_stop
        Passed through to :func:`load_siena_eeg`.
    preload : bool, default True
        Passed to MNE.
    verbose : bool | str | None, optional
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        Cropped raw object with only EEG channels.
    info : dict
        Parsed seizure-list metadata for the loaded record (includes
        ``'seizures'`` as ``(start_sec, end_sec)`` tuples relative to
        the beginning of the full EDF).

    Examples
    --------
    >>> raw, info = load_siena_with_info("./PN09", record_index=0,
    ...                                  t_minutes=(0, 60))
    >>> print(info["seizures"])
    [(6054.0, 6134.0)]
    >>> print(info["sfreq"])
    512
    """
    patient_dir = Path(patient_dir)
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    txt_files = list(patient_dir.glob("Seizures-list-*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No Seizures-list-PNxx.txt file found in {patient_dir}"
        )
    txt_file = txt_files[0]

    meta = parse_siena_seizure_list(txt_file)

    if record_index >= len(meta["records"]):
        raise ValueError(
            f"record_index={record_index} out of range "
            f"(only {len(meta['records'])} records available)"
        )

    record = meta["records"][record_index]
    edf_path = patient_dir / record["file"]

    if not edf_path.exists():
        raise FileNotFoundError(
            f"EDF file referenced in seizure list not found: {edf_path}"
        )

    raw = load_siena_eeg(
        edf_path,
        t_minutes=t_minutes,
        first_n_minutes=first_n_minutes,
        t_start=t_start,
        t_stop=t_stop,
        preload=preload,
        verbose=verbose,
    )

    info = {
        "patient_id": meta["patient_id"],
        "sfreq": meta["sfreq"],
        "channels": meta["channels"],
        "file": record["file"],
        "reg_start": record["reg_start"],
        "reg_end": record["reg_end"],
        "seizures": record["seizures"],
    }

    return raw, info


# ---------------------------------------------------------------------------
# Utility helpers for segment selection (interictal / preictal / ictal)
# ---------------------------------------------------------------------------

def get_interictal_intervals(
    info: dict,
    min_gap_sec: float = 300.0,
    exclude_first_sec: float = 0.0,
) -> list[tuple[float, float]]:
    """
    Return time intervals that are guaranteed *interictal* (far from any
    seizure).

    Parameters
    ----------
    info : dict
        Output of :func:`load_siena_with_info` (or the per-record dict
        from :func:`parse_siena_seizure_list`).
    min_gap_sec : float, default 300.0
        Minimum gap (in seconds) to keep before and after each seizure.
        Default is 5 minutes.
    exclude_first_sec : float, default 0.0
        Skip the first N seconds of the recording (e.g. calibration
        artefacts).

    Returns
    -------
    intervals : list[tuple[float, float]]
        List of ``(start_sec, end_sec)`` intervals safe for interictal
        analysis.

    Examples
    --------
    >>> info = parse_siena_seizure_list("PN09/Seizures-list-PN09.txt")
    >>> info["seizures"] = info["records"][0]["seizures"]
    >>> intervals = get_interictal_intervals(info)
    >>> print(intervals)
    [(0.0, 5754.0)]   # everything up to 5 min before seizure at 6054s
    """
    seizures = info.get("seizures", [])
    total_end = _estimate_total_duration(info)

    # Build exclusion zones around each seizure
    exclusion = []
    for s_start, s_end in seizures:
        ex_start = max(0.0, s_start - min_gap_sec)
        ex_end = min(total_end, s_end + min_gap_sec)
        exclusion.append((ex_start, ex_end))

    # Sort and merge overlapping exclusions
    exclusion.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    for ex in exclusion:
        if not merged or ex[0] > merged[-1][1]:
            merged.append(ex)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], ex[1]))

    # Compute interictal intervals (gaps between exclusions)
    intervals: list[tuple[float, float]] = []
    cursor = exclude_first_sec

    for ex_start, ex_end in merged:
        if cursor < ex_start:
            intervals.append((cursor, ex_start))
        cursor = max(cursor, ex_end)

    if cursor < total_end:
        intervals.append((cursor, total_end))

    return intervals


def get_preictal_interval(
    info: dict,
    preictal_sec: float = 600.0,
    min_gap_to_seizure: float = 0.0,
) -> tuple[float, float] | None:
    """
    Return the pre-ictal interval immediately preceding the first seizure.

    Parameters
    ----------
    info : dict
        Output of :func:`load_siena_with_info`.
    preictal_sec : float, default 600.0
        Length of the preictal window in seconds (default 10 min).
    min_gap_to_seizure : float, default 0.0
        If > 0, the interval ends this many seconds *before* the seizure
        onset (avoids including ictal activity).

    Returns
    -------
    interval : tuple[float, float] | None
        ``(start_sec, end_sec)`` or None if no seizure is annotated.
    """
    seizures = info.get("seizures", [])
    if not seizures:
        return None

    s_start = seizures[0][0]
    end = s_start - min_gap_to_seizure
    start = max(0.0, end - preictal_sec)

    if start >= end:
        return None
    return (start, end)


def _estimate_total_duration(info: dict) -> float:
    """
    Estimate total recording duration in seconds from registration times.
    Falls back to a large value if parsing fails.
    """
    reg_start = info.get("reg_start")
    reg_end = info.get("reg_end")

    if reg_start and reg_end:
        try:
            def _parse_hms(val: str) -> float:
                parts = val.strip().split(".")
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                return h * 3600.0 + m * 60.0 + s
            return _parse_hms(reg_end) - _parse_hms(reg_start)
        except Exception:
            pass

    # Fallback: try to get from MNE Raw if cached (not available here,
    # but callers can override)
    return float("inf")

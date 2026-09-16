"""
test_retest_gedai_eeg.py
========================
Loader and preprocessor for the **test-retest resting and cognitive state
EEG dataset preprocessed with Gedai** (EEGLAB .set/.fdt format).

Mirrors the API of ``test_retest_eeg.py`` so that the IgA pipeline can
offload all raw-loading logic to a single call, but reads EEGLAB files
(.set + .fdt) instead of BrainVision (.vhdr + .eeg + .vmrk).

Typical workflow::

    # Low-level: direct path to .set
    raw = load_test_retest_gedai_eeg(
        "/path/to/sub-01_ses-session1_task-eyesclosed_eeg_01-40_Gedai.set",
        t_start=0.0,
        t_stop=60.0,
        preload=False,
        verbose=True,
    )

    # High-level: build path from IDs
    raw = load_test_retest_gedai_eeg_from_ids(
        subject="sub-01",
        session="session1",
        task="eyesclosed",
        db_path="/path/to/preprocessed_01_40_Gedai",
        t_start=0.0,
        t_stop=60.0,
        verbose=True,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne

logger = logging.getLogger("test_retest_gedai_eeg")

# ---------------------------------------------------------------------------
# Channel whitelist (10-20 / 10-10 standard EEG channels)
# ---------------------------------------------------------------------------

_TEST_RETEST_GEDAI_VALID_EEG_CHANNELS: frozenset[str] = frozenset({
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
    # --- Reference / misc ---
    "A1", "A2", "M1", "M2", "NAS", "REF",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_channel_name(ch_name: str) -> str:
    """Strip type prefix and return bare channel name in upper-case."""
    stripped = ch_name.strip()
    for prefix in ("EEG ", "EOG ", "EKG ", "EMG ", "ECG "):
        if stripped.upper().startswith(prefix):
            stripped = stripped[len(prefix):]
            break
    return stripped.strip().upper()


def _resolve_set_path(subject: str, session: str, task: str, db_path: str | Path) -> Path:
    """Build the .set path from BIDS components for Gedai-preprocessed data."""
    db_path = Path(db_path)
    # Normalize subject/session naming
    subj_dir = subject if subject.startswith("sub-") else f"sub-{subject}"
    ses_dir = session if session.startswith("ses-") else f"ses-{session}"

    set_file = (
        db_path
        / subj_dir
        / ses_dir
        / "eeg"
        / f"{subj_dir}_{ses_dir}_task-{task}_eeg_01-40_Gedai.set"
    )
    return set_file


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------

def load_test_retest_gedai_eeg(
    set_file: str | Path,
    *,
    t_start: float | None = None,
    t_stop: float | None = None,
    t_minutes: tuple[float, float] | None = None,
    first_n_minutes: float | None = None,
    preload: bool = False,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Load an EEGLAB ``.set`` file from the Gedai-preprocessed test-retest
    dataset, keep only valid EEG/EOG channels, rename them to standard 10-20
    names, set the montage, and optionally crop to a time window.

    Time selection priority:

    1. ``t_minutes`` – ``(start_min, stop_min)``.
    2. ``first_n_minutes`` – keep only the first *N* minutes.
    3. ``t_start`` / ``t_stop`` – absolute seconds.

    Parameters
    ----------
    set_file : str or Path
        Path to the ``.set`` file (MNE resolves the companion ``.fdt``
        automatically from it).
    t_start, t_stop : float | None
        Absolute seconds (MNE convention). Lowest priority.
    t_minutes : tuple[float, float] | None
        ``(start_min, stop_min)`` convenience shortcut.
    first_n_minutes : float | None
        If given, ``t_start`` is set to 0 and ``t_stop`` to ``N*60``.
    preload : bool, default False
        Passed to MNE.  When ``False``, memory stays low until after crop.
    verbose : bool | str | None
        MNE verbosity level.

    Returns
    -------
    raw : mne.io.Raw
        Cropped raw object with only valid EEG channels and montage set.

    Raises
    ------
    FileNotFoundError
        If ``set_file`` or the companion ``.fdt`` do not exist.
    ValueError
        If no valid EEG channels are found or the crop window is invalid.
    """
    set_file = Path(set_file)
    if not set_file.exists():
        raise FileNotFoundError(f"EEGLAB .set file not found: {set_file}")

    # Resolve companion .fdt file
    fdt_file = set_file.with_suffix(".fdt")
    if not fdt_file.exists():
        raise FileNotFoundError(f"EEGLAB .fdt data file not found: {fdt_file}")

    # ------------------------------------------------------------------
    # Resolve time window
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
    # Load EEGLAB .set/.fdt (preload=False keeps memory low)
    # ------------------------------------------------------------------
    raw = mne.io.read_raw_eeglab(str(set_file), preload=preload, verbose=verbose)

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
            print(f"  [TestRetestGedai] Cropped to {start:.1f}s - {stop:.1f}s "
                  f"({(stop - start) / 60:.1f} min of "
                  f"{total_dur / 60:.1f} min total)")
    else:
        if verbose:
            print(f"  [TestRetestGedai] Full recording loaded: {total_dur / 60:.1f} min")

    # ------------------------------------------------------------------
    # Select valid channels
    # ------------------------------------------------------------------
    valid_chs = [
        ch for ch in raw.ch_names
        if _clean_channel_name(ch) in _TEST_RETEST_GEDAI_VALID_EEG_CHANNELS
    ]

    if not valid_chs:
        all_chs = ", ".join(raw.ch_names)
        raise ValueError(
            f"No valid EEG channels found in {set_file.name}. "
            f"Channels present: {all_chs}"
        )

    n_dropped = len(raw.ch_names) - len(valid_chs)
    raw.pick(valid_chs)

    if verbose:
        print(f"  [TestRetestGedai] Kept {len(valid_chs)} EEG/EOG channels, "
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
            print(f"  [TestRetestGedai] Montage 'standard_1020' set successfully.")
    except Exception as exc:
        if verbose:
            print(f"  [TestRetestGedai] Warning: could not set montage: {exc}")

    raw.load_data()
    return raw


# ---------------------------------------------------------------------------
# High-level loader from IDs
# ---------------------------------------------------------------------------

def load_test_retest_gedai_eeg_from_ids(
    subject: str,
    session: str,
    task: str,
    *,
    db_path: str | Path | None = None,
    t_start: float | None = None,
    t_stop: float | None = None,
    t_minutes: tuple[float, float] | None = None,
    first_n_minutes: float | None = None,
    preload: bool = False,
    verbose: bool | str | None = None,
) -> mne.io.Raw:
    """
    Convenience wrapper: builds the BIDS path from *subject*, *session*,
    and *task* identifiers, then calls :func:`load_test_retest_gedai_eeg`.

    Parameters
    ----------
    subject : str
        Subject ID, e.g. ``"sub-01"`` or ``"01"``.
    session : str
        Session ID, e.g. ``"session1"`` or ``"ses-session1"``.
    task : str
        Task name, e.g. ``"eyesclosed"``, ``"eyesopen"``, ``"mathematic"``,
        ``"memory"``, ``"music"``.
    db_path : str | Path | None
        Root of the preprocessed Gedai dataset.  If ``None``, reads from the
        environment variable ``DB_TEST_RETEST_GEDAI_PATH`` or falls back to
        ``src.utils.config.DB_TEST_RETEST_GEDAI_PATH``.
    t_start, t_stop, t_minutes, first_n_minutes, preload, verbose
        Forwarded to :func:`load_test_retest_gedai_eeg`.

    Returns
    -------
    raw : mne.io.Raw

    Raises
    ------
    FileNotFoundError
        If the resolved ``.set`` does not exist.
    """
    # Resolve db_path
    if db_path is None:
        import os
        db_path = os.environ.get("DB_TEST_RETEST_GEDAI_PATH", "")
        if not db_path:
            try:
                from src.utils.config import DB_TEST_RETEST_GEDAI_PATH as _cfg_path
                db_path = _cfg_path
            except ImportError:
                raise FileNotFoundError(
                    "No db_path provided and DB_TEST_RETEST_GEDAI_PATH not set "
                    "in environment or config."
                )

    set_file = _resolve_set_path(subject, session, task, db_path)
    if not set_file.exists():
        raise FileNotFoundError(
            f"Expected EEGLAB file not found: {set_file}\n"
            f"  (subject={subject}, session={session}, task={task}, "
            f"db_path={db_path})"
        )

    if verbose:
        print(f"  [TestRetestGedai] Resolved: {set_file}")

    return load_test_retest_gedai_eeg(
        set_file,
        t_start=t_start,
        t_stop=t_stop,
        t_minutes=t_minutes,
        first_n_minutes=first_n_minutes,
        preload=preload,
        verbose=verbose,
    )

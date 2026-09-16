"""Análisis espectral de potencia (PSD) para pipelines EEG.

Módulo no invasivo que se integra en los pipelines existentes
(``test_iga_from_eeg_latent_*.py``) mediante imports y llamadas simples:

1. :func:`compute_and_plot_raw_psd` — PSD multitaper de los canales
   originales del EEG, con figuras (matriz por canal + media) y
   persistencia en ``.npz``.
2. :func:`compute_and_plot_latent_psd` — PSD del espacio latente
   (2–5 dimensiones), con figuras y persistencia.
3. :func:`compute_channel_influence_on_latent` — influencia de los
   canales originales sobre cada dimensión latente (topoplots si hay
   montaje, barras si no), con métrica espectral opcional por bandas.
   Si además hay matriz estructural de conectividad disponible
   (``DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH`` en ``src.utils.config``),
   genera la figura cualitativa adicional ``channel_influence_graph.png``
   (influencia coloreada sobre el grafo estructural).
4. :func:`load_psd_data` — carga de los ``.npz`` generados.

Método espectral: multitaper de MNE (tapers DPSS/Slepian) con pesos
adaptativos y ``low_bias=True``. Para señales largas (cuando el número
de tapers de una pasada única supera ``MAX_TAPERS_SINGLE_SHOT``) se usa
automáticamente un multitaper segmentado con solape del 50 % y promedio
de PSDs, para acotar el coste en tiempo y memoria. Si se usa Welch como
fallback en algún punto, la ventana debe ser ``'hann'``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.time_frequency import psd_array_multitaper

from .influence_metrics import (
    compute_channel_latent_correlation,
    compute_spectral_influence,
    extract_linear_weights,
)
from .plotting_utils import (
    create_channel_grid_figure,
    plot_bar_influence,
    plot_mean_psd,
    plot_psd_subplot,
    plot_topomap_grid,
)

# Plotter del grafo estructural (src.plotters). Import defensivo: si el
# paquete de plotters no está disponible, el análisis PSD sigue
# funcionando y simplemente no se genera la figura del grafo.
try:
    from src.plotters.structural_graph_plots import (
        load_structural_matrix,
        plot_structural_graph_influence,
    )
    _GRAPH_PLOTTER_AVAILABLE = True
except Exception:  # entorno sin src.plotters accesible
    _GRAPH_PLOTTER_AVAILABLE = False

__all__ = [
    "compute_and_plot_raw_psd",
    "compute_and_plot_latent_psd",
    "compute_channel_influence_on_latent",
    "load_psd_data",
]

logger = logging.getLogger(__name__)

DEFAULT_FMIN = 1.0
DEFAULT_FMAX = 100.0
DEFAULT_BANDWIDTH = 2.5
# §7.4: con más de 64 canales se pagina la matriz de subplots.
MAX_CHANNELS_PER_FIG = 64
# Máximo de tapers DPSS para un multitaper de una sola pasada. Por encima
# (señales largas: half_nbw = bandwidth·N/(2·sfreq) crece con N) se usa un
# multitaper segmentado con solape del 50 %, porque el coste en tiempo y
# memoria del cálculo full-signal se vuelve inviable (p. ej. ~749 tapers
# de 150 000 muestras para 300 s a 500 Hz con bandwidth=2.5).
MAX_TAPERS_SINGLE_SHOT = 16
_EPS = np.finfo(float).tiny
# Nombre de la figura adicional: influencia de canales sobre el grafo
# estructural (datos no-EEG con matriz de conectividad, p. ej. Ludovico_01).
GRAPH_INFLUENCE_FILENAME = "channel_influence_graph.png"


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _to_db(psds: np.ndarray) -> np.ndarray:
    """Convierte PSD a dB: 10 * log10(psd) (con suelo numérico)."""
    return 10.0 * np.log10(np.maximum(psds, _EPS))


def _resolve_picks(raw: mne.io.Raw, picks) -> np.ndarray:
    """Resuelve ``picks`` a índices de canal. Por defecto: todos los EEG."""
    info = raw.info
    if picks is None:
        idx = mne.pick_types(info, meg=False, eeg=True, exclude="bads")
        if len(idx) == 0:  # p. ej. canales marcados como 'misc' o 'seeg'
            idx = mne.pick_types(info, meg=True, eeg=True, exclude="bads")
        if len(idx) == 0:
            idx = np.setdiff1d(np.arange(info["nchan"]), mne.pick_channels(info["ch_names"], info["bads"]) if info["bads"] else [])
        return idx
    if isinstance(picks, str):
        if picks == "all":
            return np.setdiff1d(np.arange(info["nchan"]), mne.pick_channels(info["ch_names"], info["bads"]) if info["bads"] else [])
        if picks in info["ch_names"]:
            return np.array([info["ch_names"].index(picks)])
        return mne.pick_types(info, **{picks: True}, exclude="bads")
    picks = list(picks)
    if all(isinstance(p, (int, np.integer)) for p in picks):
        return np.asarray(picks, dtype=int)
    return mne.pick_channels(info["ch_names"], include=picks)


def _effective_fmax(fmax: float, sfreq: float) -> float:
    """Limita fmax al Nyquist (para datos filtrados, ajustar fmax al llamar)."""
    fmax_eff = min(float(fmax), sfreq / 2.0)
    if fmax_eff < float(fmax):
        logger.info("fmax=%.1f Hz supera Nyquist (%.1f Hz); se usa %.1f Hz.", fmax, sfreq / 2.0, fmax_eff)
    return fmax_eff


def _multitaper_auto(
    x: np.ndarray, sfreq: float, fmin: float, fmax: float, bandwidth: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """PSD multitaper de señales ``(n_series, n_times)`` con segmentación automática.

    El número de tapers DPSS de una pasada única crece con la duración de
    la señal (``2·half_nbw − 1`` con ``half_nbw = bandwidth·N/(2·sfreq)``)
    y con él el coste en tiempo y memoria, que se vuelve inviable para
    registros largos (p. ej. ~749 tapers de 150 000 muestras para 300 s a
    500 Hz con ``bandwidth=2.5``). Cuando supera
    ``MAX_TAPERS_SINGLE_SHOT``, la señal se divide en segmentos con
    solape del 50 %, se calcula el multitaper por segmento y se promedian
    los PSD (estilo Welch-multitaper): el número de tapers por segmento
    queda acotado y el coste crece solo linealmente con la duración.

    Parameters
    ----------
    x:
        Señales ``(n_series, n_times)``.

    Returns
    -------
    psds, freqs, info:
        ``psds``: ``(n_series, n_freqs)``; ``info``: dict con el modo
        (``"single"`` | ``"segmented"``), ``n_segments``,
        ``segment_length_s``, ``n_tapers`` y ``overlap``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n_times = x.shape[-1]
    half_nbw_full = bandwidth * n_times / (2.0 * sfreq)
    kmax_full = max(int(2.0 * half_nbw_full - 1.0), 1)

    if kmax_full <= MAX_TAPERS_SINGLE_SHOT:
        psds, freqs = psd_array_multitaper(
            x, sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=bandwidth,
            adaptive=True, low_bias=True, normalization="length",
            n_jobs=1, verbose=False,
        )
        info = {
            "windowing": "single",
            "n_segments": 1,
            "segment_length_s": n_times / sfreq,
            "n_tapers": kmax_full,
        }
        return np.atleast_2d(np.asarray(psds, dtype=float)), np.asarray(freqs, dtype=float), info

    # Segmentado: cada segmento produce como mucho MAX_TAPERS_SINGLE_SHOT
    # tapers; solape del 50 % y promedio de los PSD por segmento.
    n_per_seg = int(round((MAX_TAPERS_SINGLE_SHOT + 1) * sfreq / bandwidth))
    n_per_seg = min(n_per_seg, n_times)
    step = max(n_per_seg // 2, 1)
    starts = list(range(0, n_times - n_per_seg + 1, step))
    if not starts:
        starts = [0]
    acc, freqs = None, None
    for s in starts:
        psd_seg, freqs = psd_array_multitaper(
            x[:, s : s + n_per_seg], sfreq=sfreq, fmin=fmin, fmax=fmax,
            bandwidth=bandwidth, adaptive=True, low_bias=True,
            normalization="length", n_jobs=1, verbose=False,
        )
        acc = np.asarray(psd_seg, dtype=float) if acc is None else acc + psd_seg
    psds = acc / len(starts)
    info = {
        "windowing": "segmented",
        "n_segments": len(starts),
        "segment_length_s": n_per_seg / sfreq,
        "n_tapers": MAX_TAPERS_SINGLE_SHOT,
        "overlap": 0.5,
    }
    logger.info(
        "PSD multitaper segmentado: N=%d muestras (%d segmentos de %.1f s, "
        "solape 50%%, ~%d tapers/segmento) para acotar el coste.",
        n_times, len(starts), n_per_seg / sfreq, MAX_TAPERS_SINGLE_SHOT,
    )
    return np.atleast_2d(np.asarray(psds, dtype=float)), np.asarray(freqs, dtype=float), info


def _compute_latent_psds(
    latent: np.ndarray,
    sfreq: float,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    bandwidth: float = DEFAULT_BANDWIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """PSD de cada dimensión latente. Devuelve (psds (n_dim, n_freqs), freqs)."""
    fmax_eff = _effective_fmax(fmax, sfreq)
    psds, freqs, _ = _multitaper_auto(latent.T, sfreq, fmin, fmax_eff, bandwidth)
    return psds, freqs


def _resolve_channel_names(raw: mne.io.Raw, meta: dict | None, n_channels: int) -> list[str]:
    """Nombres de canal alineados con las filas de los pesos de influencia.

    Orden de preferencia: ``meta['preprocessing']['ch_names']`` (si
    consta), mapeo por ``meta['preprocessing']['kept_indices']`` (canales
    retenidos por stage 2), los nombres de ``raw`` si el número coincide,
    y como último recurso los primeros de ``raw`` con un aviso.
    """
    if isinstance(meta, dict):
        pre = meta.get("preprocessing")
        if isinstance(pre, dict):
            ch = pre.get("ch_names")
            if ch is not None and len(ch) == n_channels:
                return [str(c) for c in ch]
            kept = pre.get("kept_indices")
            if (
                kept is not None
                and len(kept) == n_channels
                and n_channels != len(raw.ch_names)
                and len(kept) > 0
                and int(np.max(kept)) < len(raw.ch_names)
            ):
                return [raw.ch_names[int(i)] for i in kept]
    if len(raw.ch_names) == n_channels:
        return list(raw.ch_names)
    logger.warning(
        "No se pudieron alinear los nombres de canal (%d pesos vs %d canales "
        "en raw); se usan los primeros de raw.",
        n_channels, len(raw.ch_names),
    )
    return list(raw.ch_names[:n_channels])


def _get_sfreq(raw: mne.io.Raw, meta: dict | None) -> float:
    """sfreq del espacio latente: meta['preprocessing'] o raw.info."""
    if isinstance(meta, dict):
        pre = meta.get("preprocessing")
        if isinstance(pre, dict) and pre.get("sfreq"):
            return float(pre["sfreq"])
    return float(raw.info["sfreq"])


def _resolve_structural_matrix_path() -> Path | None:
    """Path de la matriz estructural desde la config del proyecto.

    Import perezoso de ``DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH`` para no
    acoplar este módulo a la config en tiempo de import (los pipelines
    EEG no la necesitan). Devuelve ``None`` si la constante no existe o
    está vacía.
    """
    try:
        from src.utils.config import DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH
    except (ImportError, AttributeError):
        return None
    if not DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH:
        return None
    return Path(DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH)


def _maybe_plot_structural_graph(
    weights: np.ndarray,
    ch_names: list[str],
    dim_labels: list[str],
    out_dir: Path,
    *,
    force_recompute: bool = False,
    structural_matrix_path: str | Path | None = None,
) -> Path | None:
    """Genera ``channel_influence_graph.png`` si hay matriz estructural.

    Figura **adicional** (cualitativa) a las barras/topoplots: la
    influencia de cada canal se dibuja como el color de un nodo del
    grafo estructural (azul = positivo, rojo = negativo, intensidad
    proporcional a ``|peso|``), un panel por dimensión latente. Cada
    nodo lleva dentro su índice (0-based), que coincide con el índice
    del canal en ``ch_names`` / ``raw.ch_names``.

    Se omite silenciosamente (devuelve ``None``) cuando: el plotter de
    ``src.plotters`` no está disponible, no hay path de matriz en la
    config, el fichero no existe, o la matriz no es cuadrada del mismo
    n.º de canales que ``weights`` (p. ej. pipelines EEG con otro n.º
    de canales, donde la matriz de Ludovico_01 no aplica). Una figura
    ya existente no se regenera salvo ``force_recompute=True``.

    Returns
    -------
    Path | None
        Ruta de la figura generada (o ya existente), o ``None`` si se
        omitió.
    """
    if not _GRAPH_PLOTTER_AVAILABLE:
        return None
    out_dir = Path(out_dir)
    fig_path = out_dir / GRAPH_INFLUENCE_FILENAME
    if fig_path.exists() and not force_recompute:
        return fig_path

    path = (
        Path(structural_matrix_path)
        if structural_matrix_path is not None
        else _resolve_structural_matrix_path()
    )
    if path is None or not path.exists():
        return None

    weights = np.asarray(weights, dtype=float)
    try:
        adjacency = load_structural_matrix(path)
    except (OSError, ValueError) as exc:
        logger.warning("No se pudo cargar la matriz estructural %s: %s", path, exc)
        return None
    if adjacency.shape[0] != weights.shape[0]:
        logger.info(
            "Matriz estructural (%d nodos) no coincide con los canales de "
            "influencia (%d); se omite el plot de grafo.",
            adjacency.shape[0], weights.shape[0],
        )
        return None

    fig = plot_structural_graph_influence(
        weights,
        adjacency,
        dim_labels=list(dim_labels),
        # Etiqueta dentro del nodo = índice del nodo (0-based): el nodo i
        # es la fila/columna i de la matriz estructural y el canal i de
        # los datos (raw.ch_names[i]). Para mostrar nombres de canal en
        # su lugar, pasar node_labels=list(ch_names).
        node_labels=None,
        show_node_labels=True,
        layout="circular"
    )
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info("Plot de influencia sobre grafo estructural guardado: %s", fig_path)
    return fig_path


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_and_plot_raw_psd(
    raw: mne.io.Raw,
    out_dir: Path,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    bandwidth: float = DEFAULT_BANDWIDTH,
    picks: str | list | None = None,
    figsize_per_channel: tuple[float, float] = (3.0, 2.0),
    n_cols: int = 6,
    filename_prefix: str = "raw_psd",
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray, np.ndarray, Path]:
    """Calcula y visualiza el PSD de todos los canales originales del EEG.

    Genera ``{filename_prefix}_all_channels.png`` (matriz por canal),
    ``{filename_prefix}_mean.png`` (media ± std) y
    ``{filename_prefix}_data.npz`` (psds, freqs, ch_names, mean_psd,
    std_psd, metadata) en ``out_dir``.

    Si el ``.npz`` ya existe y ``force_recompute=False``, los PSDs se
    cargan de disco en lugar de recalcularse (las figuras solo se
    regeneran si faltan).

    Returns
    -------
    (psds, freqs, ch_names, mean_psd, std_psd, data_path):
        ``psds``: ``(n_channels, n_freqs)``; el resto según §4.1.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{filename_prefix}_data.npz"

    if data_path.exists() and not force_recompute:
        cached = load_psd_data(data_path)
        psds = cached["psds"]
        freqs = cached["freqs"]
        ch_names = list(cached["ch_names"])
        mean_psd = cached["mean_psd"]
        std_psd = cached["std_psd"]
        logger.info("PSD de canales cargado desde caché: %s", data_path)
    else:
        idx = _resolve_picks(raw, picks)
        ch_names = [raw.ch_names[i] for i in idx]
        sfreq = float(raw.info["sfreq"])
        fmax_eff = _effective_fmax(fmax, sfreq)
        psds, freqs, mt_info = _multitaper_auto(
            raw.get_data(picks=idx), sfreq, fmin, fmax_eff, bandwidth
        )
        mean_psd = psds.mean(axis=0)
        std_psd = psds.std(axis=0)
        metadata = {
            "method": "multitaper",
            "fmin": float(fmin),
            "fmax": float(fmax_eff),
            "fmax_requested": float(fmax),
            "bandwidth": float(bandwidth),
            "adaptive": True,
            "low_bias": True,
            "normalization": "length",
            "n_channels": len(ch_names),
            "sfreq": sfreq,
            **mt_info,
        }
        np.savez(
            data_path,
            psds=psds,
            freqs=freqs,
            ch_names=np.asarray(ch_names),
            mean_psd=mean_psd,
            std_psd=std_psd,
            metadata=np.array(metadata, dtype=object),
        )

    # Figura 1 — matriz de PSD por canal (paginada si n_channels > 64).
    n_channels = len(ch_names)
    if n_channels <= MAX_CHANNELS_PER_FIG:
        pages = [(0, n_channels, out_dir / f"{filename_prefix}_all_channels.png")]
    else:
        n_pages = int(np.ceil(n_channels / MAX_CHANNELS_PER_FIG))
        pages = [
            (
                p * MAX_CHANNELS_PER_FIG,
                min((p + 1) * MAX_CHANNELS_PER_FIG, n_channels),
                out_dir / f"{filename_prefix}_all_channels_p{p + 1:02d}.png",
            )
            for p in range(n_pages)
        ]
        logger.info("%d canales: matriz de PSD paginada en %d figuras.", n_channels, n_pages)

    psds_db = _to_db(psds)
    for start, stop, fig_path in pages:
        if fig_path.exists() and not force_recompute:
            continue
        fig, axes = create_channel_grid_figure(
            stop - start, n_cols=n_cols, figsize_per_item=figsize_per_channel
        )
        for k, ax in enumerate(axes):
            plot_psd_subplot(ax, freqs, psds_db[start + k], title=ch_names[start + k])
        fig.supxlabel("Frequency (Hz)")
        fig.supylabel("PSD (dB)")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    # Figura 2 — PSD promedio across canales (media y std en escala dB).
    mean_fig_path = out_dir / f"{filename_prefix}_mean.png"
    if not mean_fig_path.exists() or force_recompute:
        fig = plot_mean_psd(
            freqs,
            psds_db.mean(axis=0),
            psds_db.std(axis=0),
            title="Mean PSD across all channels (Multitaper, Hanning tapers)",
        )
        fig.savefig(mean_fig_path, dpi=150)
        plt.close(fig)

    return psds, freqs, ch_names, mean_psd, std_psd, data_path


def compute_and_plot_latent_psd(
    latent: np.ndarray,
    sfreq: float,
    out_dir: Path,
    fmin: float = DEFAULT_FMIN,
    fmax: float = DEFAULT_FMAX,
    bandwidth: float = DEFAULT_BANDWIDTH,
    filename_prefix: str = "latent_psd",
    force_recompute: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    """Calcula y visualiza el PSD del espacio latente (2–5 dimensiones).

    Genera ``{filename_prefix}_all_dims.png`` (un subplot por dimensión),
    ``{filename_prefix}_mean.png`` (media across dimensiones) y
    ``{filename_prefix}_data.npz`` (psds, freqs, mean_psd, std_psd,
    metadata) en ``out_dir``.

    Si la señal latente es más corta que el EEG original (Hankel
    embedding), se usa la misma ``sfreq`` y MNE infiere ``n_fft``.

    Returns
    -------
    (psds, freqs, mean_psd, std_psd, data_path):
        ``psds``: ``(n_dim, n_freqs)``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2:
        raise ValueError(f"latent debe ser 2D (n_samples, n_dim); recibido shape={latent.shape}")
    n_samples, n_dim = latent.shape
    data_path = out_dir / f"{filename_prefix}_data.npz"

    if data_path.exists() and not force_recompute:
        cached = load_psd_data(data_path)
        psds = cached["psds"]
        freqs = cached["freqs"]
        mean_psd = cached["mean_psd"]
        std_psd = cached["std_psd"]
        logger.info("PSD latente cargado desde caché: %s", data_path)
    else:
        fmax_eff = _effective_fmax(fmax, float(sfreq))
        psds, freqs, mt_info = _multitaper_auto(latent.T, float(sfreq), fmin, fmax_eff, bandwidth)
        mean_psd = psds.mean(axis=0)
        std_psd = psds.std(axis=0)
        metadata = {
            "method": "multitaper",
            "fmin": float(fmin),
            "fmax": float(fmax_eff),
            "fmax_requested": float(fmax),
            "bandwidth": float(bandwidth),
            "adaptive": True,
            "low_bias": True,
            "normalization": "length",
            "latent_dim": int(n_dim),
            "n_samples": int(n_samples),
            "sfreq": float(sfreq),
            **mt_info,
        }
        np.savez(
            data_path,
            psds=psds,
            freqs=freqs,
            mean_psd=mean_psd,
            std_psd=std_psd,
            metadata=np.array(metadata, dtype=object),
        )

    psds_db = _to_db(psds)

    # Figura 1 — PSD de cada dimensión latente.
    dims_fig_path = out_dir / f"{filename_prefix}_all_dims.png"
    if not dims_fig_path.exists() or force_recompute:
        n_cols_fig = 1 if n_dim <= 3 else 2
        fig, axes = create_channel_grid_figure(
            n_dim, n_cols=n_cols_fig, figsize_per_item=(7.0, 2.6)
        )
        for d, ax in enumerate(axes):
            plot_psd_subplot(ax, freqs, psds_db[d], title=f"Latent dimension {d + 1}")
        fig.supxlabel("Frequency (Hz)")
        fig.supylabel("PSD (dB)")
        fig.tight_layout()
        fig.savefig(dims_fig_path, dpi=150)
        plt.close(fig)

    # Figura 2 — PSD promedio del espacio latente (media y std en dB).
    mean_fig_path = out_dir / f"{filename_prefix}_mean.png"
    if not mean_fig_path.exists() or force_recompute:
        fig = plot_mean_psd(
            freqs,
            psds_db.mean(axis=0),
            psds_db.std(axis=0),
            title="Mean PSD across latent dimensions (Multitaper, Hanning tapers)",
        )
        fig.savefig(mean_fig_path, dpi=150)
        plt.close(fig)

    return psds, freqs, mean_psd, std_psd, data_path


def compute_channel_influence_on_latent(
    raw: mne.io.Raw,
    latent: np.ndarray,
    meta: dict,
    out_dir: Path,
    raw_psd_path: Path | None = None,
    filename: str = "channel_influence.png",
    force_recompute: bool = False,
) -> tuple[np.ndarray, list, Path, Path]:
    """Influencia de los canales originales sobre cada dimensión latente.

    Los pesos se extraen de la transformación lineal registrada en
    ``meta`` (PCA ``components_``, modos DMD, matriz ICA; ver
    :mod:`.influence_metrics`). Para métodos no lineales (p. ej.
    Diffusion Maps) se usa como fallback la correlación canal–latente.
    Los pesos se normalizan por dimensión (máximo |valor| = 1, signo
    preservado: valores en [-1, 1]).

    Visualización: topoplots si ``raw`` tiene montaje topográfico;
    barras horizontales en caso contrario (p. ej. datos CSV/Ludovico).

    Figura adicional (cualitativa): si existe matriz estructural de
    conectividad (constante ``DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH``
    en ``src.utils.config``) y su tamaño coincide con el n.º de canales,
    se genera también ``channel_influence_graph.png`` — los nodos del
    grafo estructural coloreados por la influencia (ver
    :mod:`src.plotters.structural_graph_plots`). Esta figura se genera
    aunque el resto de resultados se carguen desde caché.

    Si se proporciona ``raw_psd_path`` (el ``.npz`` de
    :func:`compute_and_plot_raw_psd`), se añade al ``.npz`` de salida la
    métrica espectral por bandas (``spectral_contribution`` y
    ``spectral_correlation``, ver §4.3.3).

    Genera ``{filename}`` y ``channel_influence_data.npz`` en ``out_dir``.

    Returns
    -------
    (influence_weights, ch_names, fig_path, data_path):
        ``influence_weights``: ``(n_channels, n_dim)`` en ``[-1, 1]``
        (signo preservado).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latent = np.asarray(latent, dtype=float)
    n_dim = latent.shape[1]
    dim_labels = [f"LD{d + 1}" for d in range(n_dim)]
    fig_path = out_dir / filename
    data_path = out_dir / "channel_influence_data.npz"

    if data_path.exists() and fig_path.exists() and not force_recompute:
        cached = load_psd_data(data_path)
        logger.info("Influencia de canales cargada desde caché: %s", data_path)
        # Figura adicional sobre el grafo estructural: se intenta generar
        # aunque el resto venga de caché (p. ej. pipelines corridos antes
        # de añadir este plot). No-op si la figura ya existe o no hay
        # matriz estructural aplicable.
        _maybe_plot_structural_graph(
            cached["influence_weights"],
            list(cached["ch_names"]),
            list(cached.get("latent_dim_labels", dim_labels)),
            out_dir,
        )
        return cached["influence_weights"], list(cached["ch_names"]), fig_path, data_path

    # 1. Pesos de transformación (lineal) o fallback por correlación.
    weights, method = extract_linear_weights(meta, n_dim)
    if weights is None:
        logger.info(
            "No hay transformación lineal disponible (%s). "
            "Fallback: correlación canal–latente.",
            method,
        )
        # Preferir la matriz filtrada real del pipeline (meta['preprocessing']
        # ['X_filtered']) si es consistente con los canales de raw.
        data = raw.get_data()
        if isinstance(meta, dict):
            pre = meta.get("preprocessing")
            X = pre.get("X_filtered") if isinstance(pre, dict) else None
            if (
                isinstance(X, np.ndarray)
                and X.ndim == 2
                and X.shape[0] <= data.shape[0]
                and X.shape[1] >= latent.shape[0]
            ):
                data = X
                method = f"{method}|X_filtered"
        weights = compute_channel_latent_correlation(data, latent)
        method = f"correlation_fallback({method})"

    ch_names = _resolve_channel_names(raw, meta, weights.shape[0])

    # Normalización por dimensión: máximo |valor| = 1, signo preservado.
    # (Antes se aplicaba np.abs() aquí, lo que hacía que TODAS las barras
    # salieran positivas con cualquier método; los pesos firmados son
    # interpretables en signo relativo entre canales de una misma dimensión.)
    weights = np.asarray(weights, dtype=float)
    col_max = np.abs(weights).max(axis=0, keepdims=True)
    col_max[col_max == 0.0] = 1.0
    weights = weights / col_max
    method += "|maxabs_norm"

    # 2. Métrica espectral de influencia (opcional, §4.3.3).
    spectral: dict | None = None
    if raw_psd_path is not None and Path(raw_psd_path).exists():
        try:
            raw_psd = load_psd_data(raw_psd_path)
            latent_psd_path = out_dir / "latent_psd_data.npz"
            if latent_psd_path.exists():
                latent_data = load_psd_data(latent_psd_path)
                latent_psds = latent_data["psds"]
                latent_freqs = latent_data["freqs"]
            else:
                meta_raw = raw_psd.get("metadata", {})
                latent_psds, latent_freqs = _compute_latent_psds(
                    latent,
                    _get_sfreq(raw, meta),
                    fmin=float(meta_raw.get("fmin", DEFAULT_FMIN)),
                    fmax=float(meta_raw.get("fmax", DEFAULT_FMAX)),
                    bandwidth=float(meta_raw.get("bandwidth", DEFAULT_BANDWIDTH)),
                )
            spectral = compute_spectral_influence(
                raw_psd["psds"], latent_psds, raw_psd["freqs"], latent_freqs, weights
            )
        except Exception as exc:  # la métrica espectral es opcional: no abortar
            logger.warning("No se pudo calcular la métrica espectral: %s", exc)

    # 3. Visualización: topoplots si hay montaje; barras en caso contrario.
    fig = _make_influence_figure(raw, ch_names, weights, dim_labels)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # 3b. Figura adicional (cualitativa): influencia sobre el grafo
    # estructural, si hay matriz de conectividad disponible y coincide
    # con el n.º de canales. No sustituye a las barras/topoplots.
    _maybe_plot_structural_graph(
        weights, ch_names, dim_labels, out_dir, force_recompute=force_recompute,
    )

    # 4. Persistencia.
    payload = {
        "influence_weights": weights,
        "ch_names": np.asarray(ch_names),
        "latent_dim_labels": np.asarray(dim_labels),
        "method": np.array(method, dtype=object),
    }
    if spectral is not None:
        payload.update(
            spectral_contribution=spectral["spectral_contribution"],
            spectral_correlation=spectral["spectral_correlation"],
            band_names=np.asarray(spectral["band_names"]),
            band_ranges=spectral["band_ranges"],
        )
    np.savez(data_path, **payload)

    return weights, ch_names, fig_path, data_path


def _make_influence_figure(
    raw: mne.io.Raw,
    ch_names: list[str],
    weights: np.ndarray,
    dim_labels: list[str],
) -> plt.Figure:
    """Topoplot grid si hay montaje; barras horizontales en caso contrario."""
    montage = raw.get_montage()
    available = [c for c in ch_names if c in raw.ch_names]
    if montage is not None and len(available) >= 3 and len(available) == len(ch_names):
        try:
            picks = mne.pick_channels(raw.ch_names, include=available)
            info_sel = mne.pick_info(raw.info, picks, copy=True)
            return plot_topomap_grid(info_sel, weights, dim_labels)
        except Exception as exc:
            logger.warning(
                "Fallo al generar topoplots (%s); se usan barras horizontales.", exc
            )
    elif montage is None:
        logger.info("Sin montaje topográfico: se usan barras horizontales.")
    return plot_bar_influence(weights, ch_names, dim_labels)


def load_psd_data(path: Path) -> dict:
    """Carga un ``.npz`` generado por este módulo y devuelve un diccionario.

    La clave ``metadata`` se devuelve como dict de Python y las listas de
    nombres (``ch_names``, ``latent_dim_labels``, ``band_names``) como
    listas de ``str``; el resto son arrays de numpy.
    """
    path = Path(path)
    out: dict = {}
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            value = data[key]
            if value.dtype == object and value.shape == ():
                value = value.item()
            elif value.dtype.kind in ("U", "S") and key in (
                "ch_names",
                "latent_dim_labels",
                "band_names",
            ):
                value = [str(v) for v in value.tolist()]
            out[key] = value
    return out

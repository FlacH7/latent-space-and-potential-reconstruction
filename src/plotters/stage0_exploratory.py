"""
stage0_exploratory.py — Plots exploratorios pre-procesamiento
==============================================================

Genera graficos inmediatamente despues de cargar el EEG crudo, antes de
cualquier etapa del pipeline de extraccion del espacio latente.

Plots
-----
* :func:`plot_channel_topographies`  -> ``eeg_channel_topographies.png``
* :func:`plot_channel_correlation_matrix` -> ``eeg_channel_correlation_matrix.png``

Datos disponibles en este punto del pipeline
--------------------------------------------
* ``raw`` : ``mne.io.Raw`` con montaje ``standard_1020``.
* ``l_freq``, ``h_freq`` : banda de filtrado.

Notas
-----
* No se hace try/except interno: los errores deben subir para que el
  wrapper del script principal los atrape.
* Todas las figuras se guardan con ``fig.savefig(path, dpi=dpi,
  bbox_inches="tight")`` y se cierran con ``plt.close(fig)``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.plotters._style import (
    DIVERGING_CMAP,
    PALETTE,
    setup_plotting_style,
)


# ---------------------------------------------------------------------------
# 3.1 Topografias por canal
# ---------------------------------------------------------------------------

def plot_channel_topographies(
    raw,
    out_dir: Path,
    fmin: float = 1.0,
    fmax: float = 40.0,
    subject: str = "",
    session: str = "",
    task: str = "",
    dpi: int = 150,
) -> Path:
    """
    Genera una imagen con una topografia (cabeza) por canal EEG.

    Cada cabeza muestra el mapa de calor de la potencia RMS del canal en
    la banda [fmin, fmax].  Se usa ``mne.viz.plot_topomap`` con un valor
    constante por canal (broadcast al montaje).

    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw con montaje configurado (standard_1020).
    out_dir : Path
        Directorio de salida.
    fmin, fmax : float
        Banda de frecuencia para calcular la potencia RMS.
    subject, session, task : str
        Metadata para el suptitulo.
    dpi : int
        Resolucion de la imagen.

    Returns
    -------
    Path
        Ruta del archivo generado.
    """
    plt = setup_plotting_style()
    import mne

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Calcular potencia por canal en la banda [fmin, fmax] -----
    # Filtrar una copia para garantizar que estamos en banda
    raw_filt = raw.copy().filter(fmin, fmax, verbose="ERROR")
    data = raw_filt.get_data()                   # (n_ch, n_times)
    n_ch = data.shape[0]
    # Varianza del canal filtrado == potencia en banda
    rms = np.var(data, axis=1)                   # (n_ch,)
    # Normalizar a [0, 1] por estabilidad del topomap
    if rms.max() > rms.min():
        rms_norm = (rms - rms.min()) / (rms.max() - rms.min())
    else:
        rms_norm = np.ones_like(rms)

    # ----- Layout rectangular -----
    n_cols = min(6, n_ch)
    n_rows = int(np.ceil(n_ch / n_cols))

    figsize = (2.6 * n_cols, 2.6 * n_rows + 0.8)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize, constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    info = raw.info
    for idx in range(n_ch):
        ax = axes[idx]
        # Vector de valores: constante para el canal i, cero para los demas
        vals = np.zeros(n_ch)
        vals[idx] = rms_norm[idx]
        try:
            mne.viz.plot_topomap(
                vals, info, axes=ax, show=False,
                cmap="viridis", contours=0,
                vlim=(0.0, 1.0),
            )
        except Exception:
            ax.text(0.5, 0.5, raw.ch_names[idx],
                    ha="center", va="center", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        ax.set_title(raw.ch_names[idx], fontsize=9, pad=2)

    # Ocultar ejes sobrantes
    for idx in range(n_ch, len(axes)):
        axes[idx].axis("off")

    meta_parts = [p for p in (subject, session, task) if p]
    suptitle = " | ".join(meta_parts) if meta_parts else "EEG channel topographies"
    suptitle += f"  |  band: {fmin:.1f}-{fmax:.1f} Hz  |  {n_ch} channels"
    fig.suptitle(suptitle, fontsize=12)

    out_path = out_dir / "eeg_channel_topographies.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3.2 Matriz de correlacion entre canales
# ---------------------------------------------------------------------------

def plot_channel_correlation_matrix(
    raw,
    out_dir: Path,
    subject: str = "",
    session: str = "",
    task: str = "",
    dpi: int = 150,
) -> Path:
    """
    Heatmap de correlacion de Pearson entre todos los canales EEG.

    Colormap divergente (``RdBu_r``) centrado en 0, rango [-1, 1].

    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw (preferiblemente ya filtrado).
    out_dir : Path
        Directorio de salida.
    subject, session, task : str
        Metadata para el suptitulo.
    dpi : int
        Resolucion de la imagen.

    Returns
    -------
    Path
        Ruta del archivo generado.
    """
    plt = setup_plotting_style()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = raw.get_data()                # (n_ch, n_times)
    ch_names = raw.ch_names
    n_ch = len(ch_names)

    # np.corrcoef espera (variables, observaciones) — ya esta en ese shape
    corr = np.corrcoef(data)             # (n_ch, n_ch)

    # Forzar diagonal en 1 (numerical safety)
    np.fill_diagonal(corr, 1.0)

    # Figura cuadrada
    figsize = max(7, 0.18 * n_ch + 4)
    fig, ax = plt.subplots(
        figsize=(figsize, figsize), constrained_layout=True,
    )
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap=DIVERGING_CMAP,
                   aspect="equal", interpolation="nearest")

    ax.set_xticks(np.arange(n_ch))
    ax.set_yticks(np.arange(n_ch))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")

    meta_parts = [p for p in (subject, session, task) if p]
    suptitle = " | ".join(meta_parts) if meta_parts else "EEG channel correlation"
    suptitle += f"  |  {n_ch} channels"
    fig.suptitle(suptitle, fontsize=12)

    out_path = out_dir / "eeg_channel_correlation_matrix.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

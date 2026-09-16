"""Utilidades de visualización para el análisis espectral de potencia (PSD).

Todas las funciones usan el backend no interactivo ``Agg`` de matplotlib.
El llamante es responsable de guardar la figura y cerrarla con
``plt.close(fig)`` (nunca se usa ``plt.show()``).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "create_channel_grid_figure",
    "plot_psd_subplot",
    "plot_mean_psd",
    "plot_topomap_grid",
    "plot_bar_influence",
]


def create_channel_grid_figure(
    n_items: int,
    n_cols: int = 6,
    figsize_per_item: tuple[float, float] = (3.0, 2.0),
    max_figsize: tuple[float, float] = (40.0, 30.0),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Crea una figura con una cuadrícula de subplots (filas automáticas).

    Parameters
    ----------
    n_items:
        Número de subplots necesarios (p. ej. canales).
    n_cols:
        Número de columnas de la cuadrícula.
    figsize_per_item:
        Tamaño base (ancho, alto) de cada subplot en pulgadas.
    max_figsize:
        Tamaño total máximo de la figura (para no explotar con muchos
        canales, ver §7.4 de la especificación).

    Returns
    -------
    fig, axes_used:
        Figura y lista con los ``n_items`` ejes útiles. Los ejes sobrantes
        de la cuadrícula quedan ocultos. Todos los ejes comparten X e Y.
    """
    n_cols = max(1, min(int(n_cols), int(n_items)))
    n_rows = int(np.ceil(n_items / n_cols))
    figsize = (
        min(n_cols * figsize_per_item[0], max_figsize[0]),
        min(n_rows * figsize_per_item[1], max_figsize[1]),
    )
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )
    flat = axes.ravel().tolist()
    for ax in flat[n_items:]:
        ax.set_visible(False)
    return fig, flat[:n_items]


def plot_psd_subplot(
    ax: plt.Axes,
    freqs: np.ndarray,
    psd_db: np.ndarray,
    title: str | None = None,
    color: str = "C0",
    linewidth: float = 0.9,
) -> None:
    """Dibuja un PSD (ya en dB) en un eje dado."""
    ax.plot(freqs, psd_db, color=color, lw=linewidth)
    ax.grid(True, alpha=0.3, lw=0.5)
    if title:
        ax.set_title(title, fontsize=8)


def plot_mean_psd(
    freqs: np.ndarray,
    mean_db: np.ndarray,
    std_db: np.ndarray,
    title: str,
    color: str = "C0",
) -> plt.Figure:
    """Figura con el PSD medio (dB) y banda sombreada de ±1 std."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(freqs, mean_db, color=color, lw=1.4, label="Mean PSD")
    ax.fill_between(
        freqs, mean_db - std_db, mean_db + std_db,
        color=color, alpha=0.25, label="±1 std",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_topomap_grid(
    info,
    weights: np.ndarray,
    dim_labels: list[str],
    title: str = "Channel influence on latent space",
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Cuadrícula con un topoplot por dimensión latente.

    Parameters
    ----------
    info:
        ``mne.Info`` con los canales de ``weights`` y posiciones (montaje).
    weights:
        Array ``(n_channels, n_dim)`` con la influencia normalizada,
        con signo (``[-1, 1]``). Se usa un colormap divergente con
        rango simétrico para representar influencias negativas.
    dim_labels:
        Etiquetas de las dimensiones latentes.
    """
    import mne

    weights = np.asarray(weights, dtype=float)
    n_dim = weights.shape[1]
    fig, axes = plt.subplots(1, n_dim, figsize=(3.4 * n_dim, 3.8), squeeze=False)
    vmax = float(np.max(np.abs(weights))) if np.max(np.abs(weights)) > 0 else 1.0
    for d in range(n_dim):
        ax = axes[0, d]
        im, _ = mne.viz.plot_topomap(
            weights[:, d],
            info,
            axes=ax,
            show=False,
            cmap=cmap,
            vlim=(-vmax, vmax),
            contours=0,
            sensors=True,
        )
        ax.set_title(dim_labels[d], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


def plot_bar_influence(
    weights: np.ndarray,
    ch_names: list[str],
    dim_labels: list[str],
    title: str = "Channel influence on latent space",
) -> plt.Figure:
    """Barras horizontales por dimensión (fallback sin montaje topográfico).

    Cada subplot se ordena independientemente.  Los pesos pueden ser
    positivos o negativos (se usan colores distintos).
    """
    weights = np.asarray(weights, dtype=float)
    n_dim = weights.shape[1]
    n_ch = len(ch_names)
    height = max(3.0, 0.28 * n_ch + 1.2)
    fig, axes = plt.subplots(
        1, n_dim, figsize=(4.2 * n_dim, height), squeeze=False, sharey=False
    )
    fontsize = 6 if n_ch > 32 else 8
    has_negative = bool(np.any(weights < 0))
    for d in range(n_dim):
        ax = axes[0, d]
        order = np.argsort(weights[:, d])
        vals = weights[order, d]
        y_pos = np.arange(n_ch)
        if has_negative:
            colors = ["#d62728" if v < 0 else "#1f77b4" for v in vals]
            ax.barh(y_pos, vals, color=colors, alpha=0.85)
            ax.axvline(x=0, color="k", lw=0.6, ls="--")
            ax.set_xlim(-1.05, 1.05)
        else:
            ax.barh(y_pos, vals, color="C0", alpha=0.85)
            ax.set_xlim(0.0, 1.02)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([ch_names[i] for i in order], fontsize=fontsize)
        ax.set_xlabel("Influence (norm.)")
        ax.set_title(dim_labels[d], fontsize=9)
        ax.grid(True, axis="x", alpha=0.3, lw=0.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig

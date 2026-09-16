"""
latent_detail.py — Plots de detalle del espacio latente
========================================================

Plots
-----
7.1 :func:`plot_latent_timeseries_zoom` -> ``latent_timeseries_zoom.png``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.plotters._style import PALETTE, setup_plotting_style


def plot_latent_timeseries_zoom(
    latent: np.ndarray,
    sfreq: float,
    out_dir: Path,
    zoom_fraction: float = 0.05,
    dpi: int = 150,
) -> Path:
    """
    Zoom centrado de las series temporales del espacio latente.

    Muestra un segmento de ``zoom_fraction`` del tiempo total, centrado
    en la mitad del recording.  Cada dimension se dibuja como subplot
    apilado verticalmente, compartiendo eje X de tiempo (segundos).

    Parameters
    ----------
    latent : np.ndarray, shape (n_samples, n_dim)
        Espacio latente.
    sfreq : float
        Frecuencia de muestreo (Hz).
    out_dir : Path
        Directorio de salida.
    zoom_fraction : float, default 0.05
        Fraccion del tiempo total a mostrar (e.g. 0.05 = 5%).
    dpi : int
        Resolucion.

    Returns
    -------
    Path
        Ruta del archivo generado.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent = np.asarray(latent, dtype=float)
    if latent.ndim != 2:
        raise ValueError(
            f"[plot_latent_timeseries_zoom] latent must be 2D, got {latent.shape}"
        )
    n_samples, n_dim = latent.shape

    n_zoom = max(10, int(n_samples * zoom_fraction))
    center = n_samples // 2
    start = max(0, center - n_zoom // 2)
    end = min(n_samples, start + n_zoom)
    if end - start < 10:
        # Fallback: tomar los ultimos n_zoom
        start = max(0, n_samples - n_zoom)
        end = n_samples

    zoom = latent[start:end, :]   # (n_zoom, n_dim)
    t_axis = np.arange(start, end) / float(sfreq)

    fig, axes = plt.subplots(
        n_dim, 1, figsize=(14, 2.5 * n_dim),
        constrained_layout=True, sharex=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for d in range(n_dim):
        ax = axes[d]
        ax.plot(t_axis, zoom[:, d], lw=0.8,
                color=PALETTE[d % len(PALETTE)])
        ax.set_ylabel(f"x{d+1}")
        ax.grid(True, alpha=0.3)
        if d == n_dim - 1:
            ax.set_xlabel("time (s)")
        ax.set_title(f"Latent dim {d+1}", fontsize=9)

    fig.suptitle(
        f"Latent time series (zoom {zoom_fraction*100:.1f}%)  |  "
        f"window = [{t_axis[0]:.2f}, {t_axis[-1]:.2f}] s",
        fontsize=11,
    )

    out_path = out_dir / "latent_timeseries_zoom.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

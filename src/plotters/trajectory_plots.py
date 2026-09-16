"""
trajectory_plots.py
=========================
Visualización de trayectorias latentes 2D y 3D, independiente del método
de extracción. Normaliza por desviación estándar y marca inicio/fin.

Genera TODAS las combinaciones de pares (2D) y tríos (3D) de dimensiones.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_latent_trajectory(
    latent: np.ndarray,
    out_dir: str | Path,
    method_name: str = "unknown",
    *,
    figsize_2d: tuple[float, float] = (6, 5),
    figsize_3d: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> list[Path]:
    """
    Genera figuras de trayectoria latente en 2D (todas las combinaciones
    de pares de dimensiones) y/o 3D (todas las combinaciones de tríos).

    Las trayectorias se normalizan por desviación estándar por dimensión.

    Parameters
    ----------
    latent : ndarray, shape (n_samples, n_dim)
        Espacio latente extraído (filas = tiempo, columnas = dimensiones).
    out_dir : str or Path
        Directorio donde guardar las figuras.
    method_name : str
        Nombre del método (para título y nombre de archivo).
    figsize_2d, figsize_3d : tuple
        Tamaños de figura.
    dpi : int
        Resolución de salida.

    Returns
    -------
    saved_paths : list[Path]
        Rutas de las figuras generadas.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalizar por desviación estándar por dimensión
    z = latent / np.std(latent, axis=0, keepdims=True)

    n_dim = latent.shape[1]
    saved_paths: list[Path] = []

    # ---------- Todas las combinaciones 2D ----------
    if n_dim >= 2:
        for d1, d2 in combinations(range(n_dim), 2):
            fig, ax = plt.subplots(figsize=figsize_2d)
            ax.plot(z[:, d1], z[:, d2], color=[0, 0, 0.5], linewidth=0.6)
            ax.scatter(z[0, d1], z[0, d2], s=80, c="g", marker="o",
                       edgecolors="k", label="Start", zorder=3)
            ax.scatter(z[-1, d1], z[-1, d2], s=80, c="r", marker="s",
                       edgecolors="k", label="End", zorder=3)
            ax.set_title(f"2D Latent Trajectory {method_name}\n(d{d1+1} vs d{d2+1})")
            ax.set_xlabel(f"Latent dim {d1 + 1} (normalized)")
            ax.set_ylabel(f"Latent dim {d2 + 1} (normalized)")
            ax.legend(loc="best")
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")

            fname = f"latent_trajectory_2d_{method_name}_d{d1+1}_d{d2+1}.png"
            fpath = out_dir / fname
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(fpath)
            print(f"  Saved {fname}")

    # ---------- Todas las combinaciones 3D ----------
    if n_dim >= 3:
        for d1, d2, d3 in combinations(range(n_dim), 3):
            fig = plt.figure(figsize=figsize_3d)
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(z[:, d1], z[:, d2], z[:, d3],
                    color=[0, 0, 0.5], linewidth=0.6)
            ax.scatter(z[0, d1], z[0, d2], z[0, d3],
                       s=80, c="g", marker="o", edgecolors="k",
                       label="Start", zorder=3)
            ax.scatter(z[-1, d1], z[-1, d2], z[-1, d3],
                       s=80, c="r", marker="s", edgecolors="k",
                       label="End", zorder=3)
            ax.set_title(f"3D Latent Trajectory {method_name}\n(d{d1+1}, d{d2+1}, d{d3+1})")
            ax.set_xlabel(f"Dim {d1 + 1}")
            ax.set_ylabel(f"Dim {d2 + 1}")
            ax.set_zlabel(f"Dim {d3 + 1}")
            ax.legend(loc="best")
            ax.grid(True)

            fname = f"latent_trajectory_3d_{method_name}_d{d1+1}_d{d2+1}_d{d3+1}.png"
            fpath = out_dir / fname
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(fpath)
            print(f"  Saved {fname}")

    return saved_paths
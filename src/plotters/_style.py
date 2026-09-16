"""
_style.py — Configuracion global de estilo para el modulo src.plotters
=====================================================================

Centraliza la configuracion de ``matplotlib.rcParams`` para garantizar
consistencia visual en todas las imagenes generadas por el pipeline EEG.

Convenciones:
* Tema claro coherente (fondo blanco, ejes grises suaves).
* Fuente sans-serif (DejaVu Sans / Noto Sans para Latin).
* DPI por defecto 150.
* Tamano de figura global 12x6 (cada funcion puede sobreescribirlo).
* Estilo de ejes desactivado por defecto (se controla via rcParams).

Notas
-----
* NO se llama a ``matplotlib.use("Agg")`` aqui: el script principal ya lo
  hace. Hacerlo de nuevo provocaria un warning de backend ya fijado.
* Toda funcion del modulo de ploteo debe llamar a :func:`setup_plotting_style`
  al menos la primera vez (es idempotente).
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paleta de colores consistente (qualitative, colorblind-friendly)
# ---------------------------------------------------------------------------
PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Paleta para clasificacion ICLabel
ICALABEL_COLORS = {
    "brain":       "#1f77b4",
    "eye":         "#ff7f0e",
    "muscle":      "#2ca02c",
    "heart":       "#d62728",
    "other":       "#7f7f7f",
    "line_noise":  "#9467bd",
}

# Colormap divergente por defecto (correlaciones, etc.)
DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL_CMAP = "viridis"

# ---------------------------------------------------------------------------
# Defaults de figura
# ---------------------------------------------------------------------------
DEFAULT_FIGSIZE = (12, 6)
DEFAULT_DPI = 150

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
_RCPARAMS = {
    "figure.figsize":      DEFAULT_FIGSIZE,
    "figure.dpi":          100,           # dpi en pantalla
    "savefig.dpi":         DEFAULT_DPI,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.1,

    "font.family":         "sans-serif",
    "font.sans-serif":     ["DejaVu Sans", "Noto Sans", "Arial", "Helvetica"],
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,

    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "grid.linestyle":      "--",
    "grid.linewidth":      0.5,
    "axes.axisbelow":      True,

    "axes.facecolor":      "white",
    "figure.facecolor":    "white",
    "axes.edgecolor":      "#444444",
    "xtick.color":         "#444444",
    "ytick.color":         "#444444",
    "axes.labelcolor":     "#222222",
    "axes.titlecolor":     "#111111",
    "text.color":          "#111111",

    "lines.linewidth":     1.0,
    "lines.markersize":    4,

    "legend.frameon":      False,

    "image.cmap":          SEQUENTIAL_CMAP,
}


def setup_plotting_style() -> "plt":
    """
    Configura los ``rcParams`` globales de matplotlib.

    Es idempotente: llamarla multiples veces no tiene efectos secundarios.

    Returns
    -------
    matplotlib.pyplot
        El modulo ``plt`` ya configurado, por comodidad.
    """
    matplotlib.rcParams.update(_RCPARAMS)
    return plt


def _apply_style() -> None:
    """Alias interno para garantizar el estilo en cada modulo."""
    setup_plotting_style()


# Aplicar al importar el modulo por primera vez
setup_plotting_style()

"""
stage1_embedding.py — Plots del embedding Hankel (Stage 1)
==========================================================

Plots
-----
* :func:`plot_hankel_singular_values`  -> ``stage1_hankel_singular_values.png``
* :func:`plot_hankel_variance_explained` -> ``stage1_hankel_variance_explained.png``

Datos disponibles
-----------------
* ``H`` : matriz de Hankel multivariada, shape (n_ch*T, n_times - T + 1)
* ``rank`` : rango SVD elegido (linea vertical)
* ``embedding_depth`` : profundidad T del embedding (info para el titulo)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

from src.plotters._style import PALETTE, setup_plotting_style


# ---------------------------------------------------------------------------
# Helper: SVD truncada segura
# ---------------------------------------------------------------------------

def _compute_truncated_svd(H: np.ndarray, k: int = 100) -> np.ndarray:
    """
    Calcula los primeros ``k`` valores singulares de ``H`` (orden desc).

    Si ``H`` es grande, usa ``scipy.sparse.linalg.svds`` (ARPACK).
    Si ``H`` es pequeno, usa ``numpy.linalg.svd`` directo.
    """
    m, n = H.shape
    k = max(1, min(k, min(m, n) - 1))
    if min(m, n) <= 2:
        # Caso degenerate: SVD full
        s = np.linalg.svd(H, compute_uv=False)
        return s[:k] if k <= len(s) else s
    try:
        # svds devuelve en orden ASCENDENTE
        s = svds(H.astype(float), k=k, return_singular_vectors=False)
        s = np.sort(s)[::-1]
    except Exception:
        s = np.linalg.svd(H, compute_uv=False)
        s = s[:k]
    return s


# ---------------------------------------------------------------------------
# 4.1 Valores singulares de la matriz de Hankel
# ---------------------------------------------------------------------------

def plot_hankel_singular_values(
    H: np.ndarray,
    out_dir: Path,
    rank: int | None = None,
    embedding_depth: int | None = None,
    dpi: int = 150,
) -> Path:
    """
    Grafico de valores singulares de la matriz de Hankel.

    Lineas con puntos; escala log en Y si los valores abarcan varios
    ordenes de magnitud.

    Parameters
    ----------
    H : np.ndarray, shape (n_ch*T, n_times-T+1)
        Matriz de Hankel.
    out_dir : Path
        Directorio de salida.
    rank : int | None
        Rango SVD elegido (linea vertical punteada).
    embedding_depth : int | None
        Profundidad T del embedding (para el titulo).
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

    s = _compute_truncated_svd(H, k=100)
    n_sv = len(s)
    x = np.arange(1, n_sv + 1)

    # Log scale si abarca mas de 2 ordenes de magnitud
    use_log = (s.max() / max(s.min(), 1e-30)) > 100

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(x, s, marker="o", ls="-", lw=1.0, ms=3,
            color=PALETTE[0], label="singular values")
    if use_log:
        ax.set_yscale("log")
        ax.set_ylabel("singular value (log)")
    else:
        ax.set_ylabel("singular value")
    ax.set_xlabel("index (1-based)")
    title = "Hankel matrix - singular values"
    if embedding_depth is not None:
        title += f"  |  depth T={embedding_depth}"
    title += f"  |  H shape={H.shape}"
    ax.set_title(title, fontsize=11)

    if rank is not None and 0 < rank <= n_sv:
        ax.axvline(rank, ls="--", color=PALETTE[3], lw=1.2,
                   label=f"rank={rank}")
        ax.legend(loc="upper right")

    ax.grid(True, which="both", alpha=0.3)

    out_path = out_dir / "stage1_hankel_singular_values.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4.2 Varianza explicada por subespacio (SVD)
# ---------------------------------------------------------------------------

def plot_hankel_variance_explained(
    H: np.ndarray,
    out_dir: Path,
    rank: int | None = None,
    embedding_depth: int | None = None,
    dpi: int = 150,
) -> Path:
    """
    Barras individuales + acumulada de varianza explicada (SVD).

    Panel izquierdo: porcentaje individual ``sigma_i^2 / sum(sigma_j^2)``.
    Panel derecho: acumulado con umbrales 90/95/99 %.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = _compute_truncated_svd(H, k=100)
    var = s ** 2
    total = var.sum()
    if total <= 0:
        raise RuntimeError(
            "[plot_hankel_variance_explained] All singular values are zero."
        )
    pct_individual = 100.0 * var / total
    pct_cumulative = np.cumsum(pct_individual)
    n_sv = len(s)
    x = np.arange(1, n_sv + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True, sharex=True,
    )

    # --- Panel izquierdo: individual ---
    ax1.bar(x, pct_individual, color=PALETTE[0], alpha=0.85)
    ax1.set_xlabel("component index")
    ax1.set_ylabel("variance explained (%)")
    ax1.set_title("Individual", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)
    if rank is not None and 0 < rank <= n_sv:
        ax1.axvline(rank, ls="--", color=PALETTE[3], lw=1.2,
                    label=f"rank={rank}")
        ax1.legend(loc="upper right")

    # --- Panel derecho: acumulado ---
    ax2.plot(x, pct_cumulative, marker="o", ms=3, ls="-",
             color=PALETTE[1], label="cumulative")
    ax2.fill_between(x, 0, pct_cumulative, color=PALETTE[1], alpha=0.15)
    for thr, col in zip([90, 95, 99],
                        [PALETTE[2], PALETTE[3], PALETTE[4]]):
        ax2.axhline(thr, ls="--", color=col, lw=1.0, alpha=0.7,
                    label=f"{thr}%")
    ax2.set_xlabel("component index")
    ax2.set_ylabel("cumulative variance (%)")
    ax2.set_title("Cumulative", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    suptitle = "Hankel SVD - variance explained"
    if embedding_depth is not None:
        suptitle += f"  |  depth T={embedding_depth}"
    fig.suptitle(suptitle, fontsize=12)

    out_path = out_dir / "stage1_hankel_variance_explained.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

"""
stage3_selection.py — Plots de seleccion Markov (Stage 3)
=========================================================

Plots
-----
6.1 :func:`plot_markov_tau_heatmap`  -> ``stage3_markov_tau_heatmap.png``
6.2 :func:`plot_markov_tau_ranked`   -> ``stage3_markov_tau_ranked.png``
6.3 :func:`plot_markov_selection_vs_distribution``
                                  -> ``stage3_markov_selection_vs_distribution.png``
6.4 :func:`plot_markov_transition_matrix``
                                  -> ``stage3_markov_transition_matrix.png``

Dependencias
------------
* ``all_taus``: lista de ``(combination, tau)`` producida por
  ``find_best_subspace_markov(..., return_all_taus=True)``.
* ``Y2``: salida de Stage 2, shape (D, T).
* ``selected_combination``: tupla de indices seleccionados.
* ``n_bins``: numero de bins usado en la discretizacion.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np

from src.plotters._style import (
    DIVERGING_CMAP,
    PALETTE,
    SEQUENTIAL_CMAP,
    setup_plotting_style,
)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _taus_to_arrays(all_taus):
    """
    Convierte ``all_taus`` (lista de (comb, tau)) en arrays paralelos.
    """
    combs = [tuple(c) for c, _ in all_taus]
    taus = np.array([t for _, t in all_taus], dtype=float)
    return combs, taus


# ---------------------------------------------------------------------------
# 6.1 Heatmap de tiempos de Markov
# ---------------------------------------------------------------------------

def plot_markov_tau_heatmap(
    all_taus,
    out_dir: Path,
    selected_combination: tuple[int, ...] | None = None,
    dpi: int = 150,
) -> Path | None:
    """
    Heatmap cuadrado de tau(i, j) para n_dim=2.

    Solo la triangular superior es informativa.  Marca con borde negro
    grueso la combinacion seleccionada.  ``inf`` se muestra en gris.

    Retorna ``None`` si ``n_dim != 2``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combs, taus = _taus_to_arrays(all_taus)
    if not combs:
        return None
    # Detectar n_dim
    n_dim = len(combs[0])
    if n_dim != 2:
        return None

    # Inferir D a partir de los indices
    max_idx = max(max(c) for c in combs)
    D = max_idx + 1

    # Matriz DxD con NaN
    M = np.full((D, D), np.nan, dtype=float)
    inf_mask = np.zeros((D, D), dtype=bool)
    for (i, j), tau in zip(combs, taus):
        if np.isfinite(tau):
            M[i, j] = tau
            M[j, i] = tau  # espejo para visualizacion
        else:
            inf_mask[i, j] = True
            inf_mask[j, i] = True

    # Para colormap: usar solo valores finitos para normalizar
    finite_vals = M[np.isfinite(M)]
    if finite_vals.size == 0:
        return None
    vmin, vmax = float(finite_vals.min()), float(finite_vals.max())

    # Mask para imshow: NaN + inf -> color gris especial
    display = np.where(np.isfinite(M), M, np.nan)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#cccccc")  # NaN y inf en gris
    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")

    # Etiquetas
    ax.set_xticks(np.arange(D))
    ax.set_yticks(np.arange(D))
    ax.set_xlabel("component j")
    ax.set_ylabel("component i")
    ax.set_title("Markov relaxation time τ(i, j)  (n_dim=2)",
                 fontsize=11)

    # Marcar combinacion seleccionada
    if selected_combination is not None and len(selected_combination) == 2:
        i_s, j_s = sorted(selected_combination)
        # Rectangulo en (i_s, j_s) y (j_s, i_s)
        for (a, b) in [(i_s, j_s), (j_s, i_s)]:
            rect = plt.Rectangle(
                (b - 0.5, a - 0.5), 1, 1,
                fill=False, edgecolor="black", lw=3,
            )
            ax.add_patch(rect)

    # Marcar combinaciones inf con texto "∞"
    for i in range(D):
        for j in range(D):
            if inf_mask[i, j] and i != j:
                ax.text(j, i, "∞", ha="center", va="center",
                        fontsize=8, color="#444444")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("τ (steps)")

    out_path = out_dir / "stage3_markov_tau_heatmap.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6.2 Tau ordenados de todas las combinaciones
# ---------------------------------------------------------------------------

def plot_markov_tau_ranked(
    all_taus,
    out_dir: Path,
    selected_combination: tuple[int, ...] | None = None,
    dpi: int = 150,
) -> Path:
    """
    Dos paneles: tau ordenados + histograma.

    Panel superior: linea con puntos de todos los tau (menor a mayor),
    linea horizontal punteada en la media, anotaciones de tau_min,
    tau_max y tau de la combinacion seleccionada.

    Panel inferior: histograma de la distribucion de tau (excluyendo inf).
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combs, taus = _taus_to_arrays(all_taus)

    fin_mask = np.isfinite(taus)
    taus_fin = taus[fin_mask]
    n_inf = int((~fin_mask).sum())

    # Panel superior: ordenados
    if taus_fin.size > 0:
        order = np.argsort(taus_fin)
        taus_sorted = taus_fin[order]
        combs_fin = [combs[i] for i in np.where(fin_mask)[0]]
        combs_sorted = [combs_fin[i] for i in order]
    else:
        taus_sorted = np.array([])
        combs_sorted = []

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), constrained_layout=True,
    )

    # --- Panel superior ---
    if taus_sorted.size > 0:
        x = np.arange(1, len(taus_sorted) + 1)
        ax1.plot(x, taus_sorted, marker="o", ms=3, ls="-",
                 color=PALETTE[0], lw=1.0)
        mean_tau = float(taus_sorted.mean())
        ax1.axhline(mean_tau, ls="--", color=PALETTE[3], lw=1.0,
                    label=f"mean = {mean_tau:.3g}")

        # tau_min y tau_max
        i_min = int(np.argmin(taus_sorted))
        i_max = int(np.argmax(taus_sorted))
        ax1.scatter([i_min + 1], [taus_sorted[i_min]],
                    color=PALETTE[2], s=80, zorder=5,
                    label=f"τ_min = {taus_sorted[i_min]:.3g} {combs_sorted[i_min]}")
        ax1.scatter([i_max + 1], [taus_sorted[i_max]],
                    color=PALETTE[4], s=80, zorder=5,
                    label=f"τ_max = {taus_sorted[i_max]:.3g} {combs_sorted[i_max]}")

        # Combinacion seleccionada
        if selected_combination is not None:
            try:
                sel_idx = combs_sorted.index(tuple(selected_combination))
                ax1.scatter([sel_idx + 1], [taus_sorted[sel_idx]],
                            color="black", s=120, marker="*", zorder=6,
                            label=f"selected = {tuple(selected_combination)}")
            except ValueError:
                pass
        ax1.set_xlabel("rank (sorted by τ)")
        ax1.set_ylabel("τ (steps)")
        ax1.set_title(
            f"All combinations ranked by τ  |  n_finite={len(taus_sorted)}"
            f"  n_inf={n_inf}",
            fontsize=10,
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="lower right", fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No finite τ values available",
                 ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("All combinations ranked by τ", fontsize=10)

    # --- Panel inferior: histograma ---
    if taus_sorted.size > 0:
        ax2.hist(taus_sorted, bins=min(30, max(10, len(taus_sorted) // 5)),
                 color=PALETTE[1], alpha=0.85)
        ax2.set_xlabel("τ (steps)")
        ax2.set_ylabel("count")
        ax2.set_title("Distribution of τ (excluding ∞)", fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No finite τ values to histogram",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle("Stage 3 - Markov τ distribution", fontsize=12)

    out_path = out_dir / "stage3_markov_tau_ranked.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6.3 Seleccion vs distribucion
# ---------------------------------------------------------------------------

def plot_markov_selection_vs_distribution(
    all_taus,
    selected_tau: float,
    selected_combination: tuple[int, ...],
    maximize: bool,
    out_dir: Path,
    dpi: int = 150,
) -> Path | None:
    """
    Histograma de tau con flecha vertical en tau seleccionado + percentil.

    Solo meaningful para ``markov_fastest`` o ``markov_slowest``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combs, taus = _taus_to_arrays(all_taus)
    fin_mask = np.isfinite(taus)
    taus_fin = taus[fin_mask]
    if taus_fin.size == 0:
        return None

    # Percentil de selected_tau
    pct = float((taus_fin <= selected_tau).mean() * 100.0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.hist(taus_fin, bins=min(40, max(10, len(taus_fin) // 5)),
            color=PALETTE[0], alpha=0.7)
    ax.axvline(selected_tau, color=PALETTE[3], lw=2.5, ls="--",
               label=f"selected τ = {selected_tau:.3g}  (pct={pct:.1f}%)")
    ax.set_xlabel("τ (steps)")
    ax.set_ylabel("count")
    label = "slowest (maximize)" if maximize else "fastest (minimize)"
    ax.set_title(
        f"Markov selection vs distribution  |  strategy={label}  |  "
        f"selected={tuple(selected_combination)}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    out_path = out_dir / "stage3_markov_selection_vs_distribution.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 6.4 Matriz de transicion de la combinacion seleccionada
# ---------------------------------------------------------------------------

def plot_markov_transition_matrix(
    Y2: np.ndarray,
    selected_combination: tuple[int, ...],
    n_bins: int,
    out_dir: Path,
    dpi: int = 150,
) -> Path:
    """
    Heatmap de la matriz de transicion P de la combinacion seleccionada.
    """
    plt = setup_plotting_style()
    from src.plotters._markov_helpers import compute_transition_matrix

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    P, n_states, _ = compute_transition_matrix(
        Y2, selected_combination, n_bins,
    )

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    # Log scale para mejor visualizacion (muchas celdas son 0)
    P_safe = np.where(P > 0, P, np.nan)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")
    im = ax.imshow(P_safe, cmap=cmap, vmin=0, vmax=1.0,
                   aspect="equal", interpolation="nearest")

    ax.set_xlabel("state at t+1")
    ax.set_ylabel("state at t")
    ax.set_title(
        f"Transition matrix P  |  combination={tuple(selected_combination)}"
        f"  |  n_bins={n_bins}  |  n_states={n_states}",
        fontsize=11,
    )
    # Mostrar ticks solo si n_states es pequeno
    if n_states <= 30:
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
    else:
        ax.set_xticks(np.arange(0, n_states, max(1, n_states // 10)))
        ax.set_yticks(np.arange(0, n_states, max(1, n_states // 10)))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(i, j)")

    out_path = out_dir / "stage3_markov_transition_matrix.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

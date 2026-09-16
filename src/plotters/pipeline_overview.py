"""
pipeline_overview.py — Plots cross-stage (bonus)
=================================================

Plots
-----
8.1 :func:`plot_pipeline_energy_budget` -> ``pipeline_energy_budget.png``
8.2 :func:`plot_pipeline_flowchart`     -> ``pipeline_flowchart.png``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.plotters._style import PALETTE, setup_plotting_style


# ---------------------------------------------------------------------------
# 8.1 Energia a traves del pipeline
# ---------------------------------------------------------------------------

def plot_pipeline_energy_budget(
    X_filtered: np.ndarray | None,
    meta: dict,
    latent: np.ndarray,
    out_dir: Path,
    dpi: int = 150,
) -> Path:
    """
    Bar chart de la energia (suma de varianzas) en cada etapa.

    Etapas:
        1. EEG filtrado original (X_filtered)
        2. Embedding Hankel (suma de sigma_i^2 si disponible)
        3. Stage 2 output (suma de varianzas de las D componentes)
        4. Stage 3 output (suma de varianzas de las n_dim componentes finales)
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_labels = []
    energies = []

    # 1) EEG filtrado
    if X_filtered is not None:
        X_arr = np.asarray(X_filtered, dtype=float)
        e1 = float(np.var(X_arr, axis=1).sum())
        stage_labels.append("Stage 0\n(filtered EEG)")
        energies.append(e1)

    # 2) Hankel embedding (usamos sigma_i^2)
    stage1 = meta.get("stage1", {}) or {}
    H_shape = stage1.get("output_shape")
    # Si hay singular_values en stage2 (rama Hankel), lo usamos como proxy
    stage2 = meta.get("stage2", {}) or {}
    sv = stage2.get("singular_values")
    if sv is not None and H_shape is not None:
        sv_arr = np.asarray(sv, dtype=float).ravel()
        e2 = float((sv_arr ** 2).sum())
        stage_labels.append(f"Stage 1\n(Hankel, shape={H_shape})")
        energies.append(e2)
    elif H_shape is not None:
        # Sin SVD disponible: usar suma de varianzas del embedding si esta
        stage_labels.append(f"Stage 1\n(Hankel, shape={H_shape})")
        energies.append(np.nan)

    # 3) Stage 2 output (Y2)
    Y2 = meta.get("Y")
    if Y2 is not None:
        Y2_arr = np.asarray(Y2, dtype=float)
        e3 = float(np.var(Y2_arr, axis=1).sum())
        stage_labels.append(f"Stage 2\n(Y2, D={Y2_arr.shape[0]})")
        energies.append(e3)

    # 4) Stage 3 output (latent)
    latent_arr = np.asarray(latent, dtype=float)
    e4 = float(np.var(latent_arr, axis=0).sum())
    stage_labels.append(f"Stage 3\n(latent, n_dim={latent_arr.shape[1]})")
    energies.append(e4)

    # Filtrar NaN
    plot_labels = []
    plot_e = []
    for lbl, e in zip(stage_labels, energies):
        if not np.isnan(e):
            plot_labels.append(lbl)
            plot_e.append(e)

    if not plot_e:
        raise RuntimeError(
            "[plot_pipeline_energy_budget] No energy data available."
        )

    # Porcentajes relativos a la primera etapa
    e0 = plot_e[0] if plot_e[0] > 0 else max(plot_e)
    pcts = [100.0 * e / e0 for e in plot_e]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(
        plot_labels, plot_e,
        color=[PALETTE[i % len(PALETTE)] for i in range(len(plot_e))],
        alpha=0.85,
    )
    ax.set_ylabel("energy (sum of variances)")
    ax.set_title("Pipeline energy budget", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    # Anotar con valores absolutos y porcentajes
    for bar, e, p in zip(bars, plot_e, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(plot_e) * 0.01,
                f"{e:.3g}\n({p:.1f}%)",
                ha="center", fontsize=9)

    if len(plot_e) > 1:
        ax.set_ylim(0, max(plot_e) * 1.15)

    out_path = out_dir / "pipeline_energy_budget.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 8.2 Diagrama de flujo del pipeline con parametros
# ---------------------------------------------------------------------------

def plot_pipeline_flowchart(
    meta: dict,
    out_dir: Path,
    l_freq: float,
    h_freq: float,
    n_channels: int,
    sfreq: float,
    dpi: int = 150,
) -> Path:
    """
    Diagrama de flujo (flowchart) con matplotlib patches.

    Cajas rectangulares conectadas por flechas, cada una con el nombre
    de la etapa y los parametros clave.
    """
    plt = setup_plotting_style()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = meta.get("pipeline", {}) or {}
    stage1_name = pipeline.get("stage1") or "none"
    stage2_name = pipeline.get("stage2") or "pca_ica"
    stage3_name = pipeline.get("stage3") or "top_n"

    stage1_meta = meta.get("stage1", {}) or {}
    stage2_meta = meta.get("stage2", {}) or {}
    stage3_meta = meta.get("stage3", {}) or {}
    preprocessing = meta.get("preprocessing", {}) or {}

    # Construir texto de cada caja
    box0 = (
        f"Stage 0: Preprocessing\n"
        f"Filter: {l_freq}-{h_freq} Hz\n"
        f"Channels: {n_channels}  |  sfreq: {sfreq:.1f} Hz"
    )

    hankel_shape = stage1_meta.get("output_shape")
    if hankel_shape is None:
        hankel_shape = "-"
    depth = stage1_meta.get("depth")
    if depth is None:
        depth = preprocessing.get("embedding_depth")
    if stage1_name == "hankel":
        box1 = (
            f"Stage 1: Hankel Embedding\n"
            f"depth T: {depth}\n"
            f"H shape: {hankel_shape}"
        )
    else:
        box1 = (
            f"Stage 1: Identity (no Hankel)\n"
            f"depth: -\n"
            f"H shape: -"
        )

    branch = stage2_meta.get("branch", "-")
    # Use explicit None checks (svd_rank may be int or array in some metas)
    svd_rank = stage2_meta.get("svd_rank")
    if svd_rank is None:
        svd_rank = stage2_meta.get("n_components")
    if svd_rank is None:
        svd_rank = "-"
    if stage2_name == "diffusion_maps":
        sigma = stage2_meta.get("sigma_used")
        if sigma is None:
            sigma = stage2_meta.get("epsilon_fitted")
        if sigma is None:
            sigma = "auto"
        k_nn = stage2_meta.get("k_neighbors", "-")
        alpha = stage2_meta.get("alpha", "-")
        box2 = (
            f"Stage 2: Diffusion Maps\n"
            f"branch: {branch}\n"
            f"svd_rank: {svd_rank}  |  k: {k_nn}\n"
            f"alpha: {alpha}  |  sigma: {sigma}"
        )
    elif stage2_name == "dmd":
        rank = stage2_meta.get("rank")
        if rank is None:
            rank = svd_rank
        box2 = (
            f"Stage 2: DMD\n"
            f"branch: {branch}\n"
            f"rank: {rank}"
        )
    elif stage2_name == "pca_ica":
        n_comp = stage2_meta.get("n_components")
        if n_comp is None:
            n_comp = svd_rank
        ica_method = stage2_meta.get("ica_method", "-")
        box2 = (
            f"Stage 2: PCA + ICA\n"
            f"branch: {branch}\n"
            f"n_components: {n_comp}  |  ica: {ica_method}"
        )
    else:  # pca
        box2 = (
            f"Stage 2: PCA\n"
            f"branch: {branch}\n"
            f"n_components: {svd_rank}"
        )

    selected = stage3_meta.get("selected_indices", [])
    scores = stage3_meta.get("scores", {}) or {}
    n_bins = scores.get("n_bins", "-")
    if stage3_name.startswith("markov"):
        tau = scores.get("tau", "-")
        box3 = (
            f"Stage 3: {stage3_name}\n"
            f"n_bins: {n_bins}  |  tau: {tau}\n"
            f"selected: {selected}"
        )
    else:
        box3 = (
            f"Stage 3: top_n\n"
            f"selected: {selected}"
        )

    boxes = [box0, box1, box2, box3]
    n_boxes = len(boxes)

    fig, ax = plt.subplots(figsize=(7, 1.8 * n_boxes + 1.5),
                           constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_boxes + 0.5)
    ax.axis("off")

    box_w, box_h = 0.8, 1.0
    x_center = 0.5

    # Coordenadas Y de cada caja (de arriba a abajo)
    ys = [n_boxes - i - 0.5 for i in range(n_boxes)]   # centros

    for i, (txt, yc) in enumerate(zip(boxes, ys)):
        rect = FancyBboxPatch(
            (x_center - box_w / 2, yc - box_h / 2),
            box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.5,
            edgecolor=PALETTE[i % len(PALETTE)],
            facecolor="white",
        )
        ax.add_patch(rect)
        ax.text(x_center, yc, txt, ha="center", va="center",
                fontsize=9, family="monospace")

    # Flechas entre cajas
    for i in range(n_boxes - 1):
        y_from = ys[i] - box_h / 2
        y_to = ys[i + 1] + box_h / 2
        arrow = FancyArrowPatch(
            (x_center, y_from), (x_center, y_to),
            arrowstyle="-|>", mutation_scale=20,
            color="#444444", lw=1.5,
        )
        ax.add_patch(arrow)

    ax.set_title(
        f"Pipeline flowchart  |  {stage1_name}+{stage2_name}+{stage3_name}",
        fontsize=12, pad=10,
    )

    out_path = out_dir / "pipeline_flowchart.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

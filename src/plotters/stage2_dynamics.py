"""
stage2_dynamics.py — Plots de la etapa de dinamica (Stage 2)
=============================================================

Cada metrica (pca, pca_ica, dmd, diffusion_maps) produce datos
distintos en ``meta["stage2"]``.  Cada funcion debe verificar la
metrica y solo generar el grafico cuando los datos correspondientes
esten disponibles.  Si no aplica, retorna ``None`` sin generar archivo.

Plots
-----
5.1a :func:`plot_pca_variance_explained`
5.1b :func:`plot_pre_ica_component_psds`
5.1c :func:`plot_icalabel_summary`
5.1d :func:`plot_post_ica_component_psds`
5.2a :func:`plot_hankel_pca_singular_values`
5.2b :func:`plot_fastica_convergence`
5.3a :func:`plot_dmd_eigenvalue_unit_circle`
5.3b :func:`plot_dmd_frequency_damping`
5.4a :func:`plot_diffusion_eigenvalue_spectrum`
5.4b :func:`plot_diffusion_kernel_diagnostics`
5.4c :func:`plot_diffusion_2d_components`
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np

from src.plotters._style import (
    DIVERGING_CMAP,
    ICALABEL_COLORS,
    PALETTE,
    SEQUENTIAL_CMAP,
    setup_plotting_style,
)


# ===========================================================================
# 5.1 — pca_ica, rama MNE (branch="mne_ica_iclabel")
# ===========================================================================

# ---------------------------------------------------------------------------
# 5.1a Varianza explicada por PCA (pre-ICA)
# ---------------------------------------------------------------------------

def plot_pca_variance_explained(
    meta_stage2: dict,
    out_dir: Path,
    dpi: int = 150,
) -> Path | None:
    """
    Scree plot de la PCA interna de MNE ICA + varianza acumulada.

    Solo aplica cuando ``meta_stage2["branch"] == "mne_ica_iclabel"``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("branch") != "mne_ica_iclabel":
        return None

    # Buscar varianza explicada: directamente o dentro de preprocessing
    # Avoid `or` chains with numpy arrays (ambiguous truth value)
    var = meta_stage2.get("pca_explained_variance_")
    if var is None:
        var = meta_stage2.get("explained_variance_")
    if var is None:
        sub = meta_stage2.get("preprocessing") or {}
        var = sub.get("pca_explained_variance_")
    if var is None:
        return None
    var = np.asarray(var, dtype=float)
    if var.size == 0:
        return None

    total = var.sum()
    pct = 100.0 * var / total if total > 0 else np.zeros_like(var)
    cum = np.cumsum(pct)
    x = np.arange(1, len(var) + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True, sharex=True,
    )
    ax1.bar(x, pct, color=PALETTE[0], alpha=0.85)
    ax1.set_xlabel("PC index")
    ax1.set_ylabel("variance explained (%)")
    ax1.set_title("Individual (pre-ICA PCA)", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.plot(x, cum, marker="o", ms=3, color=PALETTE[1])
    ax2.fill_between(x, 0, cum, color=PALETTE[1], alpha=0.15)
    for thr, col in zip([90, 95, 99], PALETTE[2:5]):
        ax2.axhline(thr, ls="--", color=col, lw=1.0, alpha=0.7, label=f"{thr}%")
    ax2.set_xlabel("PC index")
    ax2.set_ylabel("cumulative variance (%)")
    ax2.set_title("Cumulative", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle("Stage 2 - PCA variance explained (pre-ICA, MNE branch)",
                 fontsize=12)

    out_path = out_dir / "stage2_pca_variance_explained.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5.1b PSD de componentes pre-ICA
# ---------------------------------------------------------------------------

def plot_pre_ica_component_psds(
    raw,
    meta_stage2: dict,
    out_dir: Path,
    fmin: float = 1.0,
    fmax: float = 40.0,
    n_components_show: int = 10,
    dpi: int = 150,
) -> Path | None:
    """
    Espectros de potencia (Welch) de los componentes PCA whitened pre-ICA.

    Solo aplica cuando ``branch == "mne_ica_iclabel"``.
    """
    plt = setup_plotting_style()
    from scipy.signal import welch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("branch") != "mne_ica_iclabel":
        return None

    sfreq = float(raw.info["sfreq"])

    # Localizar componentes whitened
    pcs = meta_stage2.get("pre_ica_components")
    if pcs is None:
        pcs = meta_stage2.get("whitened_components")
    if pcs is None:
        sub = meta_stage2.get("preprocessing") or {}
        pcs = sub.get("Y")
    if pcs is None:
        return None
    pcs = np.asarray(pcs, dtype=float)
    if pcs.ndim != 2:
        return None
    D, T = pcs.shape
    n_show = min(n_components_show, D)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for i in range(n_show):
        # nperseg ~ 4 * sfreq, capped a T
        nperseg = int(min(4 * sfreq, T))
        nperseg = max(64, nperseg)
        freqs, psd = welch(pcs[i], fs=sfreq, nperseg=nperseg)
        mask = (freqs >= fmin) & (freqs <= fmax)
        ax.semilogy(freqs[mask], psd[mask], lw=0.8,
                    color=PALETTE[i % len(PALETTE)],
                    alpha=0.85, label=f"PC{i+1}")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_title(f"Pre-ICA component PSDs (first {n_show} of {D})",
                 fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    out_path = out_dir / "stage2_pre_ica_component_psds.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5.1c Resumen ICLabel
# ---------------------------------------------------------------------------

def plot_icalabel_summary(
    meta_stage2: dict,
    out_dir: Path,
    dpi: int = 150,
) -> Path | None:
    """
    Bar chart horizontal con conteo de componentes por clase ICLabel.

    Solo aplica cuando ``branch == "mne_ica_iclabel"``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("branch") != "mne_ica_iclabel":
        return None

    labels_info = meta_stage2.get("labels")
    if labels_info is None:
        labels_info = {}
    # labels_info puede ser dict {"labels": [...], "y_pred_proba": [...]}
    # o directamente una lista de strings
    if isinstance(labels_info, dict):
        labels = list(labels_info.get("labels", []))
    elif isinstance(labels_info, (list, tuple, np.ndarray)):
        labels = list(labels_info)
    else:
        return None
    if not labels:
        return None

    # Conteo por clase
    classes = ["brain", "eye", "muscle", "heart", "other", "line_noise"]
    counts = {c: 0 for c in classes}
    for lab in labels:
        lab_str = str(lab).lower()
        # tolerancia: "brain", "Brain", etc.
        if lab_str in counts:
            counts[lab_str] += 1
        else:
            counts["other"] += 1

    excluded = meta_stage2.get("excluded_indices", [])
    if excluded is None:
        excluded = []
    n_total = len(labels)
    n_excluded = len(excluded) if isinstance(excluded, (list, tuple, np.ndarray)) else 0
    n_kept = n_total - n_excluded

    # Filtrar clases con count > 0 para no dibujar barras vacias
    present = [(c, counts[c]) for c in classes if counts[c] > 0]
    if not present:
        return None
    cls_arr, cnt_arr = zip(*present)
    colors = [ICALABEL_COLORS[c] for c in cls_arr]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    y_pos = np.arange(len(cls_arr))
    bars = ax.barh(y_pos, cnt_arr, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cls_arr)
    ax.set_xlabel("number of components")
    ax.set_title(
        f"ICLabel summary  |  total={n_total}  kept={n_kept}  "
        f"excluded={n_excluded}",
        fontsize=11,
    )
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)

    # Anotar conteos en cada barra
    for bar, c in zip(bars, cnt_arr):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{c}", va="center", fontsize=9)

    out_path = out_dir / "stage2_icalabel_summary.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5.1d PSD Post-ICA de componentes retenidos
# ---------------------------------------------------------------------------

def plot_post_ica_component_psds(
    meta_stage2: dict,
    sfreq: float,
    out_dir: Path,
    fmin: float = 1.0,
    fmax: float = 40.0,
    dpi: int = 150,
) -> Path | None:
    """
    PSD (Welch) de los componentes ICA retenidos + overlay pre vs post.

    Solo aplica cuando ``branch == "mne_ica_iclabel"``.
    """
    plt = setup_plotting_style()
    from scipy.signal import welch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("branch") != "mne_ica_iclabel":
        return None

    Y_post = meta_stage2.get("Y")
    if Y_post is None:
        sub = meta_stage2.get("preprocessing") or {}
        Y_post = sub.get("Y")
    if Y_post is None:
        return None
    Y_post = np.asarray(Y_post, dtype=float)
    if Y_post.ndim != 2:
        return None

    # Tambien intentar recuperar pre-ICA para overlay
    Y_pre = meta_stage2.get("pre_ica_components")
    if Y_pre is None:
        Y_pre = meta_stage2.get("whitened_components")
    if Y_pre is None:
        sub = meta_stage2.get("preprocessing") or {}
        Y_pre = sub.get("pre_ica_components")

    D, T = Y_post.shape
    nperseg = int(min(4 * sfreq, T))
    nperseg = max(64, nperseg)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True,
    )

    # --- Panel 1: PSDs individuales post-ICA ---
    psds_post = []
    for i in range(D):
        freqs, psd = welch(Y_post[i], fs=sfreq, nperseg=nperseg)
        psds_post.append(psd)
    psds_post = np.array(psds_post)
    mask = (freqs >= fmin) & (freqs <= fmax)

    for i in range(D):
        ax1.semilogy(freqs[mask], psds_post[i, mask], lw=0.8,
                     color=PALETTE[i % len(PALETTE)], alpha=0.85,
                     label=f"IC{i+1}")
    ax1.set_xlabel("frequency (Hz)")
    ax1.set_ylabel("PSD (a.u.)")
    ax1.set_title(f"Post-ICA retained components ({D})", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)
    if D <= 8:
        ax1.legend(loc="upper right", fontsize=8)

    # --- Panel 2: overlay media pre vs post ---
    mean_post = psds_post.mean(axis=0)
    ax2.semilogy(freqs[mask], mean_post[mask], lw=1.6,
                 color=PALETTE[3], label="post-ICA (mean)")
    if Y_pre is not None:
        Y_pre = np.asarray(Y_pre, dtype=float)
        psds_pre = []
        for i in range(min(Y_pre.shape[0], 30)):  # cap por coste
            _, psd = welch(Y_pre[i], fs=sfreq, nperseg=nperseg)
            psds_pre.append(psd)
        if psds_pre:
            mean_pre = np.array(psds_pre).mean(axis=0)
            ax2.semilogy(freqs[mask], mean_pre[mask], lw=1.6, ls="--",
                         color=PALETTE[0], label="pre-ICA (mean)")
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("PSD (a.u.)")
    ax2.set_title("Pre vs Post ICA (mean)", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Stage 2 - Post-ICA component PSDs (MNE branch)",
                 fontsize=12)

    out_path = out_dir / "stage2_post_ica_component_psds.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# 5.2 — pca_ica, rama Hankel (branch="svd_whitening_fastica_sklearn")
# ===========================================================================

# ---------------------------------------------------------------------------
# 5.2a Valores singulares del SVD previo al FastICA
# ---------------------------------------------------------------------------

def plot_hankel_pca_singular_values(
    meta_stage2: dict,
    out_dir: Path,
    n_components: int | None = None,
    dpi: int = 150,
) -> Path | None:
    """
    Lineas+puntos de los valores singulares del SVD previo al FastICA.

    Solo aplica cuando ``branch == "svd_whitening_fastica_sklearn"``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("branch") != "svd_whitening_fastica_sklearn":
        return None

    s = meta_stage2.get("singular_values")
    if s is None:
        return None
    s = np.asarray(s, dtype=float).ravel()
    if s.size == 0:
        return None

    n_sv = len(s)
    x = np.arange(1, n_sv + 1)
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
    ax.set_title(
        f"Stage 2 - SVD pre-FastICA (Hankel branch)  |  n={n_sv}",
        fontsize=11,
    )
    if n_components is not None and 0 < n_components <= n_sv:
        ax.axvline(n_components, ls="--", color=PALETTE[3], lw=1.2,
                   label=f"n_components={n_components}")
        ax.legend(loc="upper right")
    elif "svd_rank" in meta_stage2:
        rank = meta_stage2["svd_rank"]
        if isinstance(rank, int) and 0 < rank <= n_sv:
            ax.axvline(rank, ls="--", color=PALETTE[3], lw=1.2,
                       label=f"svd_rank={rank}")
            ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    out_path = out_dir / "stage2_hankel_pca_singular_values.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5.2b Convergencia del ICA
# ---------------------------------------------------------------------------

def plot_fastica_convergence(
    meta_stage2: dict,
    out_dir: Path,
    dpi: int = 150,
) -> Path | None:
    """
    Bar chart de ``n_iter_`` vs ``max_iter=1000`` (verde=convergio).

    Solo aplica cuando ``branch == "svd_whitening_fastica_sklearn"``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("branch") != "svd_whitening_fastica_sklearn":
        return None

    n_iter = meta_stage2.get("n_iter_")
    max_iter = meta_stage2.get("max_iter", 1000)
    if max_iter is None:
        max_iter = 1000
    max_iter = int(max_iter)
    converged = bool(meta_stage2.get("converged", True))

    if n_iter is None:
        return None
    n_iter = int(n_iter)

    color = PALETTE[2] if converged else PALETTE[3]   # green / red

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    bars = ax.bar(["n_iter", "max_iter"], [n_iter, max_iter],
                  color=[color, "#888888"], alpha=0.85)
    ax.set_ylabel("iterations")
    ax.set_title(
        f"FastICA convergence  |  {'CONVERGED' if converged else 'NOT CONVERGED'}"
        f"  |  n_iter={n_iter}/{max_iter}",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, v in zip(bars, [n_iter, max_iter]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_iter * 0.02,
                f"{v}", ha="center", fontsize=10)

    out_path = out_dir / "stage2_fastica_convergence.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# 5.3 — DMD
# ===========================================================================

# ---------------------------------------------------------------------------
# 5.3a Eigenvalores DMD en circulo unitario
# ---------------------------------------------------------------------------

def plot_dmd_eigenvalue_unit_circle(
    meta_stage2: dict,
    out_dir: Path,
    dpi: int = 150,
) -> Path | None:
    """
    Eigenvalores DMD en el plano complejo + circulo unitario.

    Color = frecuencia; tamano = |valor singular asociado|.

    Solo aplica cuando ``dynamics == "dmd"``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("dynamics") != "dmd":
        return None

    eig = meta_stage2.get("eigenvalues")
    if eig is None:
        return None
    eig = np.asarray(eig)
    if eig.size == 0:
        return None

    freqs = meta_stage2.get("frequencies_hz")
    # Avoid `or` with numpy arrays (ambiguous truth value)
    amps = meta_stage2.get("amplitudes")
    if amps is None:
        amps = meta_stage2.get("singular_values")

    re = np.real(eig)
    im = np.imag(eig)

    # Color: frecuencia absoluta
    if freqs is not None and len(freqs) == len(eig):
        c = np.abs(np.asarray(freqs, dtype=float))
    else:
        c = np.abs(eig)
    # Tamano: amplitud
    if amps is not None and len(amps) == len(eig):
        a = np.asarray(amps, dtype=float)
        s = 20 + 80 * (a / max(a.max(), 1e-12))
    else:
        s = 40 * np.ones_like(re)

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    # Circulo unitario
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), ls="--", color="#888888",
            lw=1.0, label="unit circle")

    sc = ax.scatter(re, im, c=c, s=s, cmap="plasma", alpha=0.85,
                    edgecolors="black", linewidths=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|frequency| (Hz)" if freqs is not None else "|eigenvalue|")

    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.axvline(0, color="#cccccc", lw=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_aspect("equal", "box")
    ax.set_title("DMD eigenvalues on unit circle", fontsize=11)
    ax.grid(True, alpha=0.3)

    out_path = out_dir / "stage2_dmd_eigenvalue_unit_circle.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5.3b Espectro de frecuencias y tasas de amortiguamiento
# ---------------------------------------------------------------------------

def plot_dmd_frequency_damping(
    meta_stage2: dict,
    out_dir: Path,
    dpi: int = 150,
) -> Path | None:
    """
    Dos paneles: barras de frecuencia (coloreadas por damping) y scatter
    frecuencia vs damping.

    Solo aplica cuando ``dynamics == "dmd"``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("dynamics") != "dmd":
        return None

    freqs = meta_stage2.get("frequencies_hz")
    damp = meta_stage2.get("damping_rates")
    if freqs is None or damp is None:
        return None
    freqs = np.asarray(freqs, dtype=float)
    damp = np.asarray(damp, dtype=float)
    if freqs.size == 0 or freqs.size != damp.size:
        return None

    n = len(freqs)
    idx = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), constrained_layout=True,
    )

    # --- Panel superior: barras de frecuencia coloreadas por damping ---
    sc1 = ax1.bar(idx, np.abs(freqs),
                  color=plt.cm.coolwarm(
                      (damp - damp.min()) / max(damp.max() - damp.min(), 1e-12)
                  ),
                  alpha=0.9)
    ax1.set_xlabel("mode index")
    ax1.set_ylabel("|frequency| (Hz)")
    ax1.set_title("DMD mode frequencies (color = damping)", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm",
        norm=plt.Normalize(vmin=damp.min(), vmax=damp.max()),
    )
    cbar1 = fig.colorbar(sm, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("damping (1/s)")

    # --- Panel inferior: scatter frecuencia vs damping ---
    sc2 = ax2.scatter(damp, np.abs(freqs), c=np.abs(freqs), cmap="plasma",
                      s=60, edgecolors="black", linewidths=0.5, alpha=0.85)
    ax2.axvline(0, ls="--", color="#888888", lw=1.0, label="purely oscillatory")
    ax2.set_xlabel("damping rate (1/s)")
    ax2.set_ylabel("|frequency| (Hz)")
    ax2.set_title("Frequency vs damping", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Stage 2 - DMD frequency & damping spectrum", fontsize=12)

    out_path = out_dir / "stage2_dmd_frequency_damping_spectrum.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# 5.4 — Diffusion Maps
# ===========================================================================

# ---------------------------------------------------------------------------
# 5.4a Spectrum de eigenvalores de difusion
# ---------------------------------------------------------------------------

def plot_diffusion_eigenvalue_spectrum(
    meta_stage2: dict,
    out_dir: Path,
    dpi: int = 150,
) -> Path | None:
    """
    Barras descendentes de |lambda_i| con anotacion del spectral gap.

    Solo aplica cuando ``dynamics == "diffusion_maps"``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("dynamics") != "diffusion_maps":
        return None

    eig = meta_stage2.get("eigenvalues_latent")
    if eig is None:
        return None
    eig = np.asarray(eig, dtype=float)
    eig = np.sort(np.abs(eig))[::-1]
    if eig.size < 2:
        return None

    x = np.arange(1, len(eig) + 1)
    gap = meta_stage2.get("spectral_gap")
    if gap is None:
        gap = float(eig[0] - eig[1]) if eig.size >= 2 else 0.0
    gap = float(gap)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x, eig, color=PALETTE[0], alpha=0.85)
    # Banda sombreada entre lambda_1 y lambda_2
    if eig.size >= 2:
        ax.axhspan(eig[1], eig[0], color=PALETTE[3], alpha=0.15,
                   label=f"spectral gap = {gap:.4f}")
    ax.set_xlabel("index (1-based, non-trivial)")
    ax.set_ylabel("|λ|")
    ax.set_title("Diffusion maps - eigenvalue spectrum", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    out_path = out_dir / "stage2_diffusion_eigenvalue_spectrum.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5.4b Diagnosticos del kernel
# ---------------------------------------------------------------------------

def plot_diffusion_kernel_diagnostics(
    meta_stage2: dict,
    dm_input_data: np.ndarray | None,
    out_dir: Path,
    dpi: int = 150,
) -> Path | None:
    """
    Histograma de distancias k-NN promedio + linea vertical en sigma.

    Solo aplica cuando ``dynamics == "diffusion_maps"``.
    Si ``dm_input_data`` es None o la reconstruccion falla, retorna None.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if meta_stage2.get("dynamics") != "diffusion_maps":
        return None

    sigma = meta_stage2.get("sigma_used")
    if sigma is None:
        sigma = meta_stage2.get("epsilon_fitted")
    k = meta_stage2.get("k_neighbors", 100)
    if k is None:
        k = 100
    k = int(k)

    if dm_input_data is None or sigma is None:
        return None

    X = np.asarray(dm_input_data, dtype=float)
    if X.ndim != 2:
        return None
    # Convencion pipeline: (n_features, n_samples). sklearn k-NN quiere
    # (n_samples, n_features).
    if X.shape[0] > X.shape[1]:
        Xs = X.T
    else:
        Xs = X

    # Limitar a 5000 muestras para no reventar memoria
    if Xs.shape[0] > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(Xs.shape[0], 5000, replace=False)
        Xs = Xs[idx]

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k + 1, Xs.shape[0]))
        nn.fit(Xs)
        dists, _ = nn.kneighbors(Xs)
        # Promediar distancia al vecino k-esimo (excluyendo uno mismo)
        mean_kdist = dists[:, 1:].mean(axis=0)
    except Exception:
        return None

    sigma = float(sigma)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    # Histograma de distancia media al k-esimo vecino por punto
    ax.hist(dists[:, min(k, dists.shape[1] - 1)],
            bins=50, color=PALETTE[0], alpha=0.7,
            label=f"distances to k={k} neighbor")
    ax.axvline(sigma, ls="--", color=PALETTE[3], lw=2,
               label=f"σ = {sigma:.4g}")
    ax.set_xlabel("distance")
    ax.set_ylabel("count")
    ax.set_title("Diffusion kernel diagnostics - k-NN distances vs σ",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    out_path = out_dir / "stage2_diffusion_kernel_diagnostics.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 5.4c Embedding 2D de Difusion (pares de componentes)
# ---------------------------------------------------------------------------

def plot_diffusion_2d_components(
    meta: dict,
    out_dir: Path,
    n_components_show: int = 6,
    dpi: int = 150,
) -> Path | None:
    """
    Scatter plots en pares de los primeros componentes de difusion,
    coloreados por tiempo.

    Solo aplica cuando ``meta["stage2"]["dynamics"] == "diffusion_maps"``.
    """
    plt = setup_plotting_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage2 = meta.get("stage2", {})
    if stage2.get("dynamics") != "diffusion_maps":
        return None

    Y = meta.get("Y")
    if Y is None:
        return None
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        return None
    D, T = Y.shape
    n_show = min(n_components_show, D)
    if n_show < 2:
        return None

    # Transponer a (T, D)
    Yt = Y.T

    # Pares a plotear
    pairs = list(combinations(range(n_show), 2))
    # Limitar numero de pares para no reventar la figura
    if len(pairs) > 6:
        pairs = pairs[:6]
    n_pairs = len(pairs)

    n_cols = min(3, n_pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    t_axis = np.arange(T)
    for k, (i, j) in enumerate(pairs):
        ax = axes[k]
        sc = ax.scatter(Yt[:, i], Yt[:, j], c=t_axis, cmap="viridis",
                        s=2, alpha=0.5)
        ax.set_xlabel(f"ψ{i+1}")
        ax.set_ylabel(f"ψ{j+1}")
        ax.set_title(f"ψ{i+1} vs ψ{j+1}", fontsize=9)
        ax.grid(True, alpha=0.3)

    for k in range(n_pairs, len(axes)):
        axes[k].axis("off")

    cbar = fig.colorbar(sc, ax=axes[:n_pairs], fraction=0.02, pad=0.02)
    cbar.set_label("time index")

    fig.suptitle(
        "Diffusion maps - 2D embedding (pairs of components)",
        fontsize=12,
    )

    out_path = out_dir / "stage2_diffusion_all_components_2d.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path

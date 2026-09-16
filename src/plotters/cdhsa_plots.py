from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    pass  # stage2_meta is a plain dict


# Colour palette
# ---------------
COLOR_SPECIFIC = "#e74c3c"     # red  — specific (top-N from Step D)
COLOR_COMMON = "#2980b9"       # blue — common (r0 from Step A6)
COLOR_SPECIFIC_DARK = "#c0392b"
COLOR_COMMON_DARK = "#1f618d"


# ---------------------------------------------------------------------------
# Plot 1: Eigenvalue spectrum — common (A6) vs specific (D) modes
# ---------------------------------------------------------------------------

def plot_cdhsa_eigenvalue_spectrum(
    stage2_meta: dict,
    out_dir: str | Path,
) -> Path | None:
    """
    Bar chart showing eigenvalues of **both** the common and specific modes.

    * **Blue bars (left group)** — common modes from Step A6:
      ``r0`` directions with eigenvalues ``lambda0``.
      These are the first SVD directions that passed the
      null-distribution significance test.

    * **Red bars (right group)** — specific modes from Step D
      for the target condition: ``r_c`` total, of which ``top_n``
      are selected (darker red outline / annotation).
      These come from the residual after projecting out the
      common subspace.

    The two groups are separated by a vertical dashed line.
    """
    common_eigs = stage2_meta.get("common_eigenvalues")
    r0 = stage2_meta.get("r0_common")
    n_common = stage2_meta.get("n_common_modes", 0)

    all_spec_eigs = stage2_meta.get("all_condition_eigenvalues")
    selected_indices = set(stage2_meta.get("mode_indices", []))
    condition = stage2_meta.get("condition", "unknown")
    top_n = stage2_meta.get("top_n", 0)
    rc = stage2_meta.get("total_specific_modes", 0)

    # Need at least one group
    if (not common_eigs and not all_spec_eigs):
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the data
    common_eigs = common_eigs or []
    all_spec_eigs = all_spec_eigs or []
    n_c = len(common_eigs)
    n_s = len(all_spec_eigs)
    total = n_c + n_s

    if total == 0:
        return None

    fig, ax = plt.subplots(
        figsize=(max(8, total * 1.0), 5.5), constrained_layout=True,
    )

    x = np.arange(total)
    colors = []
    edge_colors = []
    edge_widths = []

    # Common modes (blue)
    for i in range(n_c):
        colors.append(COLOR_COMMON)
        edge_colors.append("black")
        edge_widths.append(0.5)

    # Specific modes: selected=top_n in solid red, rest in lighter red
    for i in range(n_s):
        if i in selected_indices:
            colors.append(COLOR_SPECIFIC)
            edge_colors.append(COLOR_SPECIFIC_DARK)
            edge_widths.append(1.5)
        else:
            colors.append("#f1948a")  # lighter red for non-selected specific
            edge_colors.append("black")
            edge_widths.append(0.5)

    all_vals = common_eigs + all_spec_eigs
    bars = ax.bar(x, all_vals, color=colors,
                  edgecolor=edge_colors, linewidth=edge_widths)

    # Annotate common modes (blue)
    for i in range(n_c):
        ax.annotate(
            f"{common_eigs[i]:.4f}",
            xy=(i, common_eigs[i]),
            xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=7.5, color=COLOR_COMMON_DARK,
        )

    # Annotate specific modes — selected in bold, rest in italic
    for i in range(n_s):
        idx = n_c + i
        is_sel = i in selected_indices
        ax.annotate(
            f"{all_spec_eigs[i]:.4f}",
            xy=(idx, all_spec_eigs[i]),
            xytext=(0, 8), textcoords="offset points",
            ha="center",
            fontsize=9 if is_sel else 7.5,
            fontweight="bold" if is_sel else "normal",
            fontstyle="normal" if is_sel else "italic",
            color=COLOR_SPECIFIC_DARK if is_sel else "#922b21",
        )

    # Vertical separator between common and specific
    if n_c > 0 and n_s > 0:
        ax.axvline(n_c - 0.5, color="gray", linewidth=1.0,
                    linestyle="--", alpha=0.7)
        ax.text(n_c / 2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
                "COMMON\n(A6)", ha="center", va="top",
                fontsize=9, color=COLOR_COMMON_DARK, fontweight="bold",
                transform=ax.get_xaxis_transform())
        ax.text(n_c + n_s / 2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.5,
                "SPECIFIC (D)", ha="center", va="top",
                fontsize=9, color=COLOR_SPECIFIC_DARK, fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xlabel("Mode index", fontsize=11)
    ax.set_ylabel("Eigenvalue (\u03bb)", fontsize=11)
    ax.set_title(
        f"CD-HSA Eigenvalue Spectrum \u2014 {condition}\n"
        f"Common (A6): r0={n_common} | Specific (D): {top_n} selected of {rc} total",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(x)
    # Label x-ticks with group prefix
    xtick_labels = []
    for i in range(n_c):
        xtick_labels.append(f"C{i}")
    for i in range(n_s):
        marker = "*" if i in selected_indices else ""
        xtick_labels.append(f"S{i}{marker}")
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_COMMON, edgecolor="black",
              label=f"Common (A6, r0={n_common})"),
        Patch(facecolor=COLOR_SPECIFIC, edgecolor=COLOR_SPECIFIC_DARK,
              linewidth=1.5,
              label=f"Specific selected (top-{top_n})"),
        Patch(facecolor="#f1948a", edgecolor="black",
              label=f"Specific not selected ({rc - top_n})"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

    path = out_dir / "stage2_cdhsa_eigenvalue_spectrum.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 2: Mode structure — common (A6) vs specific (D) spatial distribution
# ---------------------------------------------------------------------------

def plot_cdhsa_mode_structure(
    stage2_meta: dict,
    out_dir: str | Path,
) -> Path | None:
    """
    Two-row figure comparing common and specific mode structures:

    **Top row** — common modes (A6, U0): lag activation profile + heatmap.
    **Bottom row** — specific modes (D, W_sel): lag activation profile + heatmap.

    Each panel reveals the temporal structure captured by each mode:
    which lag blocks and channel-delay combinations contribute most.

    If common modes are not available (no A6 data), only the specific
    row is shown.
    """
    W_sel = stage2_meta.get("W_sel")
    W_common = stage2_meta.get("W_common")

    if W_sel is None and W_common is None:
        return None

    L = stage2_meta.get("L")
    p_hankel = stage2_meta.get("p_hankel")
    if L is None or p_hankel is None:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    condition = stage2_meta.get("condition", "unknown")
    r0 = stage2_meta.get("r0_common", 0)
    common_eigs = stage2_meta.get("common_eigenvalues", [])
    spec_eigs_all = stage2_meta.get("all_condition_eigenvalues", [])
    selected_indices = stage2_meta.get("mode_indices", [])
    n_common = stage2_meta.get("n_common_modes", 0)

    has_common = (W_common is not None and W_common.shape[1] > 0)
    has_specific = (W_sel is not None and W_sel.shape[1] > 0)

    # Determine grid
    n_rows = int(has_common) + int(has_specific)
    if n_rows == 0:
        return None

    fig, axes = plt.subplots(
        n_rows, 2, figsize=(14, 5 * n_rows), constrained_layout=True,
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    lags = np.arange(L)
    max_cols = 100
    row_idx = 0

    # ---- Helper ----
    def _draw_group(ax_pair, W_group, eig_list, title_prefix, color,
                    mode_prefix):
        n_modes = W_group.shape[1]
        # W_group is (p, n_modes) — need to reshape to (L, p/L, n_modes)
        # for block-Hankel mode structure
        p_total = W_group.shape[0]
        p_per_lag = p_total // L
        if p_per_lag * L != p_total:
            # Not evenly divisible; show as-is without lag decomposition
            p_per_lag = p_total
            W_3d = W_group[np.newaxis, :, :]  # (1, p, n_modes)
            lag_profile = np.mean(np.abs(W_3d), axis=1)  # (1, n_modes)
            use_lag = False
        else:
            W_3d = W_group.reshape(L, p_per_lag, n_modes)  # (L, p/L, n_modes)
            lag_profile = np.mean(np.abs(W_3d), axis=1)  # (L, n_modes)
            use_lag = True

        # Panel 1: Lag activation profile
        ax1 = ax_pair[0]
        n_lag = lag_profile.shape[0]
        lag_x = np.arange(n_lag)
        for m in range(n_modes):
            ev_str = (f" \u03bb={eig_list[m]:.4f}"
                      if m < len(eig_list) else "")
            label = f"{mode_prefix}{m}" + ev_str
            ax1.bar(
                lag_x + m * 0.8 / n_modes,
                lag_profile[:, m],
                width=0.7 / n_modes,
                label=label,
                alpha=0.85,
                color=color,
                edgecolor="black", linewidth=0.3,
            )
        ax1.set_xlabel("Block-Hankel lag" if use_lag else "Block",
                       fontsize=10)
        ax1.set_ylabel("Mean |W| per lag block", fontsize=10)
        ax1.set_title(f"{title_prefix} \u2014 Lag Activation",
                       fontsize=11, fontweight="bold")
        if n_modes <= 12:
            ax1.legend(fontsize=7, loc="best")

        # Panel 2: Heatmap of first mode
        ax2 = ax_pair[1]
        W_mode = W_3d[:, :, 0]
        if p_per_lag > max_cols:
            step = max(1, p_per_lag // max_cols)
            W_show = W_mode[:, ::step]
        else:
            W_show = W_mode
            step = 1

        vmax = np.percentile(np.abs(W_show), 98)
        im = ax2.imshow(
            W_show, aspect="auto", cmap="RdBu_r",
            vmin=-vmax, vmax=vmax, interpolation="nearest",
        )
        xlabel = ("Hankel feature dim (downsampled)"
                  if step > 1 else "Hankel feature dim")
        ax2.set_xlabel(xlabel, fontsize=9)
        ax2.set_ylabel("Block-Hankel lag" if use_lag else "Feature",
                       fontsize=9)
        ax2.set_title(f"{mode_prefix}0 spatial structure",
                       fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax2, shrink=0.8, label="W weight")

    # ---- Top row: common modes (A6) ----
    if has_common:
        _draw_group(
            axes[row_idx], W_common,
            common_eigs if common_eigs else [],
            title_prefix=f"Common Modes (A6, r0={n_common})",
            color=COLOR_COMMON,
            mode_prefix="Com.",
        )
        row_idx += 1

    # ---- Bottom row: specific modes (D) ----
    if has_specific:
        sel_eigs = ([spec_eigs_all[i] for i in selected_indices]
                    if spec_eigs_all else [])
        _draw_group(
            axes[row_idx], W_sel,
            sel_eigs,
            title_prefix=f"Specific Modes (D, top-{W_sel.shape[1]})",
            color=COLOR_SPECIFIC,
            mode_prefix="Sp.",
        )

    fig.suptitle(
        f"CD-HSA Mode Structure \u2014 {condition} (L={L}, p={p_hankel})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    path = out_dir / "stage2_cdhsa_mode_structure.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plot 3: Projection power — common vs specific norms & eigenvalues
# ---------------------------------------------------------------------------

def plot_cdhsa_projection_power(
    stage2_meta: dict,
    Y2: np.ndarray | None = None,
    out_dir: str | Path | None = None,
) -> Path | None:
    """
    Side-by-side comparison of **common** (A6) vs **specific** (D) modes:

    * **Left panel** — mode vector norms (||W||) for both groups.
    * **Right panel** — eigenvalues for both groups.

    Common modes are shown in blue, specific modes in red.
    """
    mode_norms = stage2_meta.get("mode_norms")
    common_mode_norms = stage2_meta.get("common_mode_norms")
    selected_eigs = stage2_meta.get("eigenvalues")
    common_eigs = stage2_meta.get("common_eigenvalues")
    condition = stage2_meta.get("condition", "unknown")
    top_n = stage2_meta.get("top_n", 0)
    rc = stage2_meta.get("total_specific_modes", 0)
    n_common = stage2_meta.get("n_common_modes", 0)

    has_spec_norms = mode_norms is not None and len(mode_norms) > 0
    has_com_norms = (common_mode_norms is not None
                     and len(common_mode_norms) > 0)
    has_spec_eigs = selected_eigs is not None and len(selected_eigs) > 0
    has_com_eigs = common_eigs is not None and len(common_eigs) > 0

    if not (has_spec_norms or has_com_norms or has_spec_eigs or has_com_eigs):
        return None

    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # ---- Panel 1: Mode norms ----
    ax1 = axes[0]
    bar_width = 0.35

    if has_com_norms or has_spec_norms:
        n_com = len(common_mode_norms) if has_com_norms else 0
        n_spec = len(mode_norms) if has_spec_norms else 0
        max_modes = max(n_com, n_spec, 1)

        if has_com_norms:
            x_c = np.arange(n_com)
            bars_c = ax1.bar(
                x_c - bar_width / 2, common_mode_norms, bar_width,
                color=COLOR_COMMON, edgecolor="black", linewidth=0.5,
                label=f"Common (r0={n_com})",
            )
            for bar, norm in zip(bars_c, common_mode_norms):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height(),
                         f"{norm:.1f}", ha="center", va="bottom",
                         fontsize=7.5, color=COLOR_COMMON_DARK)

        if has_spec_norms:
            x_s = np.arange(n_spec)
            bars_s = ax1.bar(
                x_s + bar_width / 2, mode_norms, bar_width,
                color=COLOR_SPECIFIC, edgecolor="black", linewidth=0.5,
                label=f"Specific (top-{n_spec})",
            )
            for bar, norm in zip(bars_s, mode_norms):
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height(),
                         f"{norm:.1f}", ha="center", va="bottom",
                         fontsize=8, color=COLOR_SPECIFIC_DARK,
                         fontweight="bold")

        all_idx = list(range(max(n_com, n_spec)))
        ax1.set_xticks(all_idx)
        ax1.set_xticklabels([f"Mode {i}" for i in all_idx])
        ax1.set_xlabel("Mode", fontsize=11)
        ax1.set_ylabel("||W_mode|| (Frobenius norm)", fontsize=11)
        ax1.set_title("Mode Vector Norms", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9, loc="best")
    else:
        ax1.text(0.5, 0.5, "No norm data available",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=11, color="gray")
        ax1.set_title("Mode Vector Norms", fontsize=12)

    # ---- Panel 2: Eigenvalues ----
    ax2 = axes[1]

    if has_com_eigs or has_spec_eigs:
        n_com_e = len(common_eigs) if has_com_eigs else 0
        n_spec_e = len(selected_eigs) if has_spec_eigs else 0

        if has_com_eigs:
            x_c = np.arange(n_com_e)
            bars_c2 = ax2.bar(
                x_c - bar_width / 2, common_eigs, bar_width,
                color=COLOR_COMMON, edgecolor="black", linewidth=0.5,
                label=f"Common (r0={n_com_e})",
            )
            for bar, ev in zip(bars_c2, common_eigs):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height(),
                         f"{ev:.4f}", ha="center", va="bottom",
                         fontsize=7.5, color=COLOR_COMMON_DARK)

        if has_spec_eigs:
            x_s = np.arange(n_spec_e)
            bars_s2 = ax2.bar(
                x_s + bar_width / 2, selected_eigs, bar_width,
                color=COLOR_SPECIFIC, edgecolor="black", linewidth=0.5,
                label=f"Specific (top-{n_spec_e})",
            )
            for bar, ev in zip(bars_s2, selected_eigs):
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height(),
                         f"{ev:.4f}", ha="center", va="bottom",
                         fontsize=8, color=COLOR_SPECIFIC_DARK,
                         fontweight="bold")

        all_eig_idx = list(range(max(n_com_e, n_spec_e)))
        ax2.set_xticks(all_eig_idx)
        ax2.set_xticklabels([f"Mode {i}" for i in all_eig_idx])
        ax2.set_ylabel("Eigenvalue (\u03bb)", fontsize=11)
        ax2.set_xlabel("Mode", fontsize=11)
        ax2.set_title("Mode Eigenvalues", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9, loc="best")
    else:
        # Fallback: variance of Y2 rows
        if Y2 is not None:
            variances = np.var(Y2, axis=1)
            ax2.bar(range(len(variances)), variances, color="#2ecc71",
                    edgecolor="black", linewidth=0.5)
            ax2.set_ylabel("Variance of mode time course", fontsize=11)
            ax2.set_title("Mode Time-Course Variance (fallback)",
                           fontsize=12, fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "No eigenvalue data available",
                     transform=ax2.transAxes, ha="center", va="center",
                     fontsize=11, color="gray")
            ax2.set_title("Mode Eigenvalues", fontsize=12)

    fig.suptitle(
        f"CD-HSA Projection Power \u2014 {condition}\n"
        f"Common (A6): r0={n_common} | Specific (D): {top_n} of {rc}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    path = out_dir / "stage2_cdhsa_projection_power.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

"""
src.plotters — Modulo de ploteo del pipeline EEG
==================================================

Re-exporta todas las funciones publicas de los submodulos para que el
script principal pueda importar con una sola linea::

    from src.plotters import (
        plot_channel_topographies,
        plot_channel_correlation_matrix,
        ...
    )

Estructura
----------
* ``_style.py``            - Configuracion global de matplotlib.
* ``_markov_helpers.py``   - Helper: compute_transition_matrix.
* ``stage0_exploratory.py``  - Topografias, correlacion.
* ``stage1_embedding.py``    - Valores singulares, varianza Hankel.
* ``stage2_dynamics.py``     - PCA / ICLabel / DMD / Diffusion.
* ``stage3_selection.py``    - Markov tau heatmap, ranked, distribution,
                               transition matrix.
* ``latent_detail.py``       - Zoom de series temporales.
* ``pipeline_overview.py``   - Energy budget, flowchart.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Configuracion global (side-effect: aplica rcParams al importar)
# ---------------------------------------------------------------------------
from src.plotters._style import (
    setup_plotting_style,
    PALETTE,
    ICALABEL_COLORS,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    DEFAULT_FIGSIZE,
    DEFAULT_DPI,
)

# ---------------------------------------------------------------------------
# Stage 0 — Exploratorio
# ---------------------------------------------------------------------------
from src.plotters.stage0_exploratory import (
    plot_channel_topographies,
    plot_channel_correlation_matrix,
)

# ---------------------------------------------------------------------------
# Stage 1 — Embedding Hankel
# ---------------------------------------------------------------------------
from src.plotters.stage1_embedding import (
    plot_hankel_singular_values,
    plot_hankel_variance_explained,
)

# ---------------------------------------------------------------------------
# Stage 2 — Dinamica
# ---------------------------------------------------------------------------
from src.plotters.stage2_dynamics import (
    plot_pca_variance_explained,
    plot_pre_ica_component_psds,
    plot_icalabel_summary,
    plot_post_ica_component_psds,
    plot_hankel_pca_singular_values,
    plot_fastica_convergence,
    plot_dmd_eigenvalue_unit_circle,
    plot_dmd_frequency_damping,
    plot_diffusion_eigenvalue_spectrum,
    plot_diffusion_kernel_diagnostics,
    plot_diffusion_2d_components,
)

# ---------------------------------------------------------------------------
# Stage 3 — Seleccion Markov
# ---------------------------------------------------------------------------
from src.plotters.stage3_selection import (
    plot_markov_tau_heatmap,
    plot_markov_tau_ranked,
    plot_markov_selection_vs_distribution,
    plot_markov_transition_matrix,
)

# ---------------------------------------------------------------------------
# Latent detail
# ---------------------------------------------------------------------------
from src.plotters.latent_detail import (
    plot_latent_timeseries_zoom,
)

# ---------------------------------------------------------------------------
# Pipeline overview (cross-stage)
# ---------------------------------------------------------------------------
from src.plotters.pipeline_overview import (
    plot_pipeline_energy_budget,
    plot_pipeline_flowchart,
)

from src.plotters.structural_graph_plots import (
    load_structural_matrix,
    plot_structural_graph_influence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from src.plotters._markov_helpers import compute_transition_matrix

from src.plotters.trajectory_plots import plot_latent_trajectory


__all__ = [
    # Style
    "setup_plotting_style",
    "PALETTE",
    "ICALABEL_COLORS",
    "DIVERGING_CMAP",
    "SEQUENTIAL_CMAP",
    "DEFAULT_FIGSIZE",
    "DEFAULT_DPI",
    # Stage 0
    "plot_channel_topographies",
    "plot_channel_correlation_matrix",
    # Stage 1
    "plot_hankel_singular_values",
    "plot_hankel_variance_explained",
    # Stage 2
    "plot_pca_variance_explained",
    "plot_pre_ica_component_psds",
    "plot_icalabel_summary",
    "plot_post_ica_component_psds",
    "plot_hankel_pca_singular_values",
    "plot_fastica_convergence",
    "plot_dmd_eigenvalue_unit_circle",
    "plot_dmd_frequency_damping",
    "plot_diffusion_eigenvalue_spectrum",
    "plot_diffusion_kernel_diagnostics",
    "plot_diffusion_2d_components",
    # Stage 3
    "plot_markov_tau_heatmap",
    "plot_markov_tau_ranked",
    "plot_markov_selection_vs_distribution",
    "plot_markov_transition_matrix",
    # Latent detail
    "plot_latent_timeseries_zoom",
    # Pipeline overview
    "plot_pipeline_energy_budget",
    "plot_pipeline_flowchart",
    # Structural graph
    "load_structural_matrix",
    "plot_structural_graph_influence",
    # Helpers
    "compute_transition_matrix",
    "plot_latent_trajectory"
]

"""Módulo de análisis espectral de potencia (PSD) para pipelines EEG.

Uso::

    from src.spectral_analysis.psd_analysis import (
        compute_and_plot_raw_psd,
        compute_and_plot_latent_psd,
        compute_channel_influence_on_latent,
        load_psd_data,
    )
"""

from .psd_analysis import (
    compute_and_plot_latent_psd,
    compute_and_plot_raw_psd,
    compute_channel_influence_on_latent,
    load_psd_data,
)

__all__ = [
    "compute_and_plot_raw_psd",
    "compute_and_plot_latent_psd",
    "compute_channel_influence_on_latent",
    "load_psd_data",
]

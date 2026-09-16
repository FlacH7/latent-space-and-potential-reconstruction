"""
Modular 3-stage latent-space extraction pipeline.

Stages
------
* :mod:`stage1_embedding`  — optional delay embedding (None / Hankel).
* :mod:`stage2_dynamics`   — dynamics analysis (PCA, ICA, DMD, Diffusion Maps).
* :mod:`stage3_selection`  — mode selection (top-n, Markov fastest/slowest).
"""

from src.latent_space_extraction.pipeline.stage1_embedding import (
    PipelineContext,
    IdentityEmbedding,
    HankelEmbedding,
    STAGE1_REGISTRY,
    build_stage1,
)
from src.latent_space_extraction.pipeline.stage2_dynamics import (
    PCADynamics,
    PCAICADynamics,
    DMDDynamics,
    DiffusionMapsDynamics,
    STAGE2_REGISTRY,
    build_stage2,
)
from src.latent_space_extraction.pipeline.stage3_selection import (
    TopNSelection,
    MarkovFastestSelection,
    MarkovSlowestSelection,
    LegacyFCSelection,
    STAGE3_REGISTRY,
    build_stage3,
)

__all__ = [
    "PipelineContext",
    "IdentityEmbedding",
    "HankelEmbedding",
    "PCADynamics",
    "PCAICADynamics",
    "DMDDynamics",
    "DiffusionMapsDynamics",
    "TopNSelection",
    "MarkovFastestSelection",
    "MarkovSlowestSelection",
    "LegacyFCSelection",
    "STAGE1_REGISTRY",
    "STAGE2_REGISTRY",
    "STAGE3_REGISTRY",
    "build_stage1",
    "build_stage2",
    "build_stage3",
]

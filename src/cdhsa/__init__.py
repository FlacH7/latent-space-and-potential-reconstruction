"""CD-HSA: Common-Differential Hankel Subspace Analysis
=====================================================

Python implementation of the CD-HSA framework for finding common and
differential modes in PCA/SSA of EEG data across subjects and conditions.

Pipeline:
    A (A1-A6): Common subspace estimation and rank selection
    B:        Energy condition effects
    C:        Geometry condition effects (alignment + tangent)
    D:        Condition-specific residual modes

Step-by-step usage::

    from src.cdhsa import cdhsa_A1_A5, cdhsa_A6_common_rank, cdhsa_BC_condition_tests
    R = cdhsa_A1_A5(X, L, fixed_rank=10)
    A6 = cdhsa_A6_common_rank(R)
    BC = cdhsa_BC_condition_tests(X, L, R, A6)

Or via the pipeline in src.pipelines.run_cdhsa::

    from src.pipelines.run_cdhsa import run_cdhsa, CDHSAConfig
    cfg = CDHSAConfig(fixed_rank=10)
    result = run_cdhsa(X, L, cfg)
"""

from src.cdhsa.a_common_subspace import (
    build_block_hankel,
    truncated_left_svd,
    truncated_left_svd_with_values,
    select_rank_reproducibility,
    cdhsa_A1_A5,
)
from src.cdhsa.a6_common_rank import (
    common_basis_from_U,
    crossvalidate_common_rank,
    cdhsa_A6_common_rank,
)
from src.cdhsa.null_distributions import (
    random_subspace_null,
    hankel_preserving_null,
)
from src.cdhsa.permutation_tests import (
    within_subject_permutation_rm,
)
from src.cdhsa.b_energy import (
    compute_common_mode_metrics,
    cdhsa_BC_condition_tests,
)
from src.cdhsa.c_geometry import (
    cdhsa_tangent_geometry_test,
)
from src.cdhsa.d_condition_specific import (
    cdhsa_D_condition_specific_modes,
)

__all__ = [
    # A1-A5
    "build_block_hankel",
    "truncated_left_svd",
    "truncated_left_svd_with_values",
    "select_rank_reproducibility",
    "cdhsa_A1_A5",
    # A6
    "common_basis_from_U",
    "crossvalidate_common_rank",
    "cdhsa_A6_common_rank",
    # Nulls
    "random_subspace_null",
    "hankel_preserving_null",
    # Permutation
    "within_subject_permutation_rm",
    # B/C
    "compute_common_mode_metrics",
    "cdhsa_BC_condition_tests",
    # C tangent
    "cdhsa_tangent_geometry_test",
    # D
    "cdhsa_D_condition_specific_modes",
]

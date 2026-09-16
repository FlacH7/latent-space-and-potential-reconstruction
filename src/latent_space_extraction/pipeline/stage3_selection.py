"""
Stage 3 — Mode Selection
========================

Third stage of the modular latent-space pipeline.  It takes the Stage-2
output matrix ``Y`` of shape ``(D, T')`` — real candidate latent time
series — and extracts the final ``n_dim`` modes.

Variants
--------
* ``"top_n"``          — simple ranking: keep the first ``n_dim`` rows
  (modes are already ordered by the Stage-2 decomposition: singular
  values, |frequency|, diffusion eigenvalues, ...).
* ``"markov_fastest"`` — subspace with the *smallest* Markov relaxation
  time τ (``find_best_subspace_markov(..., maximize=False)``).
* ``"markov_slowest"`` — subspace with the *largest* τ
  (``find_best_subspace_markov(..., maximize=True)``).

Backward-compatibility extension (beyond the base spec)
-------------------------------------------------------
``"legacy_fc"`` dispatches the pre-refactor conservative-fraction based
strategies (``conservative`` / ``weighted`` / ``sequential`` / ``pareto``
/ ``independent``) so that the legacy CLI keeps working end-to-end.  It
lazily imports ``scoring.py`` and ``conservative_fraction.py`` so that the
new 3-stage pipeline has no hard dependency on them.

Shape convention
----------------
Input ``Y``: ``(D, T')``.  Output: ``(n_dim, T')`` — the orchestrator
transposes it to ``(n_samples, n_dim)`` at the very end.
"""

from __future__ import annotations

import time

import numpy as np

from src.latent_space_extraction.markov_subspace import (
    find_best_subspace_markov,
    greedy_forward_selection_markov,
)

from src.latent_space_extraction.pipeline.stage1_embedding import PipelineContext


# ---------------------------------------------------------------------------
# Top-N
# ---------------------------------------------------------------------------

class TopNSelection:
    """
    Simple top-n ranking: keep the first ``n_dim`` rows of Y.

    Stage-2 outputs are already ordered by relevance (singular value,
    diffusion eigenvalue, |frequency|, ...), so no further criterion is
    applied.
    """

    name = "top_n"

    def __init__(self, **params):
        if params:
            raise ValueError(
                f"[Stage3/top_n] takes no parameters, got {params}"
            )

    def fit_transform(
        self,
        Y: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
        **params,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        D = Y.shape[0]
        if n_dim > D:
            raise ValueError(
                f"[Stage3/top_n] n_dim ({n_dim}) exceeds the number of "
                f"Stage-2 modes D ({D})."
            )
        selected = tuple(range(n_dim))
        out = Y[:n_dim, :]
        meta = {
            "selection": "top_n",
            "selected_indices": list(selected),
            "input_shape": tuple(Y.shape),
            "output_shape": tuple(out.shape),
            "scores": {},
            "elapsed_time": time.time() - t0,
        }
        return out, meta


# ---------------------------------------------------------------------------
# Markov fastest / slowest
# ---------------------------------------------------------------------------

class MarkovSelection:
    """
    Markovian subspace selection over the Stage-2 mode matrix.

    Discretises each row into ``n_bins`` quantile bins and searches
    (exhaustively or greedily) for the ``n_dim``-component subspace with
    the smallest (``markov_fastest``) or largest (``markov_slowest``)
    relaxation time τ, via
    :func:`markov_subspace.find_best_subspace_markov` /
    :func:`markov_subspace.greedy_forward_selection_markov`.

    Parameters (via ``stage3_params``)
    ----------------------------------
    n_bins : int, default 5
        Quantile bins per component.
    search_strategy : "exhaustive" | "greedy", default "exhaustive"
        ``"greedy"`` is forced automatically when ``n_dim >= 4`` (legacy
        rule).
    maximize : bool
        ``False`` → fastest dynamics (min τ); ``True`` → slowest (max τ).
        Set internally by the registry name.
    """

    name = "markov"

    def __init__(
        self,
        n_bins: int = 5,
        search_strategy: str = "exhaustive",
        maximize: bool = False,
    ):
        if search_strategy not in ("exhaustive", "greedy"):
            raise ValueError(
                f"[Stage3/markov] search_strategy must be 'exhaustive' or "
                f"'greedy', got {search_strategy!r}"
            )
        self.n_bins = int(n_bins)
        self.search_strategy = search_strategy
        self.maximize = bool(maximize)

    def fit_transform(
        self,
        Y: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
        **params,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        if np.iscomplexobj(Y):
            # Empate 5.1 — defensive: Stage-2 already returns real outputs.
            Y = np.real(Y)

        D = Y.shape[0]
        if n_dim > D:
            raise ValueError(
                f"[Stage3/markov] n_dim ({n_dim}) exceeds the number of "
                f"Stage-2 modes D ({D})."
            )

        use_greedy = self.search_strategy == "greedy" or n_dim >= 4

        if use_greedy:
            comb, tau = greedy_forward_selection_markov(
                Y, n_dim, n_bins=self.n_bins, maximize=self.maximize,
            )
            comb = tuple(comb)
            all_taus = []  # No disponible en greedy por ahora
        else:
            result = find_best_subspace_markov(
                Y, n_dim,
                n_bins=self.n_bins,
                n_workers=ctx.n_workers,
                maximize=self.maximize,
                return_all_taus=True,
            )
            comb, tau, all_taus = result
            comb = tuple(comb)
            # Normalizar all_taus a lista de (tuple, float)
            all_taus = [(tuple(c), float(t)) for c, t in all_taus]

        if comb is None:
            raise RuntimeError(
                "[Stage3/markov] No valid (ergodic) subspace found: every "
                "combination returned tau = inf."
            )

        out = Y[list(comb), :]
        meta = {
            "selection": "markov_slowest" if self.maximize else "markov_fastest",
            "selected_indices": list(comb),
            "input_shape": tuple(Y.shape),
            "output_shape": tuple(out.shape),
            "scores": {
                "tau": float(tau),
                "n_bins": self.n_bins,
                "search": "greedy" if use_greedy else "exhaustive",
                "maximize": self.maximize,
            },
            "all_markov_taus": all_taus,  # NUEVO: lista de (comb, tau)
            "elapsed_time": time.time() - t0,
        }
        return out, meta


class MarkovFastestSelection(MarkovSelection):
    """Subspace with the *smallest* Markov relaxation time (fastest)."""

    name = "markov_fastest"

    def __init__(self, **params):
        params.pop("maximize", None)
        super().__init__(maximize=False, **params)


class MarkovSlowestSelection(MarkovSelection):
    """Subspace with the *largest* Markov relaxation time (slowest)."""

    name = "markov_slowest"

    def __init__(self, **params):
        params.pop("maximize", None)
        super().__init__(maximize=True, **params)


# ---------------------------------------------------------------------------
# Legacy conservative-fraction strategies (backward compatibility only)
# ---------------------------------------------------------------------------

class LegacyFCSelection:
    """
    Backward-compatibility shim for the pre-refactor scoring methods that
    combine the conservative fraction (fc) with the Markov time:

    ``"conservative"``, ``"weighted"``, ``"sequential"``, ``"pareto"`` and
    ``"independent"``.

    This class is **not** part of the new 3-stage API; it exists only so
    that the legacy CLI (``--scoring-method ...``) keeps working while the
    old strategies are phased out.  It lazily imports ``scoring.py`` and
    ``conservative_fraction.py``.

    Parameters (via ``stage3_params``)
    ----------------------------------
    method : str — one of the five legacy methods above.
    fc_metric, alpha, primary_criterion, sequential_K, n_bins,
    search_strategy, n_workers — legacy parameters, same semantics as
    before the refactor.
    """

    name = "legacy_fc"

    _VALID = ("conservative", "weighted", "sequential", "pareto", "independent")

    def __init__(
        self,
        method: str,
        fc_metric: str = "variance_sum",
        alpha: float | None = None,
        primary_criterion: str = "markov",
        sequential_K: int | None = None,
        n_bins: int = 5,
        search_strategy: str = "exhaustive",
    ):
        if method not in self._VALID:
            raise ValueError(
                f"[Stage3/legacy_fc] method must be one of {self._VALID}, "
                f"got {method!r}"
            )
        self.method = method
        self.fc_metric = fc_metric
        self.alpha = alpha
        self.primary_criterion = primary_criterion
        self.sequential_K = sequential_K
        self.n_bins = n_bins
        self.search_strategy = search_strategy

    def fit_transform(
        self,
        Y: np.ndarray,
        *,
        ctx: PipelineContext,
        n_dim: int,
        **params,
    ) -> tuple[np.ndarray, dict]:
        t0 = time.time()
        n_workers = ctx.n_workers
        use_greedy = self.search_strategy == "greedy" or n_dim >= 4

        # Lazy imports — the new pipeline must not hard-depend on these.
        from src.latent_space_extraction.conservative_fraction import (
            find_best_subspace_fc,
            greedy_forward_selection_fc,
        )
        from src.latent_space_extraction.scoring import (
            independent_selection,
            pareto_frontier_selection,
            sequential_filtering_selection,
            weighted_score_selection,
        )

        method = self.method

        if method == "conservative":
            if use_greedy:
                comb, fc = greedy_forward_selection_fc(Y, n_dim, metric=self.fc_metric)
                comb = tuple(comb)
            else:
                comb, fc = find_best_subspace_fc(
                    Y, n_dim, metric=self.fc_metric, n_workers=n_workers,
                )
            scores = {"fc": float(fc), "metric": self.fc_metric,
                      "search": "greedy" if use_greedy else "exhaustive"}

        elif method == "weighted":
            if self.alpha is None:
                raise ValueError(
                    "[Stage3/legacy_fc] alpha is required for method='weighted'"
                )
            result = weighted_score_selection(
                Y, n_dim,
                alphas=[self.alpha],
                metric=self.fc_metric,
                n_bins=self.n_bins,
                n_workers=n_workers,
                search_strategy="greedy" if use_greedy else "exhaustive",
            )
            comb = result["best_combinations"][0]
            scores = {
                "alpha": float(self.alpha),
                "score": float(result["best_scores"][0]),
                "fc": float(result["fc_values"][0]),
                "tau": float(result["tau_values"][0]),
            }

        elif method == "sequential":
            D = Y.shape[0]
            K = self.sequential_K
            if K is None:
                import math
                K = min(20, max(5, int(
                    np.prod([D - i for i in range(n_dim)]) / math.factorial(n_dim) * 0.05
                )))
                K = max(K, 5)
            result = sequential_filtering_selection(
                Y, n_dim, K=K, primary=self.primary_criterion,
                metric=self.fc_metric, n_bins=self.n_bins, n_workers=n_workers,
            )
            comb = result["best_combination"]
            scores = {
                "primary": self.primary_criterion,
                "K": K,
                "fc": result["best_fc"],
                "tau": result["best_tau"],
            }

        elif method == "pareto":
            result = pareto_frontier_selection(
                Y, n_dim, metric=self.fc_metric, n_bins=self.n_bins,
                n_workers=n_workers,
            )
            comb = result["frontier_combinations"][0]
            scores = {
                "frontier_size": int(result["frontier_size"]),
                "fc": float(result["frontier_fc"][0]),
                "tau": float(result["frontier_tau"][0]),
            }

        else:  # independent
            result = independent_selection(
                Y, n_dim,
                metric=self.fc_metric,
                n_bins=self.n_bins,
                n_workers=n_workers,
                fc_search="greedy" if use_greedy else "exhaustive",
                markov_search="greedy" if use_greedy else "exhaustive",
            )
            # Return the Markov winner — it discriminates better than FC
            comb = result["markov_combination"]
            scores = {
                "fc_combination": result["fc_combination"],
                "fc_value": result["fc_value"],
                "markov_combination": result["markov_combination"],
                "markov_tau": result["markov_tau"],
            }

        out = Y[list(comb), :]
        meta = {
            "selection": f"legacy_fc:{method}",
            "selected_indices": list(comb),
            "input_shape": tuple(Y.shape),
            "output_shape": tuple(out.shape),
            "scores": scores,
            "elapsed_time": time.time() - t0,
        }
        return out, meta


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

STAGE3_REGISTRY: dict[str, type] = {
    "top_n": TopNSelection,
    "markov_fastest": MarkovFastestSelection,
    "markov_slowest": MarkovSlowestSelection,
    "legacy_fc": LegacyFCSelection,  # backward compatibility only
}


def build_stage3(name: str, params: dict | None = None):
    """
    Instantiate a Stage-3 selection class by name.

    Raises
    ------
    ValueError
        If the selection name is unknown.
    """
    params = dict(params or {})
    if name not in STAGE3_REGISTRY:
        valid = ", ".join(repr(k) for k in STAGE3_REGISTRY)
        raise ValueError(
            f"[Stage3] Unknown selection {name!r}. Valid options: {valid}"
        )
    return STAGE3_REGISTRY[name](**params)

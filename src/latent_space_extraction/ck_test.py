"""
Chapman-Kolmogorov Test for Markovianity in Latent Space
==========================================================

Tests whether a low-dimensional trajectory satisfies the Chapman-Kolmogorov
equation, a necessary condition for the process to be Markovian.

The CK equation states that for a Markov process the transition probability
can be composed::

    T(tau) = T(tau/2) @ T(tau/2)

If this does not hold, the process has memory beyond the current state and
the Kramers-Moyal expansion may not be valid.

Memory optimisation (v2)
-------------------------
With D=4 and n_bins=20 the full state space has 160 000 states.  A dense
(160k, 160k) float64 matrix requires ~205 GB which is infeasible.  This
module applies three complementary strategies:

1. **Occupied-state remapping**  — only the ~10-16k states actually visited
   are kept, shrinking the matrix to (n_occ, n_occ).
2. **Sparse matrices (scipy CSR)** — even after remapping most rows have
   far fewer non-zero entries than n_occ.
3. **Power-iteration for pi** — replaces ``np.linalg.eig`` (O(n^3) dense)
   with a fast sparse-friendly fixed-point iteration.
4. **No T_matrices hoarding** — transition matrices are not stored by
   default (opt-in via ``store_T_matrices=True``).

Typical usage from the IGA pipeline::

    from ck_test import chapman_kolmogorov_test

    ck_result = chapman_kolmogorov_test(
        data,               # latent trajectory, shape (n_samples, n_dim)
        dt=dt,              # sampling interval [s]
        tau_candidates=None,  # auto-generated lags
        n_bins=20,          # discretisation bins per dimension
        threshold=0.15,     # acceptance threshold for CK error
        plot=True,
        out_dir=out_dir,
    )

    if ck_result["is_markovian"]:
        print(f"  Markovian at tau* = {ck_result['tau_star']:.4f} s")
    else:
        print(f"  NOT Markovian — min error: {ck_result['min_error']:.4f}")
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Sequence, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs as sparse_eigs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_N_BINS: int = 20
DEFAULT_THRESHOLD: float = 0.15
DEFAULT_TAU_MULTIPLIERS: Sequence[int] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16,
                                             20, 25, 32, 40, 50, 64, 80, 100,
                                             128, 160, 200, 256)


# ---------------------------------------------------------------------------
# Discretisation (unchanged)
# ---------------------------------------------------------------------------

def _discretise_uniform(
    data: np.ndarray,
    n_bins: int,
    pad: float = 0.05,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Discretise each dimension of *data* into *n_bins* uniform-width bins.
    """
    D = data.shape[1]
    edges = []
    digitised = np.empty((data.shape[0], D), dtype=int)

    for d in range(D):
        dmin, dmax = data[:, d].min(), data[:, d].max()
        margin = pad * (dmax - dmin) if dmax > dmin else 1.0
        e = np.linspace(dmin - margin, dmax + margin, n_bins + 1)
        edges.append(e)
        digitised[:, d] = np.digitize(data[:, d], e) - 1
        digitised[:, d] = np.clip(digitised[:, d], 0, n_bins - 1)

    # Mixed-radix encoding
    multipliers = np.cumprod([1] + [n_bins] * (D - 1))
    states = np.sum(digitised * multipliers, axis=1)

    return states, edges


def _discretise_by_quantiles(
    data: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Discretise each dimension of *data* into *n_bins* quantile-based bins.
    Guarantees balanced occupancy.
    """
    D = data.shape[1]
    edges = []
    digitised = np.empty((data.shape[0], D), dtype=int)

    for d in range(D):
        e = np.quantile(data[:, d], np.linspace(0, 1, n_bins + 1))
        e[-1] += 1e-12
        edges.append(e)
        digitised[:, d] = np.digitize(data[:, d], e) - 1
        digitised[:, d] = np.clip(digitised[:, d], 0, n_bins - 1)

    multipliers = np.cumprod([1] + [n_bins] * (D - 1))
    states = np.sum(digitised * multipliers, axis=1)

    return states, edges


# ---------------------------------------------------------------------------
# Occupied-state remapping
# ---------------------------------------------------------------------------

def _remap_to_occupied(
    states: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], int]:
    """
    Remap state indices to a compact 0..n_occupied-1 range.

    Parameters
    ----------
    states : ndarray, shape (n_samples,)
        Original (possibly sparse) state indices.

    Returns
    -------
    states_compact : ndarray, shape (n_samples,)
        Remapped state indices in [0, n_occupied).
    inv_map : dict[int, int]
        Mapping  compact_index -> original_state_id.
    n_occupied : int
        Number of distinct states visited.
    """
    unique_states = np.unique(states)
    n_occupied = len(unique_states)
    # Vectorised remap: build a lookup table sized to max(states)+1
    max_state = int(states.max()) + 1
    lut = np.full(max_state, -1, dtype=np.int64)
    lut[unique_states] = np.arange(n_occupied, dtype=np.int64)
    states_compact = lut[states]

    # inv_map: compact -> original (for potential downstream use)
    inv_map = {int(i): int(s) for i, s in enumerate(unique_states)}

    return states_compact, inv_map, n_occupied


# ---------------------------------------------------------------------------
# Sparse transition matrix estimation
# ---------------------------------------------------------------------------

def _estimate_transition_matrix_sparse(
    states: np.ndarray,
    lag: int,
    n_states: int,
) -> sparse.csr_matrix:
    """
    Estimate the row-normalised transition matrix T^{(lag)} using sparse
    counting.  Returns a CSR matrix of shape (n_states, n_states).
    """
    n_samples = len(states)
    src = states[:n_samples - lag]
    dst = states[lag:]

    # Build count matrix via COO -> CSR (memory-efficient)
    flat_idx = src.astype(np.int64) * np.int64(n_states) + dst.astype(np.int64)
    # Count transitions using bincount on compact flat indices
    # But flat_idx can be large, so use sparse directly
    counts = sparse.coo_matrix(
        (np.ones(len(src), dtype=np.float64), (src, dst)),
        shape=(n_states, n_states),
    ).tocsr()

    # Row-normalise
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    # Avoid division by zero for unvisited rows
    row_sums[row_sums == 0] = 1.0
    # Invert for division
    inv_sums = sparse.diags(1.0 / row_sums)
    T = inv_sums @ counts

    return T


# ---------------------------------------------------------------------------
# Stationary distribution via power iteration (sparse-friendly)
# ---------------------------------------------------------------------------

def _stationary_distribution_power(
    T: sparse.csr_matrix,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Estimate the stationary distribution pi of a transition matrix T
    using the power method on T^T.

    This avoids the O(n^3) dense eigendecomposition and works directly
    on sparse matrices.
    """
    n = T.shape[0]
    pi = np.ones(n, dtype=np.float64) / n

    Tt = T.T.tocsr()

    for _ in range(max_iter):
        pi_new = pi @ Tt  # equivalent to Tt.T @ pi, but row vector
        pi_new_sum = pi_new.sum()
        if pi_new_sum == 0:
            break
        pi_new = pi_new / pi_new_sum
        # Check convergence (L1 norm of change)
        delta = np.abs(pi_new - pi).sum()
        pi = pi_new
        if delta < tol:
            break

    # Ensure non-negative and normalised
    pi = np.abs(pi)
    pi_sum = pi.sum()
    if pi_sum > 0:
        pi = pi / pi_sum
    else:
        pi = np.ones(n, dtype=np.float64) / n

    return pi


# ---------------------------------------------------------------------------
# CK error: population-weighted Frobenius norm (sparse-aware)
# ---------------------------------------------------------------------------

def _ck_error_weighted_sparse(
    T_tau: sparse.csr_matrix,
    T_half: sparse.csr_matrix,
    pi: np.ndarray | None = None,
) -> float:
    """
    Population-weighted CK error using sparse matrices.

    Parameters
    ----------
    T_tau : sparse.csr_matrix, shape (n, n)
    T_half : sparse.csr_matrix, shape (n, n)
    pi : ndarray, shape (n,), optional
        Stationary distribution.  If None, estimated via power iteration.

    Returns
    -------
    error : float in [0, 1]
    """
    # Compose: T_half @ T_half (sparse @ sparse = sparse)
    T_composed = T_half @ T_half

    # Diff: sparse - sparse = sparse
    diff = T_tau - T_composed

    if pi is None:
        pi = _stationary_distribution_power(T_tau)

    # Weight by sqrt(pi): convert diff and T_tau to dense for the
    # Frobenius norm computation (the dense matrix is now n_occ x n_occ,
    # which is manageable)
    sqrt_pi = np.sqrt(pi)

    # Convert to dense for norm computation
    # n_occ is typically ~16k, so dense is ~2 GB — acceptable
    diff_dense = diff.toarray()
    T_tau_dense = T_tau.toarray()

    weighted_diff = diff_dense * sqrt_pi[:, None]
    weighted_T = T_tau_dense * sqrt_pi[:, None]

    numerator = np.linalg.norm(weighted_diff, ord="fro")
    denominator = np.linalg.norm(weighted_T, ord="fro")

    del diff_dense, T_tau_dense, diff, T_composed
    gc.collect()

    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chapman_kolmogorov_test(
    data: np.ndarray,
    *,
    dt: float,
    tau_candidates: Sequence[float] | np.ndarray | None = None,
    n_bins: int = DEFAULT_N_BINS,
    discretisation: Literal["uniform", "quantile"] = "uniform",
    threshold: float = DEFAULT_THRESHOLD,
    top_k_states: int | None = None,
    min_occupancy: float = 0.3,
    plot: bool = True,
    out_dir: str | Path | None = None,
    verbose: bool = True,
    store_T_matrices: bool = False,
    max_samples_for_tm: int | None = None,
) -> dict:
    """
    Test whether a low-dimensional trajectory is Markovian via the
    Chapman-Kolmogorov equation.

    For each candidate lag *tau*, the function:
    1. Discretises the state space into bins (uniform or quantile).
    2. Estimates T(tau) and T(tau/2) on the **occupied-state subspace**.
    3. Compares T(tau) with T(tau/2) @ T(tau/2).
    4. Reports the population-weighted Frobenius-norm error.

    The process is declared Markovian if the CK error falls below
    *threshold* for some lag.  The smallest such lag is tau*.

    Parameters
    ----------
    data : ndarray, shape (n_samples, D)
        Latent trajectory (continuous).  Typically D=2 or 3.
    dt : float
        Sampling interval in seconds.
    tau_candidates : sequence or ndarray, optional
        Candidate lags *in seconds*.  If None, a geometric progression
        of multipliers is applied to *dt* (see DEFAULT_TAU_MULTIPLIERS).
    n_bins : int, default 20
        Number of bins per dimension.
    discretisation : ``"uniform"`` | ``"quantile"``, default ``"uniform"``
        Binning strategy.
    threshold : float, default 0.15
        CK error threshold for declaring Markovianity.
    top_k_states : int or None, optional
        (Unused in v2; kept for API compatibility.)
    min_occupancy : float, default 0.3
        Minimum fraction of bins that must be occupied.
    plot : bool, default True
        Whether to generate the CK error vs tau plot.
    out_dir : str or Path, optional
        Directory to save the plot.
    verbose : bool, default True
        Print progress and results.
    store_T_matrices : bool, default False
        If True, store all transition matrices in the returned dict.
        **WARNING**: with n_occ ~ 16k each dense matrix is ~2 GB;
        storing 22 of them requires ~44 GB.  Use with caution.
    max_samples_for_tm : int or None, optional
        If set, subsample the trajectory to at most this many samples
        for transition matrix estimation (speed + memory).  The full
        trajectory is still used for discretisation edges.  None = no
        subsampling.

    Returns
    -------
    result : dict
        Same keys as the original implementation, plus:

        * ``n_occupied``      — int, number of occupied states used.
        * ``memory_mode``     — str, ``"sparse_remapped"``.
        * ``transition_matrices`` — dict (only if *store_T_matrices* is True).
    """
    t0 = time.time()

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_samples, D = data.shape

    if verbose:
        print("\n" + "=" * 70)
        print("  CHAPMAN-KOLMOGOROV MARKOVIANITY TEST  (memory-optimised)")
        print("=" * 70)
        print(f"  Data shape      : {data.shape}")
        print(f"  dt              : {dt:.6f} s  ({1/dt:.1f} Hz)")
        print(f"  Duration        : {n_samples * dt:.1f} s")
        print(f"  Bins per dim    : {n_bins}")
        print(f"  Total states    : {n_bins ** D}")
        print(f"  Discretisation  : {discretisation}")
        print(f"  CK threshold    : {threshold}")

    # ------------------------------------------------------------------
    # 1. Discretise
    # ------------------------------------------------------------------
    if discretisation == "uniform":
        states, edges = _discretise_uniform(data, n_bins)
    else:
        states, edges = _discretise_by_quantiles(data, n_bins)

    n_full_states = n_bins ** D
    all_unique = np.unique(states)
    n_occupied = len(all_unique)
    occupancy = n_occupied / n_full_states

    if verbose:
        print(f"\n  Occupied bins   : {n_occupied}/{n_full_states} ({occupancy:.1%})")

    if occupancy < min_occupancy:
        print(f"  [WARN] Occupancy {occupancy:.1%} < {min_occupancy} — "
              f"consider reducing n_bins (current: {n_bins})")

    # ------------------------------------------------------------------
    # 1b. Remap to occupied states (KEY OPTIMISATION)
    # ------------------------------------------------------------------
    if verbose:
        print(f"  Remapping {n_full_states} states -> {n_occupied} occupied states...")

    states_compact, inv_map, n_occ = _remap_to_occupied(states)
    del states  # free the original large-index array
    gc.collect()

    if verbose:
        mem_dense = n_full_states ** 2 * 8 / 1e9
        mem_compact = n_occ ** 2 * 8 / 1e9
        print(f"  Dense T would be : {mem_dense:.1f} GB  ({n_full_states}x{n_full_states})")
        print(f"  Compact T size   : {mem_compact:.1f} GB  ({n_occ}x{n_occ})")
        print(f"  Reduction factor : {mem_dense / mem_compact:.0f}x")

    # ------------------------------------------------------------------
    # 1c. Optional subsampling for transition counting
    # ------------------------------------------------------------------
    states_for_tm = states_compact
    if max_samples_for_tm is not None and len(states_compact) > max_samples_for_tm:
        if verbose:
            print(f"  Subsampling {len(states_compact)} -> {max_samples_for_tm} for TM estimation")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(states_compact), size=max_samples_for_tm, replace=False)
        idx = np.sort(idx)  # keep temporal order for transition counting
        states_for_tm = states_compact[idx]
        del idx
        gc.collect()

    # ------------------------------------------------------------------
    # 2. Candidate lags
    # ------------------------------------------------------------------
    if tau_candidates is None:
        tau_steps_arr = np.array(DEFAULT_TAU_MULTIPLIERS, dtype=int)
        tau_candidates = tau_steps_arr * dt
    else:
        tau_candidates = np.asarray(tau_candidates, dtype=float)

    n_tm_samples = len(states_for_tm)
    max_tau_samples = n_tm_samples // 3
    max_tau = max_tau_samples * dt
    valid_mask = tau_candidates <= max_tau
    tau_values = tau_candidates[valid_mask]

    if len(tau_values) == 0:
        raise ValueError(
            f"No tau candidates fit in data. "
            f"Max feasible tau: {max_tau:.4f} s "
            f"({max_tau_samples} samples)."
        )

    tau_steps = (tau_values / dt).astype(int)
    tau_steps = np.maximum(tau_steps, 1)

    if verbose:
        print(f"\n  Testing {len(tau_values)} tau values:")
        print(f"    min = {tau_values[0]:.4f} s ({tau_steps[0]} steps)")
        print(f"    max = {tau_values[-1]:.4f} s ({tau_steps[-1]} steps)")

    # ------------------------------------------------------------------
    # 3. Compute CK error for each tau
    # ------------------------------------------------------------------
    ck_errors = np.full(len(tau_values), np.nan)
    T_matrices = {} if store_T_matrices else None

    if verbose:
        print(f"\n  {'Tau (s)':>10}  {'Steps':>7}  {'CK error':>12}  {'Status'}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*8}")
        sys.stdout.flush()

    for i, (tau_s, lag) in enumerate(zip(tau_values, tau_steps)):
        iter_t0 = time.time()

        T_tau = _estimate_transition_matrix_sparse(states_for_tm, lag, n_occ)
        half_lag = max(lag // 2, 1)
        T_half = _estimate_transition_matrix_sparse(states_for_tm, half_lag, n_occ)

        err = _ck_error_weighted_sparse(T_tau, T_half)
        ck_errors[i] = err

        if store_T_matrices:
            T_matrices[float(tau_s)] = T_tau.toarray()

        # Free memory aggressively
        del T_tau, T_half
        gc.collect()

        status = "OK" if err < threshold else "---"
        iter_time = time.time() - iter_t0
        if verbose:
            print(f"  {tau_s:10.4f}  {lag:7d}  {err:12.6f}  {status}  ({iter_time:.1f}s)")
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # 4. Determine tau*
    # ------------------------------------------------------------------
    min_err_idx = int(np.nanargmin(ck_errors))
    min_error = float(ck_errors[min_err_idx])
    min_error_tau = float(tau_values[min_err_idx])

    below_threshold = ck_errors < threshold
    if np.any(below_threshold):
        tau_star_idx = int(np.argmax(below_threshold))
        tau_star = float(tau_values[tau_star_idx])
        is_markovian = True
    else:
        tau_star_idx = -1
        tau_star = float("inf")
        is_markovian = False

    # ------------------------------------------------------------------
    # 5. Veredicto
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  {'='*50}")
        print(f"  RESULTS")
        print(f"  {'='*50}")
        print(f"  Minimum CK error : {min_error:.6f}  (at tau = {min_error_tau:.4f} s)")
        if is_markovian:
            print(f"  tau*             : {tau_star:.4f} s  ({int(tau_star/dt)} steps)")
            print(f"  CK error at tau* : {ck_errors[tau_star_idx]:.6f}  <  {threshold}")
            print(f"  VERDICT          : MARKOVIAN  (use tau* for KM)")
        else:
            print(f"  tau*             : INF  (no tau satisfies threshold)")
            print(f"  Best CK error    : {min_error:.6f}  >=  {threshold}")
            print(f"  VERDICT          : NOT MARKOVIAN")
            print(f"  Recommendation   : Increase latent dimension or embedding depth")
        print(f"  Elapsed time     : {elapsed:.1f} s")
        print(f"  {'='*50}")

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    fig = None
    if plot:
        fig = _plot_ck_results(
            tau_values=tau_values,
            ck_errors=ck_errors,
            threshold=threshold,
            tau_star=tau_star if is_markovian else None,
            min_error_tau=min_error_tau,
            min_error=min_error,
            is_markovian=is_markovian,
            dt=dt,
            D=D,
            n_bins=n_bins,
            occupancy=occupancy,
            discretisation=discretisation,
        )
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / "ck_test.png", dpi=150)
            plt.close(fig)
            if verbose:
                print(f"\n  [PLOT] Saved CK test plot to: {out_dir / 'ck_test.png'}")

    result = {
        "is_markovian": is_markovian,
        "tau_star": tau_star,
        "tau_star_idx": tau_star_idx,
        "min_error": min_error,
        "min_error_tau": min_error_tau,
        "tau_values": tau_values,
        "ck_errors": ck_errors,
        "n_bins": n_bins,
        "n_occupied_bins": n_occupied,
        "occupancy": occupancy,
        "threshold": threshold,
        "discretisation": discretisation,
        "fig": fig,
        "n_occupied": n_occ,
        "memory_mode": "sparse_remapped",
    }

    if store_T_matrices:
        result["transition_matrices"] = T_matrices

    return result


# ---------------------------------------------------------------------------
# Plotting (unchanged)
# ---------------------------------------------------------------------------

def _plot_ck_results(
    tau_values: np.ndarray,
    ck_errors: np.ndarray,
    threshold: float,
    tau_star: float | None,
    min_error_tau: float,
    min_error: float,
    is_markovian: bool,
    dt: float,
    D: int,
    n_bins: int,
    occupancy: float,
    discretisation: str,
) -> plt.Figure:
    """Generate the CK error vs tau plot."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot CK error curve
    ax.semilogy(tau_values / dt, ck_errors, "ko-", markersize=8,
                markerfacecolor="steelblue", linewidth=1.5, label="CK error",
                zorder=3)

    # Threshold line
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=2,
               label=f"threshold = {threshold}", zorder=2)

    # tau* marker
    if is_markovian and tau_star is not None:
        tau_star_idx = int(np.argmin(np.abs(tau_values - tau_star)))
        ax.plot(tau_star / dt, ck_errors[tau_star_idx], "*", color="green",
                markersize=20, markeredgecolor="black", markeredgewidth=1.5,
                label=f"$\\tau^*$ = {tau_star:.2f} s", zorder=4)

    # Min error marker
    min_idx = int(np.nanargmin(ck_errors))
    ax.plot(tau_values[min_idx] / dt, min_error, "s", color="orange",
            markersize=10, markeredgecolor="black", markeredgewidth=1,
            label=f"min error = {min_error:.4f}", zorder=4)

    ax.set_xlabel(r"Lag $\tau$  [$\Delta t$ steps]", fontsize=12)
    ax.set_ylabel(
        r"CK error  $\|\sqrt{\pi}(T(\tau) - T(\tau/2)^2)\|_F \,/\, "
        r"\|\sqrt{\pi}\,T(\tau)\|_F$",
        fontsize=11,
    )

    verdict_str = "MARKOVIAN" if is_markovian else "NOT MARKOVIAN"
    verdict_color = "green" if is_markovian else "crimson"
    ax.set_title(
        f"Chapman-Kolmogorov Test  —  {D}D  —  "
        f"bins={n_bins}$^{D}$, {discretisation}, occ={occupancy:.0%}  —  "
        f"$\mathbf{{{verdict_str}}}$",
        fontsize=12, color=verdict_color, fontweight="bold",
    )

    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=max(ck_errors.min() * 0.3, 1e-4))

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _demo():
    """Run a quick demo with synthetic data."""
    np.random.seed(42)
    n = 150000
    dt = 1.0 / 250.0

    # 2D Ornstein-Uhlenbeck (known to be Markovian)
    tau_ou = 0.05
    sigma = 0.5
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] - x[t-1] * dt / tau_ou + sigma * np.sqrt(dt) * np.random.randn()
        y[t] = y[t-1] - y[t-1] * dt / tau_ou + sigma * np.sqrt(dt) * np.random.randn()
    data = np.column_stack([x, y])

    print("Demo: 2D Ornstein-Uhlenbeck (should be Markovian)\n")
    result = chapman_kolmogorov_test(data, dt=dt, n_bins=15,
                                     threshold=0.15, plot=False, verbose=True)
    print(f"\n  is_markovian : {result['is_markovian']}")
    print(f"  tau*         : {result['tau_star']:.4f} s")
    print(f"  min_error    : {result['min_error']:.4f}")


if __name__ == "__main__":
    import sys
    _demo()

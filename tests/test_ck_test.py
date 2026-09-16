"""
Unit tests for the Chapman-Kolmogorov Markovianity test module.
===============================================================

Generates a battery of synthetic time series with known Markovianity
properties and checks whether ``chapman_kolmogorov_test`` classifies
them correctly.

Usage::

    # Standalone (from repo root)
    python -m src.test.test_ck_test

    # With pytest
    pytest src/test/test_ck_test.py -v

Expected output: 4 test cases — Markovian processes should pass
(``is_markovian == True``) and non-Markovian processes should fail
(``is_markovian == False``).  A summary table is printed at the end.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.latent_space_extraction.ck_test import chapman_kolmogorov_test


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

DT: float = 1.0 / 250.0          # 250 Hz sampling
N_SAMPLES: int = 100_000         # ~400 s of data (enough for 2D @ 15 bins)
SEED: int = 42
CK_KWARGS = dict(
    dt=DT,
    n_bins=15,
    threshold=0.15,
    discretisation="uniform",
    plot=False,
    verbose=True,
)


# ===========================================================================
# Synthetic data generators
# ===========================================================================

def make_ou_1d(n: int, dt: float, *, tau: float = 0.05, sigma: float = 0.5,
               seed: int = SEED) -> np.ndarray:
    """
    1D Ornstein-Uhlenbeck process — Markovian by construction.
    dx = -x/tau dt + sigma dW
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] - x[t - 1] * dt / tau + sigma * np.sqrt(dt) * rng.standard_normal()
    return x.reshape(-1, 1)


def make_ou_2d(n: int, dt: float, *, tau: float = 0.05, sigma: float = 0.5,
               seed: int = SEED) -> np.ndarray:
    """
    2D independent Ornstein-Uhlenbeck — Markovian by construction.
    """
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] - x[t - 1] * dt / tau + sigma * np.sqrt(dt) * rng.standard_normal()
        y[t] = y[t - 1] - y[t - 1] * dt / tau + sigma * np.sqrt(dt) * rng.standard_normal()
    return np.column_stack([x, y])


def make_ou_2d_coupled(n: int, dt: float, *, tau: float = 0.05,
                       coupling: float = 0.3, sigma: float = 0.5,
                       seed: int = SEED) -> np.ndarray:
    """
    2D coupled Ornstein-Uhlenbeck — Markovian in 2D.
    dx1 = -(x1 + coupling*x2)/tau dt + sigma dW1
    dx2 = -(x2 + coupling*x1)/tau dt + sigma dW2
    """
    rng = np.random.default_rng(seed)
    x1, x2 = np.zeros(n), np.zeros(n)
    sqrt_dt = np.sqrt(dt)
    for t in range(1, n):
        x1[t] = x1[t-1] - (x1[t-1] + coupling * x2[t-1]) * dt / tau + sigma * sqrt_dt * rng.standard_normal()
        x2[t] = x2[t-1] - (x2[t-1] + coupling * x1[t-1]) * dt / tau + sigma * sqrt_dt * rng.standard_normal()
    return np.column_stack([x1, x2])


def make_ar2_2d(n: int, dt: float, *, phi1: float = 1.4, phi2: float = -0.6,
                sigma: float = 0.5, seed: int = SEED) -> np.ndarray:
    """
    2D AR(2) process — NOT Markovian in 2D (requires 4D for Markov representation).
    x_t = phi1 * x_{t-1} + phi2 * x_{t-2} + eps
    Each dimension is an independent AR(2).
    """
    rng = np.random.default_rng(seed)
    x, y = np.zeros(n), np.zeros(n)
    for t in range(2, n):
        x[t] = phi1 * x[t-1] + phi2 * x[t-2] + sigma * rng.standard_normal()
        y[t] = phi1 * y[t-1] + phi2 * y[t-2] + sigma * rng.standard_normal()
    return np.column_stack([x, y])


def make_delayed_feedback_2d(n: int, dt: float, *, delay_steps: int = 50,
                             tau: float = 0.05, sigma: float = 0.5,
                             seed: int = SEED) -> np.ndarray:
    """
    2D system with delayed feedback — NOT Markovian.
    x(t) follows OU but y(t) depends on x(t-delay), introducing
    explicit memory beyond the current state.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t-1] - x[t-1] * dt / tau + sigma * np.sqrt(dt) * rng.standard_normal()

    # y is a delayed, noisy version of x
    y = np.zeros(n)
    for t in range(delay_steps, n):
        y[t] = 0.8 * x[t - delay_steps] + 0.2 * sigma * rng.standard_normal()

    return np.column_stack([x, y])


# ===========================================================================
# Test harness
# ===========================================================================

@dataclass
class TestCase:
    name: str
    generator: callable
    expected_markovian: bool
    description: str


TEST_CASES: list[TestCase] = [
    TestCase(
        name="OU-1D",
        generator=lambda: make_ou_1d(N_SAMPLES, DT),
        expected_markovian=True,
        description="1D Ornstein-Uhlenbeck (Markovian by construction)",
    ),
    TestCase(
        name="OU-2D",
        generator=lambda: make_ou_2d(N_SAMPLES, DT),
        expected_markovian=True,
        description="2D independent OU (Markovian by construction)",
    ),
    TestCase(
        name="OU-2D-coupled",
        generator=lambda: make_ou_2d_coupled(N_SAMPLES, DT),
        expected_markovian=True,
        description="2D coupled OU (Markovian in 2D)",
    ),
    TestCase(
        name="AR2-2D",
        generator=lambda: make_ar2_2d(N_SAMPLES, DT),
        expected_markovian=False,
        description="2D AR(2) process (NOT Markovian in 2D)",
    ),
    TestCase(
        name="Delayed-feedback",
        generator=lambda: make_delayed_feedback_2d(N_SAMPLES, DT),
        expected_markovian=False,
        description="2D delayed feedback (NOT Markovian)",
    ),
]


def run_all_tests() -> list[dict]:
    """
    Run the full test battery and return detailed results.

    Returns
    -------
    results : list of dict
        One dict per test case with keys:
        name, expected, predicted, correct, tau_star, min_error,
        elapsed_time, data_shape.
    """
    results = []

    print("=" * 80)
    print("  CHAPMAN-KOLMOGOROV TEST — VALIDATION SUITE")
    print("=" * 80)
    print(f"  Samples : {N_SAMPLES:,}  (~{N_SAMPLES * DT:.0f} s @ {1/DT:.0f} Hz)")
    print(f"  CK params: n_bins={CK_KWARGS['n_bins']}, threshold={CK_KWARGS['threshold']}")
    print()

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {tc.name:20s} — {tc.description}")
        print(f"       Generating data ...", end=" ")
        t_gen = time.time()
        data = tc.generator()
        print(f"shape={data.shape}, {time.time()-t_gen:.2f}s")

        print(f"       Running CK test  ...", end=" ")
        t_ck = time.time()
        result = chapman_kolmogorov_test(data,out_dir = _SCRIPT_DIR / f"ck_test_results_{i}/", **CK_KWARGS)
        elapsed = time.time() - t_ck

        predicted = result["is_markovian"]
        correct = predicted == tc.expected_markovian

        status = "PASS" if correct else "FAIL"
        status_color = "\033[92m" if correct else "\033[91m"  # green / red
        reset = "\033[0m"

        print(f"{elapsed:.1f}s  →  "
              f"Markovian={predicted} (expected={tc.expected_markovian})  "
              f"{status_color}{status}{reset}")
        print(f"       tau*={result['tau_star']:.4f}s, "
              f"min_err={result['min_error']:.4f}, "
              f"occ={result['occupancy']:.1%}")
        print()

        results.append({
            "name": tc.name,
            "expected": tc.expected_markovian,
            "predicted": predicted,
            "correct": correct,
            "tau_star": result["tau_star"],
            "min_error": result["min_error"],
            "occupancy": result["occupancy"],
            "elapsed_time": elapsed,
            "data_shape": data.shape,
        })

    return results


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table of all test results."""
    n_total = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_tp = sum(1 for r in results if r["expected"] and r["predicted"])
    n_tn = sum(1 for r in results if not r["expected"] and not r["predicted"])
    n_fp = sum(1 for r in results if not r["expected"] and r["predicted"])
    n_fn = sum(1 for r in results if r["expected"] and not r["predicted"])

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print()
    print(f"  {'Test':<22} {'Expected':>10} {'Predicted':>10} {'Correct?':>10} "
          f"{'tau* (s)':>12} {'Min error':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for r in results:
        check = "YES" if r["correct"] else "NO"
        tau_str = f"{r['tau_star']:.4f}" if np.isfinite(r["tau_star"]) else "INF"
        print(f"  {r['name']:<22} {str(r['expected']):>10} {str(r['predicted']):>10} "
              f"{check:>10} {tau_str:>12} {r['min_error']:>10.4f}")

    print()
    print(f"  {'='*50}")
    print(f"  Total tests : {n_total}")
    print(f"  Correct     : {n_correct}/{n_total} ({n_correct/n_total:.0%})")
    print(f"  True  Pos   : {n_tp}  (Markovian detected as Markovian)")
    print(f"  True  Neg   : {n_tn}  (Non-Markovian detected as non-Markovian)")
    print(f"  False Pos   : {n_fp}  (Non-Markovian detected as Markovian)")
    print(f"  False Neg   : {n_fn}  (Markovian detected as non-Markovian)")

    if n_fp > 0:
        print(f"  [WARN] {n_fp} false positive(s) — threshold may be too lax")
    if n_fn > 0:
        print(f"  [WARN] {n_fn} false negative(s) — threshold may be too strict")
    if n_fp == 0 and n_fn == 0:
        print(f"  [OK]   All tests passed — perfect classification")

    print(f"  {'='*50}")
    print()

    # Exit code
    return n_correct == n_total


# ===========================================================================
# Entry points
# ===========================================================================

def main() -> int:
    """Run the full validation suite."""
    results = run_all_tests()
    all_passed = print_summary(results)
    return 0 if all_passed else 1


# Pytest compatibility
class TestChapmanKolmogorov:
    """Pytest-style test class."""

    @classmethod
    def setup_class(cls):
        cls.results = run_all_tests()

    def test_ou_1d_is_markovian(self):
        r = self.results[0]
        assert r["predicted"] == r["expected"], \
            f"OU-1D: expected {r['expected']}, got {r['predicted']}"

    def test_ou_2d_is_markovian(self):
        r = self.results[1]
        assert r["predicted"] == r["expected"], \
            f"OU-2D: expected {r['expected']}, got {r['predicted']}"

    def test_ou_2d_coupled_is_markovian(self):
        r = self.results[2]
        assert r["predicted"] == r["expected"], \
            f"OU-2D-coupled: expected {r['expected']}, got {r['predicted']}"

    def test_ar2_is_not_markovian(self):
        r = self.results[3]
        assert r["predicted"] == r["expected"], \
            f"AR2-2D: expected {r['expected']}, got {r['predicted']}"

    def test_delayed_feedback_is_not_markovian(self):
        r = self.results[4]
        assert r["predicted"] == r["expected"], \
            f"Delayed-feedback: expected {r['expected']}, got {r['predicted']}"


if __name__ == "__main__":
    sys.exit(main())

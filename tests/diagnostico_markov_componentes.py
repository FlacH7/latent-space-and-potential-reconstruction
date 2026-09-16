#!/usr/bin/env python3
"""
diagnostico_markov_componentes.py
=================================

Diagnóstico de selecciones Markov (fastest/slowest) para ejecuciones del
pipeline IgA con ``hankel + pca_ica`` (u otra combinación stage1+stage2).

Dado un sujeto/sesión/tarea, localiza los dos cachés del pipeline
(``...markov_fastest...npz`` y ``...markov_slowest...npz``) y ejecuta:

  A. Solape entre selecciones (componentes compartidas fast ∩ slow).
  B. Convergencia de FastICA (``n_iter_`` vs ``max_iter``).
  C. Espectro de valores singulares de la SVD previa al ICA (detección de
     direcciones casi nulas amplificadas por el blanqueo — p. ej. si los
     datos vienen en referencia al promedio y la matriz es rango-deficiente).
  D. Curtosis por componente (colas pesadas → discretización frágil).
  E. Paisaje de τ: τ(c, j) de cada componente compartida con todas las
     demás, y matriz τ completa de pares (verificación de que el
     mínimo/máximo reproducen las selecciones del meta).
  F. Estabilidad de las selecciones frente a ``n_bins`` (5/10/15).
  G. PSD de todas las componentes (multitaper, con las sospechosas
     resaltadas) y segmentos de serie temporal de las sospechosas.

Salidas (en ``--out-dir``; por defecto en el árbol de resultados del
pipeline: ``{BASE_RESULTS_PATH}/test_retest_gedai/{subject}/{session}/
diagnostico_markov_{task}_latent_dim_{D}_{stage1}+{stage2}``):
  - ``diagnostico_reporte.txt``
  - ``diagnostico_valores_singulares.png``
  - ``diagnostico_tau_matriz.png``
  - ``diagnostico_tau_paisaje_compartidas.png``
  - ``diagnostico_psd_componentes.png``
  - ``diagnostico_series_temporales.png``

Uso (desde la raíz del repositorio)::

    python3 diagnostico_markov_componentes.py \
        --subject sub-01 --session session1 --task eyesclosed \
        --t-start 0 --t-end 300

Las rutas por defecto se toman de ``src.utils.config`` (BASE_CACHE_PATH
para los cachés de entrada y BASE_RESULTS_PATH para la salida), igual que
hacen el batch y el pipeline; se pueden sobreescribir con ``--cache-root``,
``--results-root`` y ``--out-dir``. NO modifica ningún fichero del
pipeline: solo lee los cachés y escribe en el directorio de diagnóstico.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # backend no interactivo siempre
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Bootstrap de imports del proyecto
# ---------------------------------------------------------------------------

def _bootstrap_project_paths() -> None:
    """Añade la raíz del repo a sys.path (igual que los scripts batch)."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for base in list(candidates):
        if (base / "src" / "utils" / "config.py").exists():
            for p in (base, base / "src"):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            return
    # Último recurso: cwd tal cual (por si el repo ya está en PYTHONPATH)
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))


def _load_config_paths(
    cli_cache_root: str | None,
    cli_results_root: str | None,
) -> tuple[Path | None, Path | None]:
    """Raíces de caché y de resultados.

    Por defecto se importan ``BASE_CACHE_PATH`` y ``BASE_RESULTS_PATH`` de
    ``src.utils.config`` (las mismas variables que usan el batch y el
    pipeline); ``--cache-root`` / ``--results-root`` las sobreescriben.
    Devuelve ``(cache_root, results_root)``, ``None`` si no se pudo resolver.
    """
    base_cache = base_results = None
    try:
        from src.utils.config import BASE_CACHE_PATH, BASE_RESULTS_PATH  # type: ignore
        base_cache, base_results = Path(BASE_CACHE_PATH), Path(BASE_RESULTS_PATH)
    except Exception:
        pass
    cache_root = Path(cli_cache_root) if cli_cache_root else base_cache
    results_root = Path(cli_results_root) if cli_results_root else base_results
    return cache_root, results_root


def _import_markov_helpers():
    try:
        from src.latent_space_extraction.markov_subspace import (
            discretize_series,
            evaluate_markov_combination,
            _evaluate_markov_worker,
        )
        return discretize_series, evaluate_markov_combination, _evaluate_markov_worker
    except Exception as exc:
        sys.exit(
            "[ERROR] No se pudo importar src.latent_space_extraction.markov_subspace "
            f"({exc}). Ejecuta desde la raíz del repo."
        )


# ---------------------------------------------------------------------------
# Localización y carga de cachés
# ---------------------------------------------------------------------------

def _find_cache_npz(
    cache_root: Path,
    subject: str,
    session: str,
    task: str,
    latent_dim: int,
    chain_label: str,
    t_start: float | None,
    t_end: float | None,
) -> Path:
    """Glob del caché de un método (la etiqueta incluye el hash de params)."""
    base = cache_root / "cache_eeg_test_retest_gedai" / subject / session
    pattern = (
        f"task_{task}_latent_dim_{latent_dim}_{chain_label}_*/from*s_to_*s.npz"
    )
    matches = sorted(base.glob(pattern))
    if not matches:
        avail = sorted(p.name for p in base.glob("task_*")) if base.exists() else []
        sys.exit(
            f"[ERROR] No se encontró caché para {chain_label} en:\n  {base}\n"
            f"  patrón: {pattern}\n"
            f"  directorios disponibles: {avail or 'ninguno'}"
        )
    if t_start is not None and t_end is not None:
        def _ok(p: Path) -> bool:
            try:
                stem = p.stem  # from{X}s_to_{Y}s
                x = float(stem.split("from")[1].split("s_to_")[0])
                y = float(stem.split("s_to_")[1].rstrip("s"))
                return abs(x - t_start) < 1e-6 and abs(y - t_end) < 1e-6
            except Exception:
                return False
        matches = [p for p in matches if _ok(p)]
        if not matches:
            sys.exit(
                f"[ERROR] Hay cachés de {chain_label} pero ninguno con la "
                f"ventana t=[{t_start}, {t_end}]."
            )
    if len(matches) > 1:
        print(f"  [AVISO] {len(matches)} cachés candidatos de {chain_label}; "
              f"uso el más reciente:\n    " + "\n    ".join(str(m) for m in matches))
    return matches[-1]


def _load_cache(npz_path: Path) -> tuple[np.ndarray, dict]:
    loaded = np.load(npz_path, allow_pickle=True)
    latent = np.asarray(loaded["latent"], dtype=float)
    meta = loaded["meta"].item()
    return latent, meta


def _get_Y(meta: dict, latent: np.ndarray) -> np.ndarray:
    """Matriz Y (D × T') de stage 2.

    Preferido: ``meta['Y']`` (guardado por el orquestador). Fallback:
    recomputar stage1+stage2 desde ``meta['preprocessing']['X_filtered']``
    con los parámetros registrados (determinista vía ica_random_state).
    """
    Y = meta.get("Y")
    if Y is not None:
        Y = np.asarray(Y, dtype=float)
        # Verificación contra el latente cacheado
        sel = meta.get("selected_indices") or meta.get("stage3", {}).get("selected_indices")
        if sel is not None and Y.shape[0] > max(sel):
            diff = float(np.max(np.abs(Y[list(sel), :].T - latent))) \
                if latent.shape == Y[list(sel), :].T.shape else np.nan
            print(f"  Y cargada de meta: shape={Y.shape} | "
                  f"max|Y[sel].T - latent| = {diff:.3e}")
        return Y

    print("  [AVISO] meta['Y'] no existe (caché antiguo). Recomputando stage1+2 "
          "desde X_filtered...")
    pre = meta.get("preprocessing", {})
    X = pre.get("X_filtered")
    if X is None:
        sys.exit("[ERROR] Ni meta['Y'] ni meta['preprocessing']['X_filtered'] "
                 "están disponibles; reejecuta el pipeline con el código actual.")
    params = meta.get("pipeline", {}).get("params", {})
    s1p = dict(params.get("stage1_params") or {})
    s2p = dict(params.get("stage2_params") or {})
    from src.latent_space_extraction.pipeline.stage1_embedding import (
        HankelEmbedding, PipelineContext)
    from src.latent_space_extraction.pipeline.stage2_dynamics import PCAICADynamics
    sfreq = float(pre.get("sfreq"))
    ctx = PipelineContext(raw=None, sfreq=sfreq, dt=1.0 / sfreq, n_workers=1,
                          stage1_name="hankel")
    H, meta1 = HankelEmbedding(**s1p).fit_transform(np.asarray(X, float), ctx=ctx)
    ctx.stage1_meta = meta1
    n_dim = int(meta.get("latent_scores", {}).get("n_dim", 2) or 2)
    Y2, _ = PCAICADynamics(**s2p).fit_transform(H, ctx=ctx, n_dim=n_dim)
    return np.asarray(Y2, dtype=float)


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def _scree_report(sv: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Métricas del espectro de valores singulares (detección de cliff)."""
    lines: list[str] = []
    ratios = sv[1:] / np.maximum(sv[:-1], 1e-300)
    lines.append(f"  σ_1 = {sv[0]:.6g} | σ_D = {sv[-1]:.6g} | σ_D/σ_1 = {sv[-1]/sv[0]:.3e}")
    lines.append(f"  Ratios σ_{{i+1}}/σ_i (últimos 5): "
                 + ", ".join(f"{r:.3f}" for r in ratios[-5:]))
    i_min = int(np.argmin(ratios))
    lines.append(f"  Mayor caída relativa: σ_{i_min+2}/σ_{i_min+1} = {ratios[i_min]:.3f}")
    if sv[-1] < 1e-3 * sv[0]:
        lines.append("  ⚠ σ_D < 1e-3·σ_1: la última dirección SVD es casi nula. "
                     "Tras el blanqueo se amplifica a varianza 1 → la última "
                     "componente ICA puede ser ruido numérico (típico si los "
                     "datos están en referencia al promedio = rango deficiente).")
    if ratios[-1] < 0.7:
        lines.append(f"  ⚠ σ_D/σ_{{D-1}} = {ratios[-1]:.3f} < 0.7: salto brusco al "
                     "final del espectro; considera reducir n_components.")
    if not any("⚠" in l for l in lines):
        lines.append("  ✓ Sin cliff evidente al final del espectro.")
    return lines, ratios


def _kurtosis_table(Y: np.ndarray, flagged: float = 10.0) -> tuple[list[str], np.ndarray]:
    from scipy.stats import kurtosis
    k = np.array([kurtosis(Y[i]) for i in range(Y.shape[0])])
    lines = ["  Curtosis (Fisher) por componente:"]
    for i, ki in enumerate(k):
        mark = "  ⚠ colas pesadas" if abs(ki) > flagged else ""
        lines.append(f"    comp {i:2d}: {ki:9.2f}{mark}")
    return lines, k


def _pair_tau_matrix(
    Y: np.ndarray,
    n_bins: int,
    n_workers: int | None,
    discretize_series,
    evaluate_markov_combination,
    _evaluate_markov_worker,
) -> np.ndarray:
    """Matriz D×D de τ para todos los pares (simétrica; diag = NaN)."""
    D = Y.shape[0]
    bins_idx = discretize_series(Y, n_bins)
    combs = list(combinations(range(D), 2))
    tau = np.full((D, D), np.nan)
    args = [(c, bins_idx, n_bins) for c in combs]
    results: list[tuple[tuple[int, ...], float]] = []
    if n_workers != 1:
        try:
            import multiprocessing as mp
            with mp.Pool(processes=n_workers) as pool:
                results = pool.map(_evaluate_markov_worker, args, chunksize=4)
        except Exception as exc:
            print(f"  [AVISO] multiprocessing falló ({exc}); sigo en serie.")
            results = []
    if not results:
        results = [evaluate_markov_combination(*a) for a in args]
    for comb, tval in results:
        i, j = comb
        tau[i, j] = tau[j, i] = tval
    return tau


def _best_pairs(tau: np.ndarray) -> tuple[tuple[tuple[int, int], float], tuple[tuple[int, int], float]]:
    """(par mínimo, par máximo) replicando la lógica de find_best_subspace_markov."""
    D = tau.shape[0]
    best_min, best_max = None, None
    tau_min, tau_max = np.inf, -np.inf
    for i in range(D):
        for j in range(i + 1, D):
            t = tau[i, j]
            if np.isnan(t):
                continue
            if t < tau_min:
                tau_min, best_min = t, (i, j)
            if np.isfinite(t) and t > tau_max:
                tau_max, best_max = t, (i, j)
    return (best_min, float(tau_min)), (best_max, float(tau_max))


def _landscape_lines(comp: int, tau: np.ndarray) -> list[str]:
    row = tau[comp]
    order = np.argsort(np.where(np.isnan(row), np.inf, row))
    finite = np.isfinite(row) & ~np.isnan(row)
    lines = [f"  Paisaje τ(comp {comp}, j):"]
    lines.append("    3 parejas más rápidas: "
                 + ", ".join(f"j={j} (τ={row[j]:.2f})" for j in order[:3]))
    high = [j for j in np.argsort(np.where(finite, row, -np.inf))[::-1] if j != comp][:3]
    lines.append("    3 parejas más lentas : "
                 + ", ".join(f"j={j} (τ={row[j]:.2f})" for j in high))
    r_fast = int(np.sum(np.nanmin(np.where(np.eye(row.size, dtype=bool), np.inf, tau), axis=1)
                    < row[order[0]]))
    lines.append(f"    τ min de {comp} = {row[order[0]]:.2f} | "
                 f"nº de componentes cuyo mejor par es más rápido: ~{r_fast}")
    return lines


def _psd_components(Y: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD multitaper de todas las componentes (usa el módulo spectral si está)."""
    try:
        from src.spectral_analysis.psd_analysis import _multitaper_auto
        psds, freqs, _ = _multitaper_auto(Y, sfreq, 1.0, min(100.0, sfreq / 2), 2.5)
        return psds, freqs
    except Exception:
        from scipy.signal import welch
        nperseg = int(min(4.0 * sfreq, Y.shape[1]))
        freqs, psds = welch(Y, fs=sfreq, nperseg=nperseg, axis=1)
        m = (freqs >= 1.0) & (freqs <= min(100.0, sfreq / 2))
        return psds[:, m], freqs[m]


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _fig_scree(sv: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    idx = np.arange(1, len(sv) + 1)
    ax.semilogy(idx, sv, "o-", color="C0", lw=1.4)
    ax.axhline(1e-3 * sv[0], color="C3", ls="--", lw=1.0,
               label="1e-3 · σ₁ (sospecha de dirección nula)")
    ax.set_xlabel("Índice de componente SVD")
    ax.set_ylabel("σ (escala log)")
    ax.set_title("Valores singulares de la SVD previa al ICA")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_tau_matrix(tau: np.ndarray, sel_fast: tuple, sel_slow: tuple, out: Path) -> None:
    finite = tau[np.isfinite(tau)]
    vmax = float(np.percentile(finite, 95)) if finite.size else 1.0
    show = np.where(np.isfinite(tau), np.minimum(tau, vmax), np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(show, cmap="viridis", origin="lower")
    fig.colorbar(im, ax=ax, label=f"τ (recortado al p95 = {vmax:.1f})")
    for comb, color, tag in ((sel_fast, "red", "FAST"), (sel_slow, "white", "SLOW")):
        if comb is None:
            continue
        i, j = comb
        ax.scatter([j], [i], facecolors="none", edgecolors=color, s=180, lw=2.0,
                   label=f"{tag} {comb}")
    ax.set_xlabel("componente j")
    ax.set_ylabel("componente i")
    ax.set_title("Matriz τ de pares (NaN = no ergódico)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_landscape(shared: list[int], tau: np.ndarray, out: Path) -> None:
    n = max(len(shared), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.0), squeeze=False)
    for ax, c in zip(axes[0], shared):
        row = tau[c]
        colors = ["C0"] * row.size
        order = np.argsort(np.where(np.isnan(row), np.inf, row))
        colors[order[0]] = "C2"   # pareja más rápida
        finite = np.isfinite(row) & ~np.isnan(row)
        j_slow = int(np.argmax(np.where(finite, row, -np.inf)))
        colors[j_slow] = "C3"      # pareja más lenta
        ax.bar(range(row.size), np.where(np.isnan(row), 0, row), color=colors)
        ax.axhline(np.nanmedian(row[finite]) if finite.any() else 0.0,
                   color="k", ls="--", lw=0.8, label="mediana τ finita")
        ax.set_xlabel("pareja j")
        ax.set_ylabel("τ")
        ax.set_title(f"τ(comp {c}, j) — verde: más rápida, rojo: más lenta")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_psds(psds: np.ndarray, freqs: np.ndarray, suspicious: set[int],
              out: Path, fmax_plot: float = 45.0) -> None:
    D = psds.shape[0]
    n_cols = 5
    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.0 * n_rows),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    m = freqs <= fmax_plot
    psd_db = 10.0 * np.log10(np.maximum(psds, 1e-300))
    for d in range(D):
        ax = axes[d]
        color = "C3" if d in suspicious else "C0"
        ax.plot(freqs[m], psd_db[d, m], color=color, lw=1.0)
        ax.set_title(f"comp {d}" + ("  ⚠" if d in suspicious else ""), fontsize=8,
                     color=color)
        ax.grid(True, alpha=0.3, lw=0.5)
    for k in range(D, axes.size):
        axes[k].axis("off")
    fig.supxlabel("Frecuencia (Hz)")
    fig.supylabel("PSD (dB, a.u.)")
    fig.suptitle("PSD de las componentes de stage 2 (rojo = sospechosas)", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _fig_timeseries(Y: np.ndarray, sfreq: float, suspicious: list[int],
                    kurt: np.ndarray, out: Path, dur_s: float = 10.0) -> None:
    n = len(suspicious)
    if n == 0:
        return
    T_show = min(int(dur_s * sfreq), Y.shape[1])
    t = np.arange(T_show) / sfreq
    fig, axes = plt.subplots(n, 1, figsize=(10.0, 2.2 * n), squeeze=False)
    for ax, c in zip(axes[:, 0], suspicious):
        ax.plot(t, Y[c, :T_show], lw=0.6, color="C0")
        ax.set_ylabel(f"comp {c}")
        ax.set_title(f"comp {c} — primeros {T_show / sfreq:.1f} s | "
                     f"curtosis={kurt[c]:.1f}", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("tiempo (s)")
    fig.suptitle("Series temporales de componentes sospechosas", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnóstico de selecciones Markov fast/slow (hankel+pca_ica).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--subject", required=True, help="p. ej. sub-01")
    p.add_argument("--session", required=True, help="p. ej. session1")
    p.add_argument("--task", required=True, help="p. ej. eyesclosed")
    p.add_argument("--t-start", type=float, default=None,
                   help="inicio de la ventana (s); si se omite, se infiere del caché")
    p.add_argument("--t-end", type=float, default=None,
                   help="fin de la ventana (s); si se omite, se infiere del caché")
    p.add_argument("--latent-dim", type=int, default=2)
    p.add_argument("--stage1", default="hankel")
    p.add_argument("--stage2", default="pca_ica")
    p.add_argument("--cache-root", default=None,
                   help="raíz de caché (por defecto BASE_CACHE_PATH de src.utils.config)")
    p.add_argument("--results-root", default=None,
                   help="raíz de resultados (por defecto BASE_RESULTS_PATH de "
                        "src.utils.config); base del directorio de salida")
    p.add_argument("--out-dir", default=None,
                   help="directorio de salida exacto (sobreescribe --results-root)")
    p.add_argument("--n-workers", type=int, default=None,
                   help="procesos para la matriz τ (None = todos los cores, 1 = serie)")
    p.add_argument("--n-bins-stability", default="5,10,15",
                   help="valores de n_bins para el análisis de estabilidad")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _bootstrap_project_paths()
    cache_root, results_root = _load_config_paths(args.cache_root, args.results_root)
    if cache_root is None:
        sys.exit(
            "[ERROR] No se pudo importar BASE_CACHE_PATH desde src.utils.config. "
            "Ejecuta desde la raíz del repo o pasa --cache-root."
        )
    discretize_series, evaluate_markov_combination, _evaluate_markov_worker = \
        _import_markov_helpers()

    chain_fast = f"{args.stage1}+{args.stage2}+markov_fastest"
    chain_slow = f"{args.stage1}+{args.stage2}+markov_slowest"

    print("=" * 72)
    print("  DIAGNÓSTICO Markov fast/slow")
    print("=" * 72)
    print(f"  Sujeto/sesión/tarea: {args.subject} / {args.session} / {args.task}")
    print(f"  Raíz de caché      : {cache_root}")
    print(f"  Raíz de resultados : {results_root}")

    npz_fast = _find_cache_npz(cache_root, args.subject, args.session, args.task,
                               args.latent_dim, chain_fast, args.t_start, args.t_end)
    npz_slow = _find_cache_npz(cache_root, args.subject, args.session, args.task,
                               args.latent_dim, chain_slow, args.t_start, args.t_end)
    print(f"  Caché fast : {npz_fast}")
    print(f"  Caché slow : {npz_slow}")

    latent_f, meta_f = _load_cache(npz_fast)
    latent_s, meta_s = _load_cache(npz_slow)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if results_root is None:
            sys.exit(
                "[ERROR] No se pudo importar BASE_RESULTS_PATH desde "
                "src.utils.config. Ejecuta desde la raíz del repo, o pasa "
                "--results-root / --out-dir."
            )
        out_dir = (
            results_root
            / "test_retest_gedai" / args.subject / args.session
            / f"diagnostico_markov_{args.task}_latent_dim_{args.latent_dim}"
              f"_{args.stage1}+{args.stage2}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Salida     : {out_dir}")

    report: list[str] = []
    rep = report.append
    rep("DIAGNÓSTICO Markov fastest/slowest")
    rep("=" * 72)
    rep(f"Sujeto/sesión/tarea: {args.subject} / {args.session} / {args.task}")
    rep(f"Caché fast: {npz_fast}")
    rep(f"Caché slow: {npz_slow}")
    rep("")

    # ------------------------------------------------------------------
    # A. Solape de selecciones
    # ------------------------------------------------------------------
    sel_f = list(meta_f.get("stage3", {}).get("selected_indices")
                 or meta_f.get("selected_indices"))
    sel_s = list(meta_s.get("stage3", {}).get("selected_indices")
                 or meta_s.get("selected_indices"))
    tau_f = meta_f.get("stage3", {}).get("scores", {}).get("tau")
    tau_s = meta_s.get("stage3", {}).get("scores", {}).get("tau")
    shared = sorted(set(sel_f) & set(sel_s))
    rep("[A] Selecciones registradas en los cachés")
    rep(f"    FAST : {sel_f} (τ={tau_f})")
    rep(f"    SLOW : {sel_s} (τ={tau_s})")
    rep(f"    Compartidas: {shared if shared else 'ninguna'}")
    if shared:
        rep("    Nota: τ es propiedad del PAR, no de la componente individual;")
        rep("    que los extremos compartan miembro no es contradictorio per se.")
    rep("")

    # ------------------------------------------------------------------
    # B. Convergencia de FastICA
    # ------------------------------------------------------------------
    rep("[B] Convergencia de FastICA (stage 2)")
    suspicious: set[int] = set(shared)
    for tag, meta in (("fast", meta_f), ("slow", meta_s)):
        n_iter = meta.get("stage2", {}).get("n_iter_", None)
        rs = meta.get("stage2", {}).get("ica_random_state", None)
        verdict = ""
        if n_iter is not None and int(n_iter) >= 1000:
            verdict = ("  ⚠ alcanzó max_iter=1000 → NO convergió; las "
                       "componentes pueden ser inestables (repetir con otro "
                       "ica_random_state debería dar selecciones distintas)")
        elif n_iter is not None and int(n_iter) > 0:
            verdict = "  ✓ convergió"
        rep(f"    [{tag}] n_iter_={n_iter} | ica_random_state={rs} {verdict}")
    rep("")

    # ------------------------------------------------------------------
    # C. Valores singulares
    # ------------------------------------------------------------------
    rep("[C] Espectro de valores singulares (SVD previa al ICA)")
    sv = np.asarray(meta_f.get("stage2", {}).get("singular_values")
                    or meta_f.get("preprocessing", {}).get("singular_values"),
                    dtype=float)
    if sv.size and np.all(sv > 0):
        lines, _ = _scree_report(sv)
        report.extend(lines)
        _fig_scree(sv, out_dir / "diagnostico_valores_singulares.png")
    else:
        rep("    (no hay singular_values en el meta)")
    rep("")

    # ------------------------------------------------------------------
    # D. Curtosis
    # ------------------------------------------------------------------
    rep("[D] Curtosis por componente (Y de stage 2)")
    print("\n  Cargando Y de stage 2...")
    Y = _get_Y(meta_f, latent_f)
    sfreq = float(meta_f.get("preprocessing", {}).get("sfreq"))
    lines, kurt = _kurtosis_table(Y)
    report.extend(lines)
    heavy = set(int(i) for i in np.where(np.abs(kurt) > 10.0)[0])
    rep(f"    Componentes con |curtosis| > 10: {sorted(heavy) if heavy else 'ninguna'}")
    suspicious |= heavy
    rep("")

    # Última componente como sospechosa estructural si hay cliff
    if sv.size and sv[-1] < 1e-3 * sv[0]:
        suspicious.add(Y.shape[0] - 1)

    # ------------------------------------------------------------------
    # E. Matriz τ + verificación + paisaje
    # ------------------------------------------------------------------
    rep("[E] Matriz τ de pares y paisaje de las compartidas")
    n_bins_meta = int(meta_f.get("pipeline", {}).get("params", {})
                      .get("stage3_params", {}).get("n_bins", 10))
    print(f"  Calculando matriz τ (n_bins={n_bins_meta}, "
          f"{Y.shape[0] * (Y.shape[0] - 1) // 2} pares)...")
    tau_mat = _pair_tau_matrix(Y, n_bins_meta, args.n_workers,
                               discretize_series, evaluate_markov_combination,
                               _evaluate_markov_worker)
    (best_min, tmin), (best_max, tmax) = _best_pairs(tau_mat)
    rep(f"    Mínimo de la matriz: {best_min} (τ={tmin:.3f}) | meta dice {sel_f} (τ={tau_f})"
        + ("  ✓ coincide" if best_min is not None and list(best_min) == sel_f
           else "  ⚠ NO coincide"))
    rep(f"    Máximo de la matriz: {best_max} (τ={tmax:.3f}) | meta dice {sel_s} (τ={tau_s})"
        + ("  ✓ coincide" if best_max is not None and list(best_max) == sel_s
           else "  ⚠ NO coincide"))
    _fig_tau_matrix(tau_mat, best_min, best_max, out_dir / "diagnostico_tau_matriz.png")
    if shared:
        _fig_landscape(shared, tau_mat, out_dir / "diagnostico_tau_paisaje_compartidas.png")
        for c in shared:
            report.extend(_landscape_lines(c, tau_mat))
            row = tau_mat[c]
            finite = row[np.isfinite(row)]
            if finite.size and (finite.max() - finite.min()) < 0.5 * np.median(finite):
                rep(f"    → comp {c}: τ con todas las parejas es similar "
                    f"({finite.min():.2f}–{finite.max():.2f}); su presencia en ambos "
                    "extremos es compatible con azar de ordenación.")
            else:
                rep(f"    → comp {c}: τ muy variable según la pareja "
                    f"({finite.min():.2f}–{finite.max():.2f}); no es 'rápida' ni "
                    "'lenta' por sí misma: depende del acompañante.")
    rep("")

    # ------------------------------------------------------------------
    # F. Estabilidad frente a n_bins
    # ------------------------------------------------------------------
    rep("[F] Estabilidad de las selecciones frente a n_bins")
    bins_list = sorted({int(b) for b in str(args.n_bins_stability).split(",")})
    rep(f"    {'n_bins':>7} | {'fast':>12} | {'slow':>12} | ¿solape?")
    for nb in bins_list:
        print(f"  Estabilidad n_bins={nb}...")
        tm = tau_mat if nb == n_bins_meta else _pair_tau_matrix(
            Y, nb, args.n_workers, discretize_series,
            evaluate_markov_combination, _evaluate_markov_worker)
        (bf, tf_), (bs, ts_) = _best_pairs(tm)
        rep(f"    {nb:>7} | {str(bf):>12} | {str(bs):>12} | "
            f"{sorted(set(bf) & set(bs)) if set(bf) & set(bs) else '—'}")
    rep("    Si la pareja compartida cambia con n_bins → artefacto de la")
    rep("    discretización, no propiedad de los datos.")
    rep("")

    # ------------------------------------------------------------------
    # G. PSD y series temporales
    # ------------------------------------------------------------------
    rep("[G] PSD de componentes y series temporales de sospechosas")
    print("  Calculando PSD de las componentes...")
    psds, freqs = _psd_components(Y, sfreq)
    _fig_psds(psds, freqs, suspicious, out_dir / "diagnostico_psd_componentes.png",
              fmax_plot=min(45.0, sfreq / 2))
    for c in sorted(suspicious):
        i_peak = int(np.argmax(psds[c])) if psds[c].size else 0
        rep(f"    comp {c}: pico espectral en {freqs[i_peak]:.2f} Hz "
            f"(curtosis={kurt[c]:.1f})")
    susp_list = sorted(suspicious)
    _fig_timeseries(Y, sfreq, susp_list[:6], kurt,
                    out_dir / "diagnostico_series_temporales.png")
    rep("")

    # ------------------------------------------------------------------
    # Lectura recomendada
    # ------------------------------------------------------------------
    rep("LECTURA RECOMENDADA")
    rep("-" * 72)
    rep("1. Si [B] marca no-convergencia: repite con otro ica_random_state antes")
    rep("   de interpretar nada.")
    rep("2. Si [C] marca cliff: la última componente ICA es probablemente una")
    rep("   dirección casi nula amplificada por el blanqueo (rango deficiente,")
    rep("   p. ej. por referencia al promedio heredada). Considera reducir")
    rep("   n_components por debajo del rango efectivo.")
    rep("3. Si la compartida tiene curtosis alta o PSD plana/espuria → artefacto;")
    rep("   exclúyela y reevalúa las selecciones.")
    rep("4. Solo si es estable en n_bins Y entre sujetos/sesiones tendría sentido")
    rep("   una interpretación neurocientífica (hub de acoplamiento no lineal).")

    txt = "\n".join(report)
    (out_dir / "diagnostico_reporte.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print(f"\n  ✔ Diagnóstico guardado en: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

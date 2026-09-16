"""Smoke test del módulo src.spectral_analysis con EEG sintético.

Cubre los criterios de aceptación del prompt:
- compute_and_plot_raw_psd genera la matriz de canales y la media.
- Los .npz se guardan y recargan (load_psd_data).
- compute_and_plot_latent_psd funciona para n_dim de 2 a 5.
- compute_channel_influence_on_latent produce topoplots (con montaje) y
  barras (sin montaje), con pesos correctos para PCA, DMD y PCA+ICA, y
  fallback graceful para Diffusion Maps.
- Caché: no recalcula si ya existe el .npz (salvo force_recompute).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne

from src.spectral_analysis.psd_analysis import (
    compute_and_plot_latent_psd,
    compute_and_plot_raw_psd,
    compute_channel_influence_on_latent,
    load_psd_data,
)

SFREQ = 250.0
DURATION = 60.0  # s
CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
N_CH = len(CH_NAMES)


def make_raw(with_montage: bool = True, seed: int = 0) -> tuple[mne.io.Raw, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_times = int(SFREQ * DURATION)
    t = np.arange(n_times) / SFREQ
    alpha = np.sin(2 * np.pi * 10 * t)
    beta = np.sin(2 * np.pi * 20 * t)
    slow = np.sin(2 * np.pi * 3 * t)
    w_alpha = rng.uniform(0.1, 1.0, N_CH)
    w_beta = rng.uniform(0.1, 1.0, N_CH)
    w_slow = rng.uniform(0.1, 1.0, N_CH)
    data = (
        np.outer(w_alpha, alpha)
        + np.outer(w_beta, beta)
        + np.outer(w_slow, slow)
        + 0.3 * rng.standard_normal((N_CH, n_times))
    ) * 1e-6
    info = mne.create_info(CH_NAMES, SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    if with_montage:
        raw.set_montage("standard_1020", on_missing="warn", verbose=False)
    W = np.stack([w_alpha, w_beta, w_slow], axis=1)  # (n_ch, 3)
    return raw, W


def make_latent(raw: np.ndarray, W: np.ndarray, hankel_depth: int = 1) -> np.ndarray:
    latent = raw.T @ W
    if hankel_depth > 1:  # simula n_samples = n_times - depth + 1
        latent = latent[: latent.shape[0] - hankel_depth + 1]
    return latent


def make_meta(stage2: str, W: np.ndarray, ch_names=CH_NAMES) -> dict:
    meta = {
        "pipeline": {"stage1": "hankel", "stage2": stage2, "stage3": "variance"},
        "stage2": {},
        "selected_indices": [0, 1, 2],
        "preprocessing": {"sfreq": SFREQ, "ch_names": list(ch_names), "n_channels": len(ch_names)},
    }
    if stage2 in ("pca", "pca_ica"):
        meta["stage2"]["components_"] = W.T  # sklearn: (n_components, n_features)
    elif stage2 == "dmd":
        phases = np.exp(1j * np.linspace(0, np.pi, W.shape[1]))
        meta["stage2"]["modes"] = W * phases  # complejos: (n_ch, n_modes)
    return meta


def check(cond: bool, msg: str) -> None:
    status = "OK " if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        raise AssertionError(msg)


def spearman_abs(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(abs(spearmanr(a, b).statistic))


def main() -> None:
    out_root = Path(tempfile.mkdtemp(prefix="psd_test_"))
    print(f"out_root: {out_root}")

    # ------------------------------------------------------------------
    # 1. Pipeline con montaje (caso test-retest Gedai) + PCA
    # ------------------------------------------------------------------
    raw, W = make_raw(with_montage=True)
    latent = make_latent(raw.get_data(), W, hankel_depth=50)  # latente más corto
    meta = make_meta("pca", W)
    out_dir = out_root / "pca_montage"

    psds, freqs, ch_names, mean_psd, std_psd, raw_psd_path = compute_and_plot_raw_psd(
        raw, out_dir=out_dir, fmin=1.0, fmax=100.0, bandwidth=2.5
    )
    check(psds.shape == (N_CH, len(freqs)), "raw psds shape (n_ch, n_freqs)")
    check((out_dir / "raw_psd_all_channels.png").exists(), "raw_psd_all_channels.png existe")
    check((out_dir / "raw_psd_mean.png").exists(), "raw_psd_mean.png existe")
    check(raw_psd_path.exists(), "raw_psd_data.npz existe")

    loaded = load_psd_data(raw_psd_path)
    check(loaded["psds"].shape == psds.shape, "npz recargable: psds")
    check(isinstance(loaded["metadata"], dict) and loaded["metadata"]["method"] == "multitaper",
          "npz recargable: metadata dict")
    check(list(loaded["ch_names"]) == CH_NAMES, "npz recargable: ch_names")

    # El pico alfa debe estar ~10 Hz en el PSD medio
    peak = freqs[np.argmax(mean_psd[(freqs >= 8) & (freqs <= 13)]) + np.argmax(freqs >= 1)]
    mask = (freqs >= 8) & (freqs <= 13)
    peak = freqs[mask][np.argmax(mean_psd[mask])]
    check(abs(peak - 10.0) < 1.0, f"pico alfa en ~10 Hz (obtenido {peak:.2f} Hz)")

    # Caché: segunda llamada sin force_recompute devuelve los mismos datos
    cached = compute_and_plot_raw_psd(raw, out_dir=out_dir)
    check(np.allclose(cached[0], psds), "caché: no recalcula PSDs existentes")

    # PSD latente
    psds_l, freqs_l, mean_l, std_l, latent_psd_path = compute_and_plot_latent_psd(
        latent, sfreq=SFREQ, out_dir=out_dir, fmin=1.0, fmax=100.0
    )
    check(psds_l.shape == (3, len(freqs_l)), "latent psds shape (n_dim, n_freqs)")
    check((out_dir / "latent_psd_all_dims.png").exists(), "latent_psd_all_dims.png existe")
    check((out_dir / "latent_psd_mean.png").exists(), "latent_psd_mean.png existe")
    check(load_psd_data(latent_psd_path)["metadata"]["latent_dim"] == 3, "metadata latent_dim")

    # Influencia de canales (PCA + montaje → topoplots)
    weights, ch_out, fig_path, data_path = compute_channel_influence_on_latent(
        raw, latent, meta, out_dir=out_dir, raw_psd_path=raw_psd_path
    )
    check(weights.shape == (N_CH, 3), "influence_weights shape (n_ch, n_dim)")
    check(0 <= weights.min() and weights.max() <= 1.0, "pesos normalizados en [0, 1]")
    check(fig_path.exists() and data_path.exists(), "channel_influence.png y .npz existen")
    sims = [spearman_abs(weights[:, d], np.abs(W[:, d])) for d in range(3)]
    check(min(sims) > 0.95, f"pesos PCA recuperan W (spearman mínimo {min(sims):.3f})")
    inv = load_psd_data(data_path)
    check("spectral_contribution" in inv and inv["spectral_contribution"].shape == (N_CH, 3, 5),
          "métrica espectral guardada (n_ch, n_dim, n_bands)")
    check(inv["spectral_correlation"].shape == (N_CH, 3, 5), "spectral_correlation shape")
    method = inv["method"]
    check("pca" in str(method), f"method describe PCA: {method}")

    # ------------------------------------------------------------------
    # 2. DMD (modos complejos) y PCA+ICA
    # ------------------------------------------------------------------
    for stage2 in ("dmd", "pca_ica"):
        out_d = out_root / stage2
        w_d, _, _, _ = compute_channel_influence_on_latent(
            raw, latent, make_meta(stage2, W), out_dir=out_d
        )
        sims = [spearman_abs(w_d[:, d], np.abs(W[:, d])) for d in range(3)]
        check(min(sims) > 0.95, f"pesos {stage2} recuperan W (spearman mínimo {min(sims):.3f})")

    # ------------------------------------------------------------------
    # 3. Diffusion Maps (no lineal) → fallback graceful por correlación
    # ------------------------------------------------------------------
    meta_dm = make_meta("diffusion_maps", W)
    out_dm = out_root / "diffusion_maps"
    w_dm, _, _, dm_path = compute_channel_influence_on_latent(raw, latent, meta_dm, out_dir=out_dm)
    check(w_dm.shape == (N_CH, 3), "diffusion_maps: fallback produce pesos")
    check("correlation" in str(load_psd_data(dm_path)["method"]),
          "diffusion_maps: method registra el fallback por correlación")

    # ------------------------------------------------------------------
    # 4. Sin montaje (caso CSV/Ludovico) → barras
    # ------------------------------------------------------------------
    raw_nomont, W2 = make_raw(with_montage=False, seed=1)
    latent2 = make_latent(raw_nomont.get_data(), W2)
    out_nm = out_root / "sin_montaje"
    compute_and_plot_raw_psd(raw_nomont, out_dir=out_nm)
    w_nm, _, fig_nm, _ = compute_channel_influence_on_latent(
        raw_nomont, latent2, make_meta("pca", W2), out_dir=out_nm
    )
    check(fig_nm.exists(), "sin montaje: figura de barras generada")

    # ------------------------------------------------------------------
    # 5. n_dim de 2 a 5 + fmax por encima de Nyquist (clamp)
    # ------------------------------------------------------------------
    for n_dim in (2, 4, 5):
        lat = latent[:, :3] @ np.random.default_rng(2).normal(size=(3, n_dim))
        out_l = out_root / f"latent_{n_dim}d"
        pl, fl, ml, sl, _ = compute_and_plot_latent_psd(lat, SFREQ, out_l, fmax=200.0)
        check(pl.shape[0] == n_dim, f"latent PSD con n_dim={n_dim}")
        check(fl.max() <= SFREQ / 2 + 1e-6, "fmax clampado a Nyquist")

    # ------------------------------------------------------------------
    # 6. Paginación con 70 canales (> 64)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(3)
    n70 = 70
    names70 = [f"E{i + 1}" for i in range(n70)]
    data70 = rng.standard_normal((n70, int(SFREQ * 8))) * 1e-6
    raw70 = mne.io.RawArray(data70, mne.create_info(names70, SFREQ, "eeg"), verbose=False)
    out70 = out_root / "paginado"
    p70, _, ch70, *_ = compute_and_plot_raw_psd(raw70, out_dir=out70, fmax=60.0)
    check(p70.shape[0] == n70 and len(ch70) == n70, "70 canales procesados")
    check((out70 / "raw_psd_all_channels_p01.png").exists()
          and (out70 / "raw_psd_all_channels_p02.png").exists(),
          "paginación en 2 figuras para >64 canales")

    # ------------------------------------------------------------------
    # 7. Meta con la estructura REAL de extract_latent_space
    #    (preprocessing sin ch_names; con kept_indices y X_filtered)
    # ------------------------------------------------------------------
    from types import SimpleNamespace

    raw7, W7 = make_raw(with_montage=True, seed=4)
    data7 = raw7.get_data()
    n_times7 = data7.shape[1]
    latent7 = make_latent(data7, W7)

    # 7a. stage2 guarda un estimador ICA (mixing_matrix_) en vez de arrays
    meta_ica = {
        "pipeline": {"stage1": None, "stage2": "pca_ica", "stage3": "top_n"},
        "stage1": {"embedding": None, "n_time_lost": 0},
        "stage2": {"ica": SimpleNamespace(mixing_matrix_=W7),
                   "singular_values": np.ones(3), "branch": "mne_ica"},
        "stage3": {"selected_indices": [0, 1, 2], "scores": {}},
        "selected_indices": [0, 1, 2],
        "scoring_method": "none+pca_ica+top_n",
        "latent_scores": {},
        "preprocessing": {"l_freq": 1.0, "h_freq": 40.0, "sfreq": SFREQ, "dt": 1 / SFREQ,
                          "D": 3, "T": n_times7, "n_samples_latent": n_times7,
                          "n_time_lost_by_embedding": 0, "excluded_indices": [],
                          "kept_indices": list(range(N_CH)),
                          "n_channels": N_CH, "n_times": n_times7},
        "Y": latent7.T, "Y_shape": latent7.T.shape, "elapsed_time": 1.0,
    }
    out7a = out_root / "meta_real_ica"
    w7a, ch7a, _, p7a = compute_channel_influence_on_latent(raw7, latent7, meta_ica, out_dir=out7a)
    check(w7a.shape == (N_CH, 3), "meta real: pesos desde estimador ICA (mixing_matrix_)")
    check("mixing_matrix_" in str(load_psd_data(p7a)["method"]), "method registra mixing_matrix_")
    check(ch7a == CH_NAMES, "meta real: ch_names sin ch_names en preprocessing")
    sims = [spearman_abs(w7a[:, d], np.abs(W7[:, d])) for d in range(3)]
    check(min(sims) > 0.95, f"meta real ICA recupera W (spearman mínimo {min(sims):.3f})")

    # 7b. Fallback correlación con X_filtered y canales excluidos (kept_indices)
    excluded = [2, 5]
    kept = [i for i in range(N_CH) if i not in excluded]
    X_filt = data7 * 0.9  # simula la matriz filtrada del pipeline
    meta_dm7 = {
        "pipeline": {"stage1": "hankel", "stage2": "diffusion_maps", "stage3": "top_n"},
        "stage1": {"embedding": "hankel", "n_time_lost": 0, "depth": 1},
        "stage2": {"sigma_used": 1.0, "k_neighbors": 100},
        "stage3": {"selected_indices": [0, 1, 2], "scores": {}},
        "selected_indices": [0, 1, 2],
        "scoring_method": "hankel+diffusion_maps+top_n",
        "latent_scores": {},
        "preprocessing": {"l_freq": 1.0, "h_freq": 40.0, "sfreq": SFREQ, "dt": 1 / SFREQ,
                          "D": 3, "T": n_times7, "n_samples_latent": n_times7,
                          "n_time_lost_by_embedding": 0,
                          "excluded_indices": excluded, "kept_indices": kept,
                          "n_channels": len(kept), "n_times": n_times7,
                          "X_filtered": X_filt[kept]},
        "Y": latent7.T, "Y_shape": latent7.T.shape, "elapsed_time": 1.0,
    }
    out7b = out_root / "meta_real_dm"
    w7b, ch7b, _, p7b = compute_channel_influence_on_latent(raw7, latent7, meta_dm7, out_dir=out7b)
    m7b = str(load_psd_data(p7b)["method"])
    check(w7b.shape == (len(kept), 3), "fallback X_filtered: pesos sobre canales retenidos")
    check(ch7b == [CH_NAMES[i] for i in kept], "fallback X_filtered: nombres vía kept_indices")
    check("correlation" in m7b and "X_filtered" in m7b, "method registra correlación + X_filtered")
    # Los canales con mayor peso verdadero deben correlacionar más
    top_true = int(np.argmax(np.abs(W7[kept, 0])))
    check(w7b[top_true, 0] >= np.median(w7b[:, 0]), "fallback: ranking de influencia razonable")

    print("\nTodos los checks pasaron.")


if __name__ == "__main__":
    main()

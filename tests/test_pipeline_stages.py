"""
Tests del pipeline modular de 3 etapas (Embedding → Dinámica → Selección).

Verifica que:
    1. Todas las combinaciones de etapas llegan hasta el final sin errores
       y producen shapes coherentes (regla (n_features, n_samples)
       interna; latent final (n_samples, n_dim)).
    2. Los empates críticos se comportan según la especificación
       (Hankel→ICA sklearn, Hankel→DM NLSA, DMD sin Hankel, parte real).
    3. La API legacy (``scoring_method=...``) se mapea correctamente y
       sigue funcionando.
    4. La metadata cumple el contrato (claves pipeline/stage1/2/3 +
       claves legacy selected_indices/latent_scores/preprocessing/...).
    5. Las combinaciones inválidas lanzan ValueError con mensaje claro.

Ejecutar con:  pytest tests/test_pipeline_stages.py -v
(o directamente: python tests/test_pipeline_stages.py)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Make the repo root importable (src package)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mne

from src.latent_space_extraction.extract_latent_subspace import (
    extract_latent_space,
    map_legacy_scoring_method,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic EEG
# ---------------------------------------------------------------------------

SFREQ = 250.0
N_CH = 8
N_TIMES = 3000  # 12 s


def _synthetic_raw(n_ch: int = N_CH, n_times: int = N_TIMES, sfreq: float = SFREQ) -> mne.io.RawArray:
    """Multi-channel synthetic EEG: alpha/theta oscillations + pink-ish noise."""
    rng = np.random.default_rng(7)
    t = np.arange(n_times) / sfreq
    data = np.zeros((n_ch, n_times))
    for ch in range(n_ch):
        f1 = 8.0 + 0.5 * ch          # alpha-ish
        f2 = 4.0 + 0.3 * ch          # theta-ish
        data[ch] = (
            np.sin(2 * np.pi * f1 * t)
            + 0.6 * np.sin(2 * np.pi * f2 * t + 0.4 * ch)
            + 0.3 * rng.standard_normal(n_times)
        )
    # Common spatial mixing so channels are correlated
    mix = rng.standard_normal((n_ch, n_ch)) * 0.1 + np.eye(n_ch)
    data = mix @ data
    data *= 1e-6  # EEG scale (volts)

    ch_names = [f"E{ch + 1}" for ch in range(n_ch)]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture(scope="module")
def raw() -> mne.io.RawArray:
    return _synthetic_raw()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_meta_contract(meta: dict, n_dim: int, expected_chain: str) -> None:
    """Contrato de metadata: claves nuevas + legacy (Empate 6)."""
    # New pipeline traceability
    assert "pipeline" in meta
    for key in ("stage1", "stage2", "stage3", "params"):
        assert key in meta["pipeline"]
    assert meta["pipeline"]["chain"] == expected_chain
    for key in ("stage1", "stage2", "stage3"):
        assert key in meta, f"meta['{key}'] missing"

    # Legacy top-level keys
    for key in ("selected_indices", "scoring_method", "latent_scores",
                "preprocessing", "Y", "Y_shape", "elapsed_time"):
        assert key in meta, f"legacy key meta['{key}'] missing"

    assert len(meta["selected_indices"]) == n_dim
    assert meta["Y_shape"] == tuple(meta["Y"].shape)
    assert meta["Y"].shape[0] >= n_dim  # D >= n_dim
    assert meta["elapsed_time"] > 0
    # Stage-2 output must be real (Empate 5.1)
    assert np.isrealobj(meta["Y"])


def _check_latent(latent: np.ndarray, n_dim: int, expected_samples: int) -> None:
    assert latent.shape == (expected_samples, n_dim)
    assert np.isrealobj(latent)
    assert np.all(np.isfinite(latent))


# ---------------------------------------------------------------------------
# 1. Legacy mapping (unit)
# ---------------------------------------------------------------------------

class TestLegacyMapping:
    def test_markov_maps_to_ica_markov_fastest(self):
        spec = map_legacy_scoring_method("markov", n_dim=2, n_bins=7)
        assert spec["stage1_embedding"] is None
        assert spec["stage2_dynamics"] == "pca_ica"
        assert spec["stage3_selection"] == "markov_fastest"
        assert spec["stage3_params"]["n_bins"] == 7

    def test_markov_inverted_maps_to_markov_slowest(self):
        spec = map_legacy_scoring_method("markov_inverted", n_dim=2)
        assert spec["stage3_selection"] == "markov_slowest"

    def test_hankel_dmd_maps_to_hankel_dmd_topn(self):
        spec = map_legacy_scoring_method("hankel_dmd", n_dim=3,
                                         hankel_embedding_depth=60)
        assert spec["stage1_embedding"] == "hankel"
        assert spec["stage1_params"]["depth"] == 60
        assert spec["stage2_dynamics"] == "dmd"
        # rank = n_dim → reproduces legacy numerics
        assert spec["stage2_params"]["rank"] == 3
        assert spec["stage3_selection"] == "top_n"

    def test_diffusion_maps_maps_to_dm_topn(self):
        spec = map_legacy_scoring_method("diffusion_maps", n_dim=2,
                                         diffusion_k=80)
        assert spec["stage1_embedding"] is None
        assert spec["stage2_dynamics"] == "diffusion_maps"
        assert spec["stage2_params"]["n_components"] == 2
        assert spec["stage2_params"]["k"] == 80
        assert spec["stage3_selection"] == "top_n"

    def test_fc_methods_map_to_legacy_fc(self):
        for method in ("conservative", "weighted", "sequential",
                       "pareto", "independent"):
            spec = map_legacy_scoring_method(method, n_dim=2, alpha=0.5)
            assert spec["stage2_dynamics"] == "pca_ica"
            assert spec["stage3_selection"] == "legacy_fc"
            assert spec["stage3_params"]["method"] == method

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown legacy scoring_method"):
            map_legacy_scoring_method("does_not_exist")


# ---------------------------------------------------------------------------
# 2. End-to-end: new API combinations (no Hankel)
# ---------------------------------------------------------------------------

class TestNoHankelCombinations:
    def test_pca_topn(self, raw):
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding=None,
            stage2_dynamics="pca",
            stage2_params={"n_components": 6},
            stage3_selection="top_n",
            verbose=False,
        )
        _check_latent(latent, 2, N_TIMES)
        _check_meta_contract(meta, 2, "none+pca+top_n")
        # PCA: rows must be ordered by singular value (top-n = first rows)
        assert meta["selected_indices"] == [0, 1]

    def test_pca_markov_fastest(self, raw):
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding=None,
            stage2_dynamics="pca",
            stage2_params={"n_components": 5},
            stage3_selection="markov_fastest",
            stage3_params={"n_bins": 4, "search_strategy": "exhaustive"},
            verbose=False,
        )
        _check_latent(latent, 2, N_TIMES)
        _check_meta_contract(meta, 2, "none+pca+markov_fastest")
        assert np.isfinite(meta["latent_scores"]["tau"])

    def test_pca_markov_slowest(self, raw):
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding=None,
            stage2_dynamics="pca",
            stage2_params={"n_components": 5},
            stage3_selection="markov_slowest",
            stage3_params={"n_bins": 4},
            verbose=False,
        )
        _check_latent(latent, 2, N_TIMES)
        assert meta["stage3"]["scores"]["maximize"] is True

    def test_dmd_direct_topn(self, raw):
        """Empate 4.2 — DMD directo sobre X (AR-1 espacial)."""
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding=None,
            stage2_dynamics="dmd",
            stage2_params={"rank": 5},
            stage3_selection="top_n",
            verbose=False,
        )
        _check_latent(latent, 2, N_TIMES)
        _check_meta_contract(meta, 2, "none+dmd+top_n")
        assert meta["stage2"]["branch"] == "direct_ar1_channel_space"
        assert "frequencies_hz" in meta["latent_scores"]

    def test_diffusion_maps_topn(self, raw):
        """Empate 3 — DM sobre instantes temporales (puntos en R^{N_c})."""
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding=None,
            stage2_dynamics="diffusion_maps",
            stage2_params={"n_components": 4, "k": 50},
            stage3_selection="top_n",
            verbose=False,
        )
        _check_latent(latent, 2, N_TIMES)
        _check_meta_contract(meta, 2, "none+diffusion_maps+top_n")
        assert meta["stage2"]["branch"] == "dm_over_time_instants"
        assert meta["latent_scores"]["sigma_used"] is not None  # BGH

    def test_diffusion_maps_markov(self, raw):
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding=None,
            stage2_dynamics="diffusion_maps",
            stage2_params={"n_components": 4, "k": 50},
            stage3_selection="markov_fastest",
            stage3_params={"n_bins": 4},
            verbose=False,
        )
        _check_latent(latent, 2, N_TIMES)
        assert np.isfinite(meta["latent_scores"]["tau"])


# ---------------------------------------------------------------------------
# 3. End-to-end: Hankel combinations
# ---------------------------------------------------------------------------

class TestHankelCombinations:
    DEPTH = 30
    EFF_SAMPLES = N_TIMES - DEPTH + 1

    def test_hankel_pca_topn(self, raw):
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding="hankel",
            stage1_params={"depth": self.DEPTH},
            stage2_dynamics="pca",
            stage2_params={"n_components": 6},
            stage3_selection="top_n",
            verbose=False,
        )
        _check_latent(latent, 2, self.EFF_SAMPLES)
        _check_meta_contract(meta, 2, "hankel+pca+top_n")
        # Hankel shape: (N_c·T, N_t−T+1)
        assert meta["stage1"]["output_shape"] == (N_CH * self.DEPTH, self.EFF_SAMPLES)
        assert meta["preprocessing"]["embedding_depth"] == self.DEPTH

    def test_hankel_pca_markov_slowest(self, raw):
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding="hankel",
            stage1_params={"depth": self.DEPTH},
            stage2_dynamics="pca",
            stage2_params={"n_components": 5},
            stage3_selection="markov_slowest",
            stage3_params={"n_bins": 4},
            verbose=False,
        )
        _check_latent(latent, 2, self.EFF_SAMPLES)
        assert np.isfinite(meta["latent_scores"]["tau"])

    def test_hankel_dmd_topn(self, raw):
        """Empate 4.1 — eeg_hankel_dmd_core directamente."""
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding="hankel",
            stage1_params={"depth": self.DEPTH},
            stage2_dynamics="dmd",
            stage2_params={"rank": 6},
            stage3_selection="top_n",
            verbose=False,
        )
        _check_latent(latent, 2, self.EFF_SAMPLES)
        _check_meta_contract(meta, 2, "hankel+dmd+top_n")
        assert meta["stage2"]["branch"] == "hankel_eeg_hankel_dmd_core"
        assert "frequencies_hz" in meta["latent_scores"]
        assert "damping_rates" in meta["latent_scores"]
        assert len(meta["latent_scores"]["singular_values"]) == 6

    def test_hankel_pca_ica_sklearn_markov(self, raw):
        """Empate 1 — SVD de H + FastICA de sklearn (sin MNE ICA/ICLabel)."""
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding="hankel",
            stage1_params={"depth": self.DEPTH},
            stage2_dynamics="pca_ica",
            stage2_params={"n_components": 8, "ica_solver": "fastica"},
            stage3_selection="markov_fastest",
            stage3_params={"n_bins": 4},
            verbose=False,
        )
        _check_latent(latent, 2, self.EFF_SAMPLES)
        _check_meta_contract(meta, 2, "hankel+pca_ica+markov_fastest")
        assert meta["stage2"]["branch"] == "svd_whitening_fastica_sklearn"
        assert meta["stage2"]["svd_rank"] == 8

    def test_hankel_diffusion_maps_nlsa(self, raw):
        """Empate 2 — NLSA: SVD de H a svd_rank + DM sobre V_r Σ_r."""
        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding="hankel",
            stage1_params={"depth": self.DEPTH},
            stage2_dynamics="diffusion_maps",
            stage2_params={"svd_rank": 20, "n_components": 4, "k": 50},
            stage3_selection="top_n",
            verbose=False,
        )
        _check_latent(latent, 2, self.EFF_SAMPLES)
        _check_meta_contract(meta, 2, "hankel+diffusion_maps+top_n")
        assert meta["stage2"]["branch"] == "nlsa_svd_then_dm"
        assert meta["stage2"]["svd_rank"] == 20
        # DM operates in R^{svd_rank}, not in R^{N_c·T}
        assert meta["latent_scores"]["sigma_used"] is not None  # BGH


# ---------------------------------------------------------------------------
# 4. Legacy API end-to-end (backward compatibility)
# ---------------------------------------------------------------------------

class TestLegacyAPI:
    DEPTH = 30

    def test_legacy_hankel_dmd(self, raw):
        """scoring_method='hankel_dmd' → hankel + dmd(rank=n_dim) + top_n."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            latent, meta = extract_latent_space(
                raw, n_dim=2,
                scoring_method="hankel_dmd",
                hankel_embedding_depth=self.DEPTH,
                verbose=False,
            )
        _check_latent(latent, 2, N_TIMES - self.DEPTH + 1)
        # Legacy keys preserved
        assert meta["scoring_method"] == "hankel_dmd"
        assert meta["selected_indices"] == [0, 1]
        assert "frequencies_hz" in meta["latent_scores"]
        assert "damping_rates" in meta["latent_scores"]
        assert meta["preprocessing"]["embedding_depth"] == self.DEPTH
        # New traceability keys also present
        assert meta["pipeline"]["chain"] == "hankel+dmd+top_n"

    def test_legacy_hankel_dmd_auto_depth(self, raw):
        """depth=None → auto clip(sfreq*0.25, 50, 200) = 62 (a 250 Hz)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            latent, meta = extract_latent_space(
                raw, n_dim=2,
                scoring_method="hankel_dmd",
                hankel_embedding_depth=None,
                verbose=False,
            )
        auto_depth = int(np.clip(SFREQ * 0.25, 50, 200))
        assert meta["preprocessing"]["embedding_depth"] == auto_depth
        _check_latent(latent, 2, N_TIMES - auto_depth + 1)

    def test_legacy_diffusion_maps(self, raw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            latent, meta = extract_latent_space(
                raw, n_dim=2,
                scoring_method="diffusion_maps",
                diffusion_k=50,
                verbose=False,
            )
        _check_latent(latent, 2, N_TIMES)
        assert meta["scoring_method"] == "diffusion_maps"
        assert meta["pipeline"]["chain"] == "none+diffusion_maps+top_n"

    def test_legacy_emits_deprecation_warning(self, raw):
        with pytest.warns(DeprecationWarning):
            extract_latent_space(
                raw, n_dim=2,
                scoring_method="hankel_dmd",
                hankel_embedding_depth=20,
                verbose=False,
            )


# ---------------------------------------------------------------------------
# 5. MNE-ICA branch wiring (mocked preprocessing — ICLabel needs models)
# ---------------------------------------------------------------------------

class TestMNEICABranch:
    def _fake_preprocessing(self, D: int = 6):
        def _fake(raw, *, l_freq, h_freq, n_components, ica_method,
                  ica_random_state, retained_labels, verbose):
            rng = np.random.default_rng(3)
            n_times = raw.n_times
            Y = rng.standard_normal((D, n_times))
            Y -= Y.mean(axis=1, keepdims=True)
            return {
                "Y": Y,
                "ica": None,
                "kept_indices": list(range(D)),
                "excluded_indices": [],
                "labels": {"labels": ["brain"] * D},
                "raw_filtered": raw,
                "D": D,
                "T": n_times,
                "n_channels": len(raw.ch_names),
                "sfreq": raw.info["sfreq"],
            }
        return _fake

    def test_pca_ica_mne_branch_markov_fastest(self, raw, monkeypatch):
        """Empate: sin Hankel → ICA de MNE; verifica el cableado completo."""
        import src.latent_space_extraction.eeg_preprocessing as pre
        monkeypatch.setattr(pre, "run_full_preprocessing", self._fake_preprocessing(D=6))

        latent, meta = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding=None,
            stage2_dynamics="pca_ica",
            stage2_params={"n_components": None, "ica_method": "picard"},
            stage3_selection="markov_fastest",
            stage3_params={"n_bins": 4, "search_strategy": "exhaustive"},
            verbose=False,
        )
        _check_latent(latent, 2, N_TIMES)
        _check_meta_contract(meta, 2, "none+pca_ica+markov_fastest")
        assert meta["stage2"]["branch"] == "mne_ica_iclabel"
        assert np.isfinite(meta["latent_scores"]["tau"])

    def test_legacy_markov_routes_to_mne_branch(self, raw, monkeypatch):
        import src.latent_space_extraction.eeg_preprocessing as pre
        monkeypatch.setattr(pre, "run_full_preprocessing", self._fake_preprocessing(D=5))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            latent, meta = extract_latent_space(
                raw, n_dim=2,
                scoring_method="markov",
                n_bins=4,
                verbose=False,
            )
        _check_latent(latent, 2, N_TIMES)
        assert meta["scoring_method"] == "markov"
        assert meta["pipeline"]["chain"] == "none+pca_ica+markov_fastest"


# ---------------------------------------------------------------------------
# 6. Validation of invalid combinations
# ---------------------------------------------------------------------------

class TestValidation:
    def test_unknown_stage1(self, raw):
        with pytest.raises(ValueError, match="Unknown embedding"):
            extract_latent_space(raw, n_dim=2, stage1_embedding="wavelet",
                                 verbose=False)

    def test_unknown_stage2(self, raw):
        with pytest.raises(ValueError, match="Unknown dynamics"):
            extract_latent_space(raw, n_dim=2, stage2_dynamics="emd",
                                 verbose=False)

    def test_unknown_stage3(self, raw):
        with pytest.raises(ValueError, match="Unknown selection"):
            extract_latent_space(raw, n_dim=2, stage2_dynamics="pca",
                                 stage3_selection="random", verbose=False)

    def test_identity_with_params_raises(self, raw):
        with pytest.raises(ValueError, match="takes no parameters"):
            extract_latent_space(raw, n_dim=2, stage1_embedding=None,
                                 stage1_params={"depth": 10}, verbose=False)

    def test_hankel_depth_too_large(self, raw):
        with pytest.raises(ValueError, match="embedding depth"):
            extract_latent_space(raw, n_dim=2, stage1_embedding="hankel",
                                 stage1_params={"depth": N_TIMES + 10},
                                 stage2_dynamics="pca", verbose=False)

    def test_n_components_below_n_dim(self, raw):
        with pytest.raises(ValueError, match="must be >= n_dim"):
            extract_latent_space(raw, n_dim=4, stage2_dynamics="pca",
                                 stage2_params={"n_components": 3},
                                 verbose=False)

    def test_pca_ica_hankel_rejects_mne_solver(self, raw):
        with pytest.raises(ValueError, match="ica_solver"):
            extract_latent_space(raw, n_dim=2, stage1_embedding="hankel",
                                 stage1_params={"depth": 30},
                                 stage2_dynamics="pca_ica",
                                 stage2_params={"ica_solver": "picard"},
                                 verbose=False)

    def test_pca_ica_without_raw_raises(self):
        """stage2='pca_ica' sin Hankel sobre datos que no son mne.Raw."""
        from src.latent_space_extraction.pipeline import (
            PipelineContext, build_stage2,
        )
        stage2 = build_stage2("pca_ica", {})
        ctx = PipelineContext(raw=None, sfreq=250.0, dt=1 / 250.0)
        with pytest.raises(ValueError, match="requires an mne.Raw"):
            stage2.fit_transform(np.zeros((4, 100)), ctx=ctx, n_dim=2)


# ---------------------------------------------------------------------------
# 7. Reproducibility / equivalence checks
# ---------------------------------------------------------------------------

class TestEquivalence:
    def test_new_api_matches_legacy_hankel_dmd(self, raw):
        """La nueva API con rank=n_dim reproduce la rama legacy hankel_dmd."""
        depth = 30
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            latent_legacy, _ = extract_latent_space(
                raw, n_dim=2, scoring_method="hankel_dmd",
                hankel_embedding_depth=depth, verbose=False,
            )
        latent_new, _ = extract_latent_space(
            raw, n_dim=2,
            stage1_embedding="hankel", stage1_params={"depth": depth},
            stage2_dynamics="dmd", stage2_params={"rank": 2},
            stage3_selection="top_n", verbose=False,
        )
        assert latent_new.shape == latent_legacy.shape
        # Mismo subespacio POD (signo indeterminado por SVD)
        corr = [abs(np.corrcoef(latent_new[:, i], latent_legacy[:, i])[0, 1])
                for i in range(2)]
        assert all(c > 0.999 for c in corr), f"correlations: {corr}"

    def test_topn_deterministic(self, raw):
        """La SVD truncada debe ser determinista (convención de signo
        canónica vía svd_flip): 3 ejecuciones deben coincidir exactamente."""
        kw = dict(n_dim=2, stage1_embedding=None, stage2_dynamics="pca",
                  stage2_params={"n_components": 5}, stage3_selection="top_n",
                  verbose=False)
        l1, _ = extract_latent_space(raw.copy(), **kw)
        for _ in range(2):
            l2, _ = extract_latent_space(raw.copy(), **kw)
            np.testing.assert_allclose(l1, l2, rtol=1e-8, atol=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))

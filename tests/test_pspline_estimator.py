"""Tests del estimador P-splines de coeficientes de Kramers-Moyal.

Cubre el estimador global (PSplineKMEstimator), el adaptador del pipeline
(``extract_km_coefficients`` con ``estimation_method='p_splines'``) y la
compatibilidad de su salida con la etapa de reconstrucción IgA.

Proceso de referencia: Ornstein-Uhlenbeck 2D con deriva teórica conocida
f(x) = -THETA x, lo que permite validar la precisión del ajuste.
"""
import numpy as np
import pytest

from src.potential_reconstruction.pspline_estimator import (
    PSplineConfig,
    PSplineTensorBasis,
    PSplineKMEstimator,
)
from src.potential_reconstruction.km_tools_v2 import (
    extract_km_coefficients,
    estimate_drift_diffusion,
)
from src.potential_reconstruction.iga_reconstructor import (
    reconstruct_potential_iga,
)

THETA = np.array([[1.0, 0.2], [0.2, 0.8]])
SIGMA = np.diag([0.5, 0.3])
DT_SIM = 0.01


def simulate_ou_2d(n_steps=60_000, dt=DT_SIM, seed=7):
    """Trayectoria 2D del OU dX = -THETA X dt + SIGMA dW (Euler-Maruyama)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_steps, 2))
    sqrt_dt = np.sqrt(dt)
    for i in range(1, n_steps):
        X[i] = (X[i - 1]
                - THETA @ X[i - 1] * dt
                + SIGMA @ rng.standard_normal(2) * sqrt_dt)
    return X


def true_drift_grid(centers):
    """Deriva teórica f(x) = -THETA x sobre puntos (..., 2)."""
    return -np.einsum('ij,...j->...i', THETA, centers)


def grid_centers(edges):
    """Puntos de la grilla (n1, ..., nD, D) a partir de centros por dim."""
    mesh = np.meshgrid(*edges, indexing='ij')
    return np.stack(mesh, axis=-1)


def empirical_density(data, edges):
    """Histograma sobre las celdas centradas en ``edges`` (formato pipeline)."""
    hist_edges = []
    for c in edges:
        half = (c[1] - c[0]) / 2.0 if len(c) > 1 else 0.5
        hist_edges.append(np.concatenate([[c[0] - half], c + half]))
    density, _ = np.histogramdd(data, bins=hist_edges)
    return density.astype(float)


@pytest.fixture(scope='module')
def ou2d():
    return simulate_ou_2d()


@pytest.fixture(scope='module')
def fitted(ou2d):
    est = PSplineKMEstimator()
    est.fit(ou2d, dt=DT_SIM)
    return est


class TestPSplineEstimator:
    """Estimador global: ajuste, evaluación y propiedades de D."""

    def test_fit_and_predict(self, fitted, ou2d):
        f_pred, D_pred = fitted.predict(ou2d[:128])
        assert f_pred.shape == (128, 2)
        assert D_pred.shape == (128, 2, 2)
        assert np.all(np.isfinite(f_pred))
        assert np.all(np.isfinite(D_pred))
        # Simetría de D en cada punto
        assert np.allclose(D_pred, np.swapaxes(D_pred, -1, -2))

    def test_evaluate_grid_shapes(self, fitted):
        edges = fitted.grid_edges([25, 30])
        assert len(edges) == 2 and len(edges[0]) == 25 and len(edges[1]) == 30
        drift, diffusion = fitted.evaluate_grid(edges)
        assert drift.shape == (2, 25, 30)
        assert diffusion.shape == (2, 2, 25, 30)
        assert np.all(np.isfinite(drift))
        assert np.all(np.isfinite(diffusion))

    def test_diffusion_spd_on_grid(self, fitted):
        edges = fitted.grid_edges([12, 12])
        _, diffusion = fitted.evaluate_grid(edges)
        Dm = np.moveaxis(diffusion, (0, 1), (-2, -1))
        eigvals = np.linalg.eigvalsh(Dm)
        assert np.all(eigvals > 0)

    def test_drift_accuracy(self, fitted, ou2d):
        """El ajuste reproduce la deriva teórica en la región con soporte.

        Sobre celdas con densidad >= 5% del pico:
          (a) correlación componente a componente con f teórica > 0.95;
          (b) error relativo global ||f_est - f_true|| / ||f_true|| < 0.20.

        La cota (b) refleja el ruido estadístico inherente a la regresión
        (varianza ~ sigma^2/dt por transición): el error escala como
        1/sqrt(N dt) con N muestras, y para este OU (60k pasos, dt=0.01)
        incluso el ajuste lineal de mínimos cuadrados global —la solución
        exacta para esta verdad— incurre en ~10% de error mediano por
        celda al normalizar por |f|, que se anula en el origen.
        """
        edges = fitted.grid_edges([30, 30])
        drift, _ = fitted.evaluate_grid(edges)
        drift_field = np.moveaxis(drift, 0, -1)  # (n1, n2, 2)
        centers = grid_centers(edges)
        f_true = true_drift_grid(centers)

        density = empirical_density(ou2d, edges)
        mask = density >= 0.05 * density.max()
        est_m = drift_field[mask]
        true_m = f_true[mask]

        # (a) correlación por componente
        for d in range(2):
            r = np.corrcoef(est_m[:, d], true_m[:, d])[0, 1]
            assert r > 0.95, f"correlación deriva dim {d}: {r:.3f}"

        # (b) error relativo global (Frobenius)
        rel_global = np.linalg.norm(est_m - true_m) / np.linalg.norm(true_m)
        assert rel_global < 0.20, f"error relativo global: {rel_global:.3f}"

    def test_config_validation(self, ou2d):
        # Clave desconocida en el dict de configuración
        with pytest.raises(ValueError):
            PSplineConfig.from_any({'clave_inexistente': 1})
        # n_basis debe ser >= degree + 1
        with pytest.raises(ValueError):
            PSplineTensorBasis(dim=2, n_basis=3, degree=3)
        # dt no positivo
        with pytest.raises(ValueError):
            PSplineKMEstimator().fit(ou2d, dt=-1.0)
        # data con forma inválida
        with pytest.raises(ValueError):
            PSplineKMEstimator().fit(ou2d[:, 0], dt=DT_SIM)

    def test_max_tensor_bases_guard(self, ou2d):
        cfg = PSplineConfig(n_basis=50, max_tensor_bases=100)
        with pytest.raises(ValueError, match='max_tensor_bases'):
            PSplineKMEstimator(cfg).fit(ou2d, dt=DT_SIM)


class TestPipelineIntegration:
    """Adaptador del pipeline y compatibilidad con la etapa IgA."""

    def test_adapter_psplines_returns_three_or_four(self, ou2d):
        out = extract_km_coefficients(
            ou2d, bins=[15, 15], dt=DT_SIM, estimation_method='p_splines')
        assert len(out) == 3
        drift, diffusion, edges = out
        assert drift.shape == (2, 15, 15)
        assert diffusion.shape == (2, 2, 15, 15)
        assert len(edges) == 2 and all(len(e) == 15 for e in edges)

        out4 = extract_km_coefficients(
            ou2d, bins=[15, 15], dt=DT_SIM, estimation_method='p_splines',
            return_estimation_info=True)
        assert len(out4) == 4
        info = out4[3]
        assert info['method'] == 'p_splines'
        assert 'lambda_drift' in info and 'n_basis' in info

    def test_adapter_nadaraya_watson_default(self, ou2d):
        """Sin estimation_method la ruta histórica NW sigue siendo default."""
        out = extract_km_coefficients(ou2d, bins=[15, 15], dt=DT_SIM)
        assert len(out) == 3
        drift, diffusion, edges = out
        assert drift.shape == (2, 15, 15)
        assert diffusion.shape == (2, 2, 15, 15)

    def test_adapter_invalid_method(self, ou2d):
        with pytest.raises(ValueError, match='estimation_method'):
            extract_km_coefficients(
                ou2d, bins=[15, 15], dt=DT_SIM,
                estimation_method='metodo_inexistente')

    def test_wrapper_estimate_drift_diffusion(self, ou2d):
        out = estimate_drift_diffusion(
            ou2d, bins=[15, 15], dt=DT_SIM, estimation_method='p_splines')
        assert len(out) == 3
        out4 = estimate_drift_diffusion(
            ou2d, bins=[15, 15], dt=DT_SIM, estimation_method='p_splines',
            return_info=True)
        assert len(out4) == 4 and out4[3]['method'] == 'p_splines'

    def test_reconstruct_potential_compatibility(self, ou2d):
        """La salida P-splines alimenta directamente reconstruct_potential_iga."""
        drift, diffusion, edges = extract_km_coefficients(
            ou2d, bins=[12, 12], dt=DT_SIM, estimation_method='p_splines')
        density = empirical_density(ou2d, edges)
        result = reconstruct_potential_iga(
            drift, diffusion, edges, density,
            degree=2, decompose_helmholtz=True,
        )
        assert result['potential'].shape == (12, 12)
        mask = result['density_mask']
        assert mask.any()
        assert np.all(np.isfinite(result['potential'][mask]))
        assert 'eta' in result and np.isfinite(result['eta'])

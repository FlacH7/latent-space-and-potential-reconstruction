"""
pspline_estimator.py
====================
Estimación global de los coeficientes de Kramers–Moyal (deriva ``f`` y
difusión ``D``) mediante P-splines penalizados tensoriales, directamente
desde pares de observaciones consecutivas ``(X_i, X_{i+1})``, sin
discretización espacial previa en bins.

Modelo
------
Proceso de difusión de Itô:

    dX_t = f(X_t) dt + σ(X_t) dW_t,     D(x) = σ(x) σ(x)ᵀ / 2,

donde ``D`` es el segundo coeficiente de Kramers–Moyal (convención del
repositorio y de ``iga_reconstructor``).

Estimador implementado (``fit_method='moments'``)
-------------------------------------------------
- Deriva: regresión penalizada de ``y_i = ΔX_i / Δt`` sobre una base
  B-spline tensorial, con penalización por diferencias de orden
  ``penalty_order`` entre coeficientes (P-splines de Eilers–Marx).
- Difusión: regresión penalizada de los segundos momentos de los
  residuos ``R_i = ΔX_i − Δt·f̂(X_i)``, es decir
  ``S_i,jk = R_i,j R_i,k / (2Δt)``, uno por componente ``j ≤ k``.
- Reconstrucción SPD: sobre los campos de momentos ya ajustados
  ``M̂_jk(x)`` se aplica
      s_j(x)  = log(max(M̂_jj(x), min_diffusion)),
      r_jk(x) = atanh(clip(M̂_jk(x) / sqrt(M̂_jj M̂_kk), ±correlation_clip)),
      D_jj    = exp(s_j),   ρ_jk = tanh(r_jk),   D_jk = ρ_jk sqrt(D_jj D_kk),
  seguido de un recorte espectral de salvaguarda por celda. Las
  transformaciones se aplican a los campos SUAVIZADOS (no a los momentos
  muestrales individuales, que son de rango 1 y degenerados), por lo que
  el estimador de momentos subyacente permanece insesgado salvo el
  sesgo O(Δt) habitual del esquema de Euler–Maruyama.

Alcance
-------
``fit_method='qmle'`` (cuasi-verosimilitud de transición) está reservado
como extensión futura y NO está implementado en esta versión.

Interfaz con el pipeline IgA
----------------------------
El ajuste se hace sobre las muestras; la entrega al reconstructor IgA se
hace evaluando los campos spline sobre una grilla uniforme de centros de
celda (``grid_edges`` + ``evaluate_grid``), con el mismo formato de
arrays que produce la ruta Nadaraya–Watson de ``km_tools_v2``:

- ``drift``:     ``(D, n1, ..., nD)``
- ``diffusion``: ``(D, D, n1, ..., nD)``
- ``edges``:     lista de ``D`` arrays con los centros de celda.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy.interpolate import BSpline
from scipy import sparse
from scipy.sparse import linalg as spla

__all__ = [
    'PSplineConfig',
    'PSplineTensorBasis',
    'PSplineKMEstimator',
    'estimate_drift_diffusion_psplines',
]

_EPS_RIGHT = np.nextafter(1.0, 0.0)  # design_matrix es cerrado a derecha, pero evitamos el borde exacto


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class PSplineConfig:
    """Hiperparámetros del estimador P-splines.

    Parameters
    ----------
    n_basis : int o Sequence[int]
        Número de funciones B-spline por dimensión. Si es escalar, se usa
        el mismo valor en todas las dimensiones. Debe ser >= degree + 1.
    degree : int
        Grado de los B-splines (típicamente 3, cúbico).
    penalty_order : int
        Orden de las diferencias entre coeficientes en la penalización
        (2 = penaliza curvatura, convención estándar de P-splines).
    lambda_drift : float
        Penalización de suavidad para la deriva. Análogo funcional del
        bandwidth en Nadaraya–Watson: λ → 0 interpola, λ → ∞ fuerza un
        polinomio global de grado ``penalty_order - 1``.
    lambda_diffusion : float
        Penalización de suavidad para los campos de segundo momento.
    domain_quantile : float
        Cuantil robusto para definir el dominio a partir de los datos
        (descarta colas extremas con soporte estadístico pobre).
    domain_padding : float
        Fracción del rango robusto añadida a cada lado del dominio.
    grid_margin : float
        Fracción adicional del rango del dominio robusto añadida por lado
        AL GENERAR LA GRILLA DE SALIDA (``grid_edges``). Su propósito es
        que las celdas del borde de la grilla queden fuera del soporte de
        datos (densidad ~0), de modo que la máscara de densidad de la
        etapa IgA no toque la frontera del dominio de integración; sin ese
        margen vacío, los DOFs de B-spline cercanos al borde quedan casi
        sin restringir y el sistema de Galerkin produce modos espurios
        (picos de magnitud enorme en el potencial). No afecta al ajuste,
        solo a la grilla de evaluación. El valor por defecto (0.30) se
        validó en 2D (doble pozo y proceso OU): márgenes menores pueden
        dejar datos fuera de la grilla cuando las colas exceden el
        dominio robusto.
    clip_to_domain : bool
        Si True, las evaluaciones fuera del dominio se recortan al borde
        (extrapolación constante) en lugar de extrapolar los splines.
    min_diffusion : float
        Cota inferior para varianzas y autovalores de ``D`` (salvaguarda
        de definición positiva).
    correlation_clip : float
        Cota |ρ| <= correlation_clip para las correlaciones.
    chunk_size : int
        Número de transiciones procesadas por bloque (controla memoria).
    max_tensor_bases : int
        Máximo número total de coeficientes tensoriales permitido
        (``prod(n_basis)``). Protege contra explosión de memoria en
        dimensión alta; si se excede, reduce ``n_basis`` o usa una
        representación estructurada (aditiva/sparse-grid, fuera del
        alcance de esta versión).
    fit_method : str
        'moments' (implementado) o 'qmle' (reservado, no implementado).
    direct_solver_max_basis : int
        Umbral de coeficientes para usar factorización directa dispersa;
        por encima se usa gradiente conjugado con precondicionador
        diagonal.
    """
    n_basis: int | Sequence[int] = 8
    degree: int = 3
    penalty_order: int = 2
    lambda_drift: float = 1e-3
    lambda_diffusion: float = 1e-3
    domain_quantile: float = 0.005
    domain_padding: float = 0.05
    grid_margin: float = 0.30
    clip_to_domain: bool = True
    min_diffusion: float = 1e-12
    correlation_clip: float = 0.999
    chunk_size: int = 50_000
    max_tensor_bases: int = 200_000
    fit_method: str = 'moments'
    direct_solver_max_basis: int = 20_000

    @classmethod
    def from_any(cls, config=None, **overrides):
        """Construye un PSplineConfig desde None, dict, otro PSplineConfig,
        y aplica overrides puntuales."""
        if config is None:
            params = {}
        elif isinstance(config, cls):
            params = asdict(config)
        elif isinstance(config, dict):
            valid = set(cls.__dataclass_fields__)
            unknown = set(config) - valid
            if unknown:
                raise ValueError(
                    f"Claves desconocidas en pspline_config: {sorted(unknown)}. "
                    f"Válidas: {sorted(valid)}"
                )
            params = dict(config)
        else:
            raise TypeError(
                f"pspline_config debe ser None, dict o PSplineConfig, no {type(config).__name__}"
            )
        params.update(overrides)
        return cls(**params)


# ---------------------------------------------------------------------------
# Utilidades dispersas
# ---------------------------------------------------------------------------

def _row_kron_csr(A, B):
    """Producto de Kronecker por filas de dos matrices CSR con igual número
    de filas: ``C[i, :] = kron(A[i, :], B[i, :])``.

    La indexación resultante es la de ``np.kron``/``scipy.sparse.kron``:
    el índice de la segunda matriz varía más rápido.
    """
    A = sparse.csr_matrix(A)
    B = sparse.csr_matrix(B)
    n, Ka = A.shape
    Kb = B.shape[1]
    assert B.shape[0] == n, "A y B deben tener el mismo número de filas"

    Ap, Aj, Ax = A.indptr, A.indices, A.data
    Bp, Bj, Bx = B.indptr, B.indices, B.data
    nnzA = np.diff(Ap)
    nnzB = np.diff(Bp)

    # Expansión de las entradas de A: cada entrada de la fila i se repite
    # nnzB[i] veces, en orden — coincide con el layout [a=0: b's | a=1: b's | ...].
    repB = np.repeat(nnzB, nnzA)
    Aj_exp = np.repeat(Aj, repB)
    Ax_exp = np.repeat(Ax, repB)

    nnzC = nnzA * nnzB
    Cp = np.concatenate([[0], np.cumsum(nnzC)])
    block_start = Cp[:-1]

    row_of_B = np.repeat(np.arange(n), nnzB)
    b_pos = np.arange(B.nnz) - Bp[:-1][row_of_B]

    Cj = np.empty(Cp[-1], dtype=np.int64)
    Cx = np.empty(Cp[-1], dtype=np.float64)
    maxA = int(nnzA.max()) if n else 0
    for a in range(maxA):
        sel = nnzA[row_of_B] > a
        rows_sel = row_of_B[sel]
        dest = block_start[rows_sel] + a * nnzB[rows_sel] + b_pos[sel]
        Cj[dest] = Bj[sel]
        Cx[dest] = Bx[sel]

    return sparse.csr_matrix(
        (Ax_exp * Cx, Aj_exp.astype(np.int64) * Kb + Cj, Cp),
        shape=(n, Ka * Kb),
    )


def _difference_penalty(k, order):
    """Matriz de penalización P = ΔᵀΔ con Δ de diferencias de orden ``order``
    sobre ``k`` coeficientes."""
    if order < 1:
        raise ValueError("penalty_order debe ser >= 1")
    if order >= k:
        warnings.warn(
            f"penalty_order={order} >= n_basis={k}: la penalización es nula "
            "en esa dimensión (equivalente a ridge nulo).",
            UserWarning,
        )
        return sparse.csr_matrix((k, k))
    Dm = np.diff(np.eye(k), n=order, axis=0)
    return sparse.csr_matrix(Dm.T @ Dm)


def _solve_penalized(G, P, lam, rhs, direct_threshold=20_000):
    """Resuelve (G + λP) a = rhs con salvaguardas de rango.

    G y P son semidefinidas positivas dispersas; rhs puede ser matriz
    (K, n_targets) densa. Si la factorización directa detecta deficiencia
    de rango, se añade un jitter diagonal creciente.
    """
    K = G.shape[0]
    A = G + lam * P
    scale = float(np.mean(np.abs(A.diagonal()))) if K else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    rhs = np.asarray(rhs, dtype=float)
    if rhs.ndim == 1:
        rhs = rhs[:, None]

    jitter = 0.0
    last_err = None
    for _ in range(6):
        A_j = A if jitter == 0.0 else A + (jitter * scale) * sparse.identity(K, format='csr')
        if K <= direct_threshold:
            with warnings.catch_warnings():
                warnings.simplefilter('error', spla.MatrixRankWarning)
                try:
                    sol = spla.spsolve(A_j, rhs)
                    if np.all(np.isfinite(sol)):
                        return sol
                except Exception as exc:  # MatrixRankWarning, RuntimeWarning, etc.
                    last_err = exc
        else:
            diag = A_j.diagonal().copy()
            diag[diag == 0] = 1.0
            M_inv = spla.LinearOperator((K, K), matvec=lambda v, d=diag: v / d)
            sol = np.empty_like(rhs)
            ok = True
            for t in range(rhs.shape[1]):
                col, info = spla.cg(A_j, rhs[:, t], M=M_inv, rtol=1e-8, atol=0.0, maxiter=2000)
                if info != 0 or not np.all(np.isfinite(col)):
                    ok = False
                    last_err = RuntimeError(f"cg no convergió (info={info}) en el target {t}")
                    break
                sol[:, t] = col
            if ok:
                return sol
        jitter = 1e-10 if jitter == 0.0 else jitter * 100.0

    raise np.linalg.LinAlgError(
        f"No se pudo resolver el sistema penalizado (K={K}, λ={lam}). Último error: {last_err}"
    )


def _apply_grid_kron(design_mats, coef):
    """Aplica matrices de diseño por dimensión a un tensor de coeficientes.

    Parameters
    ----------
    design_mats : lista de D matrices dispersas (n_d, K_d)
    coef : ndarray (K_1, ..., K_D, T)

    Returns
    -------
    ndarray (n_1, ..., n_D, T)
    """
    V = np.asarray(coef, dtype=float)
    D = len(design_mats)
    for d in range(D):
        A = design_mats[d]
        Kd = A.shape[1]
        Vd = np.moveaxis(V, d, 0).reshape(Kd, -1)
        Vd = A @ Vd
        rest = V.shape[:d] + V.shape[d + 1:]
        V = Vd.reshape((A.shape[0],) + tuple(rest))
        V = np.moveaxis(V, 0, d)
    return V


# ---------------------------------------------------------------------------
# Base B-spline tensorial con dominio robusto
# ---------------------------------------------------------------------------

class PSplineTensorBasis:
    """Base B-spline tensorial sobre un dominio rectangular robusto.

    Cada coordenada se normaliza internamente a [0, 1] antes de evaluar
    los splines, de modo que los nudos son uniformes e independientes del
    escalado físico de cada dimensión latente.
    """

    def __init__(self, dim, n_basis=8, degree=3, penalty_order=2):
        self.dim = int(dim)
        self.degree = int(degree)
        self.penalty_order = int(penalty_order)

        if np.isscalar(n_basis):
            n_basis = [int(n_basis)] * self.dim
        if len(n_basis) != self.dim:
            raise ValueError(f"n_basis debe ser escalar o tener {self.dim} entradas")
        self.n_basis = [int(k) for k in n_basis]
        for k in self.n_basis:
            if k < self.degree + 1:
                raise ValueError(
                    f"n_basis={k} < degree+1={self.degree + 1}: aumenta n_basis o reduce degree"
                )

        self.knots = []
        for k in self.n_basis:
            p = self.degree
            interior = np.linspace(0.0, 1.0, k - p + 1)[1:-1]
            t = np.concatenate([np.zeros(p + 1), interior, np.ones(p + 1)])
            self.knots.append(t)

        self.n_coeffs = int(np.prod(self.n_basis))
        self._domain_lo = None
        self._domain_hi = None

    # -- dominio -----------------------------------------------------------
    def set_domain(self, lo, hi):
        lo = np.asarray(lo, dtype=float).ravel()
        hi = np.asarray(hi, dtype=float).ravel()
        if lo.shape != (self.dim,) or hi.shape != (self.dim,):
            raise ValueError("lo/hi deben tener una entrada por dimensión")
        if np.any(hi <= lo):
            raise ValueError("Dominio inválido: se requiere hi > lo en cada dimensión")
        self._domain_lo = lo
        self._domain_hi = hi
        return self

    @property
    def domain(self):
        if self._domain_lo is None:
            raise RuntimeError("Dominio no definido: llama primero a set_domain")
        return self._domain_lo.copy(), self._domain_hi.copy()

    def normalize(self, X, clip=True):
        lo, hi = self.domain
        Z = (np.asarray(X, dtype=float) - lo) / (hi - lo)
        if clip:
            Z = np.clip(Z, 0.0, _EPS_RIGHT)
        return Z

    # -- matrices de diseño ------------------------------------------------
    def design_matrix_1d(self, z_d, d):
        """Matriz de diseño (n, K_d) de la dimensión d sobre puntos ya
        normalizados a [0, 1]."""
        return sparse.csr_matrix(
            BSpline.design_matrix(
                np.asarray(z_d, dtype=float), self.knots[d], self.degree,
                extrapolate=False,
            )
        )

    def design_matrix(self, X, clip=True):
        """Matriz de diseño tensorial (n, prod(K_d)) para puntos físicos X."""
        Z = self.normalize(X, clip=clip)
        B = None
        for d in range(self.dim):
            A_d = self.design_matrix_1d(Z[:, d], d)
            B = A_d if B is None else _row_kron_csr(B, A_d)
        return B

    def grid_design_matrices(self, edges):
        """Matrices de diseño por dimensión sobre centros de grilla físicos."""
        lo, hi = self.domain
        mats = []
        for d, e in enumerate(edges):
            z = (np.asarray(e, dtype=float) - lo[d]) / (hi[d] - lo[d])
            z = np.clip(z, 0.0, _EPS_RIGHT)
            mats.append(self.design_matrix_1d(z, d))
        return mats

    # -- penalización ------------------------------------------------------
    def penalty_matrix(self):
        """Penalización tensorial P = Σ_d I ⊗ ... ⊗ P_d ⊗ ... ⊗ I (dispersa)."""
        P_total = None
        for d in range(self.dim):
            P_d = _difference_penalty(self.n_basis[d], self.penalty_order)
            term = None
            for d2 in range(self.dim):
                M = P_d if d2 == d else sparse.identity(self.n_basis[d2], format='csr')
                term = M if term is None else sparse.kron(term, M, format='csr')
            P_total = term if P_total is None else P_total + term
        return P_total.tocsr()

    def grid_edges(self, bins, margin=0.0):
        """Centros de celda uniformes sobre el dominio robusto, en el formato
        de ``edges`` del pipeline (lista de D arrays de centros).

        ``margin`` es una fracción del rango del dominio añadida por lado;
        ver ``PSplineConfig.grid_margin``.
        """
        lo, hi = self.domain
        span = hi - lo
        lo = lo - margin * span
        hi = hi + margin * span
        if np.isscalar(bins):
            bins = [int(bins)] * self.dim
        if len(bins) != self.dim:
            raise ValueError(f"bins debe ser escalar o tener {self.dim} entradas")
        edges = []
        for d in range(self.dim):
            n_d = int(bins[d])
            h = (hi[d] - lo[d]) / n_d
            edges.append(lo[d] + h * (np.arange(n_d) + 0.5))
        return edges


# ---------------------------------------------------------------------------
# Estimador KM por P-splines
# ---------------------------------------------------------------------------

class PSplineKMEstimator:
    """Estimador global P-splines de deriva y difusión.

    Uso típico::

        est = PSplineKMEstimator(PSplineConfig(n_basis=8, degree=3))
        est.fit(data, dt)
        edges = est.grid_edges(bins=[50, 50])
        drift, diffusion = est.evaluate_grid(edges)
        info = est.diagnostics()

    Atributos tras ``fit``
    ----------------------
    basis_ : PSplineTensorBasis
    coef_drift_ : ndarray (K, D)
        Coeficientes de la deriva, organizados como tensor
        ``(K_1, ..., K_D, D)`` al evaluar en grilla.
    coef_moments_ : ndarray (K, D*(D+1)/2)
        Coeficientes de los segundos momentos, en el orden de pares
        ``pairs_ = [(0,0), (0,1), ..., (0,D-1), (1,1), ..., (D-1,D-1)]``.
    """

    def __init__(self, config=None, **overrides):
        self.config = PSplineConfig.from_any(config, **overrides)
        if self.config.fit_method != 'moments':
            raise NotImplementedError(
                f"fit_method='{self.config.fit_method}' no está implementado. "
                "En esta versión solo 'moments' (P-splines penalizados por "
                "mínimos cuadrados sobre los momentos de transición). La "
                "cuasi-verosimilitud de transición ('qmle') es una extensión "
                "futura documentada en el reporte técnico."
            )
        self.basis_ = None
        self.coef_drift_ = None
        self.coef_moments_ = None
        self.pairs_ = None
        self._fit_info = {}

    # ------------------------------------------------------------------
    def fit(self, data, dt=1.0):
        """Ajusta deriva y difusión desde una trayectoria (N, D)."""
        cfg = self.config
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("data debe ser un array (N, D) con N >= 2")
        if dt <= 0:
            raise ValueError("dt debe ser positivo")

        mask = np.all(np.isfinite(data), axis=1)
        n_bad = int((~mask).sum())
        if n_bad:
            warnings.warn(f"Se descartan {n_bad} muestras con valores no finitos", UserWarning)
            data = data[mask]

        N, D = data.shape
        n_basis = cfg.n_basis
        if np.isscalar(n_basis):
            n_basis_per_dim = [int(n_basis)] * D
        else:
            n_basis_per_dim = [int(k) for k in n_basis]
        K_total = int(np.prod(n_basis_per_dim))
        if K_total > cfg.max_tensor_bases:
            raise ValueError(
                f"prod(n_basis)={K_total} excede max_tensor_bases="
                f"{cfg.max_tensor_bases}. Reduce n_basis o implementa una "
                "representación estructurada (aditiva/sparse-grid)."
            )
        if N - 1 < K_total // 4:
            warnings.warn(
                f"Solo {N - 1} transiciones para {K_total} coeficientes "
                "tensoriales: el ajuste dependerá fuertemente de la "
                "penalización (riesgo de sobreajuste/estimación degenerada).",
                UserWarning,
            )

        # --- dominio robusto ------------------------------------------
        q = cfg.domain_quantile
        lo = np.quantile(data, q, axis=0)
        hi = np.quantile(data, 1.0 - q, axis=0)
        span = hi - lo
        span = np.where(span > 0, span, 1.0)
        lo = lo - cfg.domain_padding * span
        hi = hi + cfg.domain_padding * span

        basis = PSplineTensorBasis(D, n_basis=n_basis_per_dim,
                                   degree=cfg.degree,
                                   penalty_order=cfg.penalty_order)
        basis.set_domain(lo, hi)
        self.basis_ = basis
        self.pairs_ = [(i, j) for i in range(D) for j in range(i, D)]

        X0 = data[:-1]
        dX = np.diff(data, axis=0)
        y = dX / dt
        n_trans = X0.shape[0]

        # --- pasada 1: Gram y RHS de la deriva -------------------------
        G = None
        rhs_f = np.zeros((K_total, D))
        for s in range(0, n_trans, cfg.chunk_size):
            e = min(s + cfg.chunk_size, n_trans)
            B = basis.design_matrix(X0[s:e], clip=cfg.clip_to_domain)
            G_c = (B.T @ B).tocsr()
            G = G_c if G is None else G + G_c
            rhs_f += B.T @ y[s:e]
        G = G / n_trans
        rhs_f /= n_trans

        P = basis.penalty_matrix()
        self._P = P

        coef_f = _solve_penalized(G, P, cfg.lambda_drift, rhs_f,
                                  direct_threshold=cfg.direct_solver_max_basis)
        self.coef_drift_ = coef_f

        # --- pasada 2: residuos y segundos momentos --------------------
        T2 = len(self.pairs_)
        rhs_m = np.zeros((K_total, T2))
        ss_res = np.zeros(D)
        for s in range(0, n_trans, cfg.chunk_size):
            e = min(s + cfg.chunk_size, n_trans)
            B = basis.design_matrix(X0[s:e], clip=cfg.clip_to_domain)
            f_hat = B @ coef_f
            R = dX[s:e] - dt * f_hat
            ss_res += np.sum((y[s:e] - f_hat) ** 2, axis=0)
            S = np.empty((e - s, T2))
            for c, (i, j) in enumerate(self.pairs_):
                S[:, c] = R[:, i] * R[:, j] / (2.0 * dt)
            rhs_m += B.T @ S
        rhs_m /= n_trans

        coef_m = _solve_penalized(G, P, cfg.lambda_diffusion, rhs_m,
                                  direct_threshold=cfg.direct_solver_max_basis)
        self.coef_moments_ = coef_m

        # --- diagnósticos de ajuste ------------------------------------
        rmse = np.sqrt(ss_res / n_trans)
        var_y = np.var(y, axis=0)
        r2 = 1.0 - ss_res / np.maximum(n_trans * var_y, 1e-300)

        self._fit_info = {
            'method': 'p_splines',
            'fit_method': cfg.fit_method,
            'n_samples': int(N),
            'n_transitions': int(n_trans),
            'dim': int(D),
            'dt': float(dt),
            'n_basis': list(n_basis_per_dim),
            'n_coefficients': K_total,
            'degree': cfg.degree,
            'penalty_order': cfg.penalty_order,
            'lambda_drift': float(cfg.lambda_drift),
            'lambda_diffusion': float(cfg.lambda_diffusion),
            'grid_margin': float(cfg.grid_margin),
            'domain_lo': lo.tolist(),
            'domain_hi': hi.tolist(),
            'drift_rmse': rmse.tolist(),
            'drift_r2': r2.tolist(),
        }
        return self

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if self.coef_drift_ is None or self.coef_moments_ is None:
            raise RuntimeError("Estimador no ajustado: llama primero a fit(data, dt)")

    def _moments_to_spd(self, M_vals):
        """Reconstruye matrices D SPD a partir de campos de segundos momentos.

        M_vals : ndarray (..., T2) con el orden de pares de ``pairs_``.
        Retorna ndarray (..., D, D) con autovalores >= min_diffusion.
        """
        cfg = self.config
        D = self.basis_.dim
        shape = M_vals.shape[:-1]
        M = M_vals.reshape(-1, len(self.pairs_))
        n = M.shape[0]

        Dmat = np.zeros((n, D, D))
        var = np.zeros((n, D))
        for i in range(D):
            c = self.pairs_.index((i, i))
            var[:, i] = np.maximum(M[:, c], cfg.min_diffusion)
        for c, (i, j) in enumerate(self.pairs_):
            if i == j:
                Dmat[:, i, i] = var[:, i]
            else:
                denom = np.sqrt(var[:, i] * var[:, j])
                rho = np.clip(M[:, c] / denom,
                              -cfg.correlation_clip, cfg.correlation_clip)
                Dmat[:, i, j] = Dmat[:, j, i] = rho * denom

        # Salvaguarda espectral vectorizada (por si la correlación
        # reconstruida no fuera PD en D >= 3).
        w, V = np.linalg.eigh(Dmat)
        w = np.maximum(w, cfg.min_diffusion)
        Dmat = (V * w[:, None, :]) @ np.transpose(V, (0, 2, 1))
        return Dmat.reshape(shape + (D, D))

    # ------------------------------------------------------------------
    def predict(self, points):
        """Evalúa f(x) y D(x) en puntos arbitrarios (n, D).

        Returns
        -------
        f : ndarray (n, D)
        D_mat : ndarray (n, D, D), simétrica definida positiva.
        """
        self._check_fitted()
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != self.basis_.dim:
            raise ValueError(f"points debe tener {self.basis_.dim} columnas")

        cfg = self.config
        n = points.shape[0]
        f_out = np.empty((n, self.basis_.dim))
        m_out = np.empty((n, len(self.pairs_)))
        for s in range(0, n, cfg.chunk_size):
            e = min(s + cfg.chunk_size, n)
            B = self.basis_.design_matrix(points[s:e], clip=cfg.clip_to_domain)
            f_out[s:e] = B @ self.coef_drift_
            m_out[s:e] = B @ self.coef_moments_
        return f_out, self._moments_to_spd(m_out)

    # ------------------------------------------------------------------
    def evaluate_grid(self, edges):
        """Evalúa los campos sobre una grilla de centros (formato IgA).

        Parameters
        ----------
        edges : lista de D arrays con los centros de celda (coordenadas
            físicas), tal como los produce ``grid_edges`` o la ruta
            Nadaraya–Watson.

        Returns
        -------
        drift : ndarray (D, n1, ..., nD)
        diffusion : ndarray (D, D, n1, ..., nD), SPD por celda.
        """
        self._check_fitted()
        D = self.basis_.dim
        if len(edges) != D:
            raise ValueError(f"edges debe tener {D} entradas")

        mats = self.basis_.grid_design_matrices(edges)
        Ks = self.basis_.n_basis

        f_grid = _apply_grid_kron(
            mats, self.coef_drift_.reshape(tuple(Ks) + (D,))
        )  # (n1, ..., nD, D)
        m_grid = _apply_grid_kron(
            mats, self.coef_moments_.reshape(tuple(Ks) + (len(self.pairs_),))
        )  # (n1, ..., nD, T2)

        D_grid = self._moments_to_spd(m_grid)  # (n1, ..., nD, D, D)

        drift = np.moveaxis(f_grid, -1, 0)          # (D, n1, ..., nD)
        diffusion = np.moveaxis(D_grid, (-2, -1), (0, 1))  # (D, D, n1, ..., nD)
        return drift, diffusion

    # ------------------------------------------------------------------
    def grid_edges(self, bins):
        """Centros de celda uniformes sobre el dominio robusto ajustado,
        incluyendo el margen vacío ``config.grid_margin`` alrededor (para
        que la máscara de densidad de IgA no toque el borde de la grilla).
        """
        self._check_fitted()
        return self.basis_.grid_edges(bins, margin=self.config.grid_margin)

    # ------------------------------------------------------------------
    def diagnostics(self):
        """Diagnósticos del ajuste y de la matriz D estimada.

        Incluye tamaños muestrales, hiperparámetros, RMSE/R² de la deriva
        y estadísticas espectrales de D sobre una grilla de sondeo
        (autovalor mínimo y número de condición por celda).
        """
        self._check_fitted()
        info = dict(self._fit_info)
        D = self.basis_.dim
        n_probe = max(2, int(np.floor(4096.0 ** (1.0 / D))))
        edges = self.basis_.grid_edges([n_probe] * D)
        _, diffusion = self.evaluate_grid(edges)
        Dm = np.moveaxis(diffusion, (0, 1), (-2, -1)).reshape(-1, D, D)
        w = np.linalg.eigvalsh(Dm)
        lam_min = w[:, 0]
        lam_max = w[:, -1]
        cond = lam_max / np.maximum(lam_min, 1e-300)
        info.update({
            'diffusion_min_eigenvalue_min': float(lam_min.min()),
            'diffusion_min_eigenvalue_median': float(np.median(lam_min)),
            'diffusion_condition_median': float(np.median(cond)),
            'diffusion_condition_max': float(cond.max()),
            'probe_grid_per_dim': int(n_probe),
        })
        return info


# ---------------------------------------------------------------------------
# Función de conveniencia compatible con el pipeline
# ---------------------------------------------------------------------------

def estimate_drift_diffusion_psplines(data, bins, dt=1.0, config=None,
                                      **config_overrides):
    """Estima f y D con P-splines y las evalúa sobre la grilla del pipeline.

    Parameters
    ----------
    data : ndarray (N, D)
        Trayectoria (o superposición de trayectorias apiladas en tiempo).
    bins : int o Sequence[int]
        Resolución de la grilla de salida (centros uniformes sobre el
        dominio robusto). No interviene en el ajuste, que es global.
    dt : float
        Paso temporal entre observaciones consecutivas.
    config : None, dict o PSplineConfig
        Hiperparámetros del estimador.
    **config_overrides
        Overrides puntuales, p.ej. ``lambda_drift=1e-2``.

    Returns
    -------
    drift : ndarray (D, n1, ..., nD)
    diffusion : ndarray (D, D, n1, ..., nD)
    edges : lista de D arrays (centros de celda)
    info : dict
        Diagnósticos del estimador (ver ``PSplineKMEstimator.diagnostics``).
    """
    estimator = PSplineKMEstimator(config, **config_overrides)
    estimator.fit(data, dt=dt)
    edges = estimator.grid_edges(bins)
    drift, diffusion = estimator.evaluate_grid(edges)
    info = estimator.diagnostics()
    return drift, diffusion, edges, info

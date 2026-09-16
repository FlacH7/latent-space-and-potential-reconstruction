"""
data_analysis_tools.py
======================
Utilidades de pre-procesamiento y análisis para series temporales
N-dimensionales.

Lógica simple: para cada dimensión se detectan outliers, se unen los
índices en un set (sin repetidos), y se eliminan esas filas del array
original.  El array retornado conserva su ndim y todas las dimensiones
excepto la primera (samples), que se reduce.

Nuevas funciones:
  - augment_dimensions : replica dimensiones con ruido gaussiano para
                         llegar a un D_target.
  - augment_samples    : aumenta la longitud temporal via block bootstrap
                         o simulación sintética con KM estimado.

Funciones principales:
  - outliers_cleaning : elimina filas donde ALGUNA dimensión tiene outlier.
  - detect_outliers   : retorna máscara booleana 1D de filas válidas.
  - winsorize_data    : recorta (clip) outliers por dimensión.
  - summarize_data    : estadísticas descriptivas por dimensión.
  - augment_dimensions: aumenta el número de features (columnas).
  - augment_samples   : aumenta el número de muestras (filas).

Uso típico::

    from data_analysis_tools import (
        outliers_cleaning, augment_dimensions, augment_samples
    )
    data_clean = outliers_cleaning(data, method='mad', threshold=3.5)
    data_50d   = augment_dimensions(data_clean, D_target=50, noise_std=0.01)
    data_long  = augment_samples(data_50d, n_target=16_000_000, method='block_bootstrap')
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


# ============================================================================
# 0. HELPERS INTERNOS
# ============================================================================

def _estimate_block_size(data: np.ndarray, threshold: float = 0.05) -> int:
    """
    Estima un tamaño de bloque útil para block bootstrap a partir de la
    autocorrelación.  Devuelve el primer lag donde la autocorrelación
    promedio (sobre todas las dimensiones) cae por debajo de ``threshold``.
    """
    n_samples, n_dim = data.shape
    max_lag = max(1, n_samples // 4)
    best_lags = []
    for d in range(n_dim):
        x = data[:, d] - np.mean(data[:, d])
        # autocorrelación normalizada (unilateral)
        autocorr = np.correlate(x, x, mode="full")[n_samples - 1:]
        if autocorr[0] == 0:
            best_lags.append(1)
            continue
        autocorr = autocorr / autocorr[0]
        below = np.where(np.abs(autocorr[1:max_lag + 1]) < threshold)[0]
        if len(below) > 0:
            best_lags.append(below[0] + 1)
        else:
            best_lags.append(max_lag)
    return max(1, int(np.median(best_lags)))


def _block_bootstrap_resample(
    data: np.ndarray, n_target: int, block_size: int, seed: int | None = None
) -> np.ndarray:
    """
    Block bootstrap con reemplazo.  Concatena bloques de longitud
    ``block_size`` muestreados aleatoriamente hasta alcanzar ``n_target``.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_dim = data.shape
    n_blocks_needed = int(np.ceil(n_target / block_size))
    indices = []
    max_start = n_samples - block_size
    if max_start < 0:
        # Si la serie es más corta que el bloque, repetir la serie entera
        repeats = int(np.ceil(n_target / n_samples))
        indices = np.tile(np.arange(n_samples), repeats)[:n_target]
        return data[indices]

    for _ in range(n_blocks_needed):
        start = rng.integers(0, max_start + 1)
        indices.extend(range(start, start + block_size))
    indices = np.array(indices[:n_target])
    return data[indices]


def _synthetic_km_resample(
    data: np.ndarray, n_target: int, dt: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """
    Estima drift/diffusion via kramersmoyal (o km_tools_v2 si existe) y
    simula una trayectoria más larga con Euler-Maruyama.

    Soporta D=1 y D=2.  Para D>2 lanza NotImplementedError.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_dim = data.shape
    if n_target <= n_samples:
        return data[:n_target].copy()

    # -----------------------------------------------------------------
    # 1. Estimar KM
    # -----------------------------------------------------------------
    try:
        from src.potential_reconstruction.km_tools_v2 import extract_km_coefficients
        bins = [max(20, int(n_samples ** (1.0 / n_dim) / 2))] * n_dim
        drift, diffusion, edges = extract_km_coefficients(
            data, bins=bins, p=2, bw=None, kernel="epanechnikov", dt=dt
        )
    except ImportError:
        try:
            from kramersmoyal import km
            from kramersmoyal.kernels import epanechnikov
            bins = [max(20, int(n_samples ** (1.0 / n_dim) / 2))] * n_dim
            kmc, edges, bw_used, powers = km(
                data, bins=bins, powers=2, kernel=epanechnikov, bw=None,
                center_edges=True, full=True,
            )
            # Reconstruir drift y diffusion manualmente
            D = n_dim
            sum_powers = np.sum(powers, axis=1)
            first_order_idx = np.where(sum_powers == 1)[0]
            comp_idx = [np.argmax(powers[i]) for i in first_order_idx]
            order = np.argsort(comp_idx)
            first_order_idx = first_order_idx[order]
            drift = np.array([kmc[idx] / dt for idx in first_order_idx])
            second_order_idx = np.where(sum_powers == 2)[0]
            diff_numeric = np.full((D, D) + drift.shape[1:], np.nan, dtype=float)
            for idx in second_order_idx:
                comb = powers[idx]
                nonzero = np.where(comb > 0)[0]
                if len(nonzero) == 1:
                    i = nonzero[0]
                    diff_numeric[i, i] = kmc[idx] / dt
                elif len(nonzero) == 2:
                    i, j = nonzero[0], nonzero[1]
                    diff_numeric[i, j] = kmc[idx] / dt
                    diff_numeric[j, i] = kmc[idx] / dt
            diffusion = diff_numeric
        except ImportError as exc:
            raise ImportError(
                "method='synthetic_km' requiere 'km_tools_v2' o 'kramersmoyal'. "
                "Instala uno de ellos o usa method='block_bootstrap'."
            ) from exc

    # -----------------------------------------------------------------
    # 2. Construir interpoladores
    # -----------------------------------------------------------------
    try:
        from scipy.interpolate import RegularGridInterpolator
    except ImportError as exc:
        raise ImportError(
            "method='synthetic_km' requiere scipy para interpolar."
        ) from exc

    if n_dim == 1:
        # Drift interpolator
        drift_interp = RegularGridInterpolator(
            (edges[0],), drift[0], bounds_error=False, fill_value=0.0
        )
        # Diffusion scalar interpolator
        diff_interp = RegularGridInterpolator(
            (edges[0],), diffusion[0, 0], bounds_error=False, fill_value=0.0
        )

        x0 = data[-1, 0] if n_samples > 0 else 0.0
        X_new = np.zeros(n_target)
        X_new[0] = x0
        for t in range(1, n_target):
            x = X_new[t - 1]
            d = drift_interp([[x]])[0] * dt
            sigma = np.sqrt(2.0 * max(diff_interp([[x]])[0], 0.0) * dt)
            X_new[t] = x + d + sigma * rng.standard_normal()
        return X_new.reshape(-1, 1)

    elif n_dim == 2:
        drift_interp = [
            RegularGridInterpolator(
                (edges[0], edges[1]), drift[i], bounds_error=False, fill_value=0.0
            )
            for i in range(2)
        ]
        # Difusión como matriz 2x2 de interpoladores
        diff_interp = [[None, None], [None, None]]
        for i in range(2):
            for j in range(2):
                if not np.all(np.isnan(diffusion[i, j])):
                    diff_interp[i][j] = RegularGridInterpolator(
                        (edges[0], edges[1]), diffusion[i, j],
                        bounds_error=False, fill_value=0.0
                    )

        x0 = data[-1] if n_samples > 0 else np.zeros(2)
        X_new = np.zeros((n_target, 2))
        X_new[0] = x0
        for t in range(1, n_target):
            x = X_new[t - 1]
            d = np.array([drift_interp[i]([x])[0] for i in range(2)]) * dt
            D_mat = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    if diff_interp[i][j] is not None:
                        D_mat[i, j] = diff_interp[i][j]([x])[0]
            # Asegurar semidefinida positiva
            D_sym = (D_mat + D_mat.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(D_sym)
            eigvals = np.maximum(eigvals, 0.0)
            sqrt_D = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            dW = rng.standard_normal(2)
            X_new[t] = x + d + sqrt_D @ dW * np.sqrt(dt)
        return X_new

    else:
        raise NotImplementedError(
            f"method='synthetic_km' solo soporta D=1 o D=2 (recibido D={n_dim}). "
            f"Usa method='block_bootstrap' para D>2."
        )


# ============================================================================
# 1. DETECCIÓN DE OUTLIERS (máscara por fila)
# ============================================================================

def detect_outliers(
    data: np.ndarray,
    method: Literal["iqr", "zscore", "mad", "percentile", "mahalanobis"] = "iqr",
    threshold: float = 1.5,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    mahalanobis_alpha: float = 0.99,
) -> np.ndarray:
    """
    Detecta outliers en cada dimensión y retorna una máscara 1D
    donde ``True`` significa "fila válida" (no outlier en ninguna dim).

    Parameters
    ----------
    data : ndarray, shape (n_samples, ...)  (última dim = features)
        Puede ser 1D (n_samples,), 2D (n_samples, n_dims), o N-D.
    method : {'iqr', 'zscore', 'mad', 'percentile', 'mahalanobis'}
    threshold : float
        Factor IQR (default 1.5), sigmas zscore/mad, o factor Mahalanobis.
    lower_pct, upper_pct : float
        Solo para ``method='percentile'``.
    mahalanobis_alpha : float
        Nivel de confianza χ² para Mahalanobis.

    Returns
    -------
    mask_keep : ndarray (bool), shape (n_samples,)
    """
    if data.ndim == 0:
        raise ValueError("data no puede ser un escalar (0-D).")

    n_samples = data.shape[0]
    if n_samples == 0:
        return np.ones(0, dtype=bool)

    # Ajustar defaults de threshold según método
    if method == "zscore" and threshold == 1.5:
        threshold = 3.0
    elif method == "mad" and threshold == 1.5:
        threshold = 3.5

    features = data.reshape(n_samples, -1)
    n_features = features.shape[1]

    # -----------------------------------------------------------------
    # IQR
    # -----------------------------------------------------------------
    if method == "iqr":
        q1 = np.percentile(features, 25, axis=0)
        q3 = np.percentile(features, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask_per_dim = (features >= lower) & (features <= upper)
        mask_keep = np.all(mask_per_dim, axis=1)

    # -----------------------------------------------------------------
    # Z-score
    # -----------------------------------------------------------------
    elif method == "zscore":
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1e-12
        z = np.abs((features - mean) / std)
        mask_keep = np.all(z < threshold, axis=1)

    # -----------------------------------------------------------------
    # MAD (robusto)
    # -----------------------------------------------------------------
    elif method == "mad":
        median = np.median(features, axis=0)
        mad = np.median(np.abs(features - median), axis=0)
        mad[mad == 0] = 1e-12
        modified_z = 0.6745 * (features - median) / mad
        mask_keep = np.all(np.abs(modified_z) < threshold, axis=1)

    # -----------------------------------------------------------------
    # Percentiles
    # -----------------------------------------------------------------
    elif method == "percentile":
        lower = np.percentile(features, lower_pct, axis=0)
        upper = np.percentile(features, upper_pct, axis=0)
        mask_per_dim = (features >= lower) & (features <= upper)
        mask_keep = np.all(mask_per_dim, axis=1)

    # -----------------------------------------------------------------
    # Mahalanobis (multivariante)
    # -----------------------------------------------------------------
    elif method == "mahalanobis":
        try:
            from scipy.stats import chi2
        except ImportError as exc:
            raise ImportError(
                "method='mahalanobis' requiere scipy. "
                "Instálalo con: pip install scipy"
            ) from exc

        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        cov_reg = cov + 1e-6 * np.eye(n_features)
        try:
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_reg)

        diff = features - mean
        dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        cutoff = np.sqrt(chi2.ppf(mahalanobis_alpha, df=n_features))
        mask_keep = dist < threshold * cutoff

    else:
        raise ValueError(f"method='{method}' no reconocido.")

    return mask_keep


# ============================================================================
# 2. LIMPIEZA (eliminación de filas)
# ============================================================================

def outliers_cleaning(
    data: np.ndarray,
    method: Literal["iqr", "zscore", "mad", "percentile", "mahalanobis"] = "iqr",
    threshold: float = 1.5,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    return_mask: bool = False,
    verbose: bool = True,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Elimina filas (primer eje) donde ALGUNA dimensión/feature contiene
    un outlier.  El array retornado conserva su ndim original.

    Parameters
    ----------
    data : ndarray
        1D, 2D o N-D.  La primera dimensión es ``n_samples``.
    method, threshold, lower_pct, upper_pct :
        Ver ``detect_outliers``.
    return_mask : bool
        Si ``True``, retorna ``(data_clean, mask_keep)``.
    verbose : bool
        Imprime resumen.

    Returns
    -------
    data_clean : ndarray
        Mismo ndim que ``data``, primera dimensión reducida.
    mask_keep : ndarray (bool)  [solo si return_mask=True]
    """
    if data.ndim == 0:
        raise ValueError("data no puede ser un escalar (0-D).")

    n_before = data.shape[0]

    mask_keep = detect_outliers(
        data,
        method=method,
        threshold=threshold,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
    )

    data_clean = data[mask_keep]
    n_after = data_clean.shape[0]
    n_removed = n_before - n_after
    pct_removed = 100.0 * n_removed / n_before if n_before > 0 else 0.0

    if verbose:
        print(f"[outliers_cleaning] method={method}, threshold={threshold}")
        print(f"  Input shape    : {data.shape}")
        print(f"  Samples before : {n_before}")
        print(f"  Samples after  : {n_after}")
        print(f"  Removed        : {n_removed}  ({pct_removed:.2f}%)")
        if n_removed > 0 and data.size > 0:
            print(f"  Range before   : [{np.min(data):.4f}, {np.max(data):.4f}]")
            print(f"  Range after    : [{np.min(data_clean):.4f}, {np.max(data_clean):.4f}]")

    if return_mask:
        return data_clean, mask_keep
    return data_clean


# ============================================================================
# 3. WINSORIZACIÓN (clip por dimensión)
# ============================================================================

def winsorize_data(
    data: np.ndarray,
    method: Literal["iqr", "percentile"] = "iqr",
    threshold: float = 1.5,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Recorta (clip) los outliers en cada dimensión/feature a los límites
    calculados, preservando el shape exacto de la entrada.

    Parameters
    ----------
    data : ndarray
        1D, 2D o N-D.  La primera dimensión es ``n_samples``.

    Returns
    -------
    data_winsorized : ndarray, mismo shape que ``data``.
    """
    if data.ndim == 0:
        raise ValueError("data no puede ser un escalar (0-D).")

    n_samples = data.shape[0]
    features = data.reshape(n_samples, -1)
    n_features = features.shape[1]
    data_out = features.copy()

    if method == "iqr":
        q1 = np.percentile(features, 25, axis=0)
        q3 = np.percentile(features, 75, axis=0)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    elif method == "percentile":
        lower = np.percentile(features, lower_pct, axis=0)
        upper = np.percentile(features, upper_pct, axis=0)
    else:
        raise ValueError(f"method='{method}' no soportado para winsorize.")

    for d in range(n_features):
        data_out[:, d] = np.clip(data_out[:, d], lower[d], upper[d])

    data_out = data_out.reshape(data.shape)

    if verbose:
        n_clipped = np.sum(data_out != features)
        print(f"[winsorize_data] Valores recortados: {n_clipped}")

    return data_out


# ============================================================================
# 4. AUMENTAR DIMENSIONES (n_features -> D_target)
# ============================================================================

# def augment_dimensions(
#     data: np.ndarray,
#     D_target: int,
#     noise_std: float = 0.01,
#     seed: int | None = None,
# ) -> np.ndarray:
#     """
#     Aumenta el número de dimensiones (features) de ``data`` hasta
#     ``D_target`` generando nuevas columnas como perturbaciones
#     gaussianas de las dimensiones originales.

#     Cada nueva dimensión ``k`` se construye como::

#         new_k = base_j + N(0, noise_std * std(base_j))

#     donde ``base_j`` es una de las dimensiones originales, asignadas
#     cíclicamente.

#     Parameters
#     ----------
#     data : ndarray, shape (n_samples, n_dim)
#         Serie temporal.  Debe ser al menos 2D (la última dim es features).
#     D_target : int
#         Número de dimensiones deseado.  Si ``D_target <= n_dim``,
#         retorna ``data[:, :D_target]``.
#     noise_std : float
#         Desviación estándar del ruido relativo a la std de la dimensión
#         base.  Default ``0.01`` (1 % de la std).
#     seed : int, optional
#         Semilla para reproducibilidad.

#     Returns
#     -------
#     data_aug : ndarray, shape (n_samples, D_target)

#     Examples
#     --------
#     >>> data = np.random.randn(1_600_000, 2)
#     >>> data_50 = augment_dimensions(data, D_target=50, noise_std=0.01)
#     >>> data_50.shape
#     (1600000, 50)
#     """
#     rng = np.random.default_rng(seed)
#     if data.ndim < 2:
#         raise ValueError(
#             f"augment_dimensions requiere data.ndim >= 2 (n_samples, n_dims). "
#             f"Recibido: {data.ndim}D"
#         )

#     n_samples, n_dim = data.shape[0], data.shape[-1]
#     if D_target <= n_dim:
#         return data[..., :D_target].copy()

#     n_new = D_target - n_dim
#     new_dims = []
#     for k in range(n_new):
#         base_j = k % n_dim
#         std_base = np.std(data[..., base_j])
#         noise = rng.normal(0.0, noise_std * std_base, n_samples)
#         new_col = data[..., base_j] + noise
#         new_dims.append(new_col.reshape(n_samples, 1))

#     data_aug = np.concatenate([data.reshape(n_samples, n_dim)] + new_dims, axis=1)
#     return data_aug

def augment_dimensions(
    data: np.ndarray,
    D_target: int,
    noise_std: float = 0.01,
    seed: int | None = None,
    mode: str = "mix",
) -> np.ndarray:
    """
    Aumenta el número de dimensiones (features) de ``data`` hasta ``D_target``.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_dim)
        Serie temporal. Debe ser al menos 2D (la última dim es features).
    D_target : int
        Número de dimensiones deseado.  Si ``D_target <= n_dim``,
        retorna ``data[:, :D_target]``.
    noise_std : float
        - Si ``mode='noise'``: desviación estándar del ruido relativo a la std de la dimensión base.
        - Si ``mode='mix'``: desviación estándar del ruido independiente añadido a la mezcla
          (en unidades absolutas; se multiplica por la desviación estándar global de los datos).
    seed : int, optional
        Semilla para reproducibilidad.
    mode : {'noise', 'mix'}
        - ``'noise'`` (default): genera nuevas columnas como copias ruidosas de las originales.
        - ``'mix'``: genera nuevas columnas como combinaciones lineales de las originales
          con pesos aleatorios N(0,1), más un pequeño ruido independiente.

    Returns
    -------
    data_aug : ndarray, shape (n_samples, D_target)

    Examples
    --------
    >>> data = np.random.randn(1000, 2)
    >>> data_aug = augment_dimensions(data, D_target=5, mode='mix', seed=42)
    >>> data_aug.shape
    (1000, 5)
    """
    rng = np.random.default_rng(seed)
    if data.ndim < 2:
        raise ValueError(
            f"augment_dimensions requiere data.ndim >= 2 (n_samples, n_dims). "
            f"Recibido: {data.ndim}D"
        )

    n_samples, n_dim = data.shape[0], data.shape[-1]
    if D_target <= n_dim:
        return data[..., :D_target].copy()

    if mode == "noise":
        # --- Modo original: copias ruidosas ---
        n_new = D_target - n_dim
        new_dims = []
        for k in range(n_new):
            base_j = k % n_dim
            std_base = np.std(data[..., base_j])
            noise = rng.normal(0.0, noise_std * std_base, n_samples)
            new_col = data[..., base_j] + noise
            new_dims.append(new_col.reshape(n_samples, 1))
        data_aug = np.concatenate([data.reshape(n_samples, n_dim)] + new_dims, axis=1)

    elif mode == "mix":
        # --- Modo mezcla lineal con pesos gaussianos ---
        # Matriz de pesos (D_target, n_dim) con N(0,1)
        weights = rng.normal(0, 1, size=(D_target, n_dim))
        # Generar mezclas: data_aug = data @ weights.T  (n_samples, D_target)
        data_aug = data @ weights.T

        # Añadir ruido independiente (opcional) para simular ruido de medición
        if noise_std > 0:
            # Escala del ruido: usamos la desviación global de los datos originales
            global_std = np.std(data)
            if global_std == 0:
                global_std = 1.0
            noise = rng.normal(0, noise_std * global_std, size=data_aug.shape)
            data_aug += noise

    else:
        raise ValueError(f"mode='{mode}' no reconocido. Use 'noise' o 'mix'.")

    return data_aug


# ============================================================================
# 5. AUMENTAR MUESTRAS (n_samples -> n_target)
# ============================================================================

def augment_samples(
    data: np.ndarray,
    n_target: int,
    method: Literal["block_bootstrap", "synthetic_km"] = "block_bootstrap",
    block_size: int | None = None,
    dt: float = 1.0,
    seed: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Aumenta la longitud temporal de ``data`` hasta ``n_target`` muestras.

    * ``block_bootstrap`` (default): divide la serie en bloques y los
      concatena muestreados con reemplazo.  Preserva la estructura
      local (autocorrelación) y por tanto las propiedades de KM.
    * ``synthetic_km``: estima drift/diffusion y simula una trayectoria
      más larga con Euler-Maruyama.  Más fiel a la dinámica subyacente
      pero solo soporta D=1 o D=2 y requiere ``km_tools_v2`` o
      ``kramersmoyal`` + ``scipy``.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_dim)
    n_target : int
        Número de muestras deseado.
    method : {'block_bootstrap', 'synthetic_km'}
    block_size : int, optional
        Tamaño de bloque para bootstrap.  Si ``None``, se estima
        automáticamente desde la autocorrelación.
    dt : float
        Paso temporal (solo usado por ``synthetic_km``).
    seed : int, optional
    verbose : bool

    Returns
    -------
    data_aug : ndarray, shape (n_target, n_dim)

    Examples
    --------
    >>> data = np.random.randn(1_600_000, 2)
    >>> data_long = augment_samples(data, n_target=16_000_000, method='block_bootstrap')
    >>> data_long.shape
    (16000000, 2)
    """
    if data.ndim < 2:
        raise ValueError(
            f"augment_samples requiere data.ndim >= 2. Recibido: {data.ndim}D"
        )

    n_samples = data.shape[0]
    if n_target <= n_samples:
        if verbose:
            print(f"[augment_samples] n_target ({n_target}) <= n_samples ({n_samples}). "
                  f"Retornando submuestra.")
        return data[:n_target].copy()

    if method == "block_bootstrap":
        if block_size is None:
            block_size = _estimate_block_size(data)
        if verbose:
            print(f"[augment_samples] block_bootstrap  |  block_size={block_size}")
        data_aug = _block_bootstrap_resample(data, n_target, block_size, seed)

    elif method == "synthetic_km":
        if verbose:
            print(f"[augment_samples] synthetic_km  |  dt={dt}")
        data_aug = _synthetic_km_resample(data, n_target, dt, seed)

    else:
        raise ValueError(f"method='{method}' no reconocido.")

    if verbose:
        print(f"  Input shape  : {data.shape}")
        print(f"  Output shape : {data_aug.shape}")

    return data_aug


# ============================================================================
# 6. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================

def summarize_data(data: np.ndarray, name: str = "data") -> None:
    """
    Imprime estadísticas descriptivas por feature (última dimensión).
    Soporta 1D, 2D y N-D.
    """
    if data.ndim == 0:
        raise ValueError("data no puede ser un escalar (0-D).")

    n_samples = data.shape[0]
    features = data.reshape(n_samples, -1)

    print(f"\n{'='*60}")
    print(f"  Summary: {name}")
    print(f"{'='*60}")
    print(f"  Shape      : {data.shape}")
    print(f"  Samples    : {n_samples}")
    print(f"  Features   : {features.shape[1]}")
    print(f"  Mean       : {np.mean(features, axis=0)}")
    print(f"  Std        : {np.std(features, axis=0)}")
    print(f"  Min        : {np.min(features, axis=0)}")
    print(f"  Max        : {np.max(features, axis=0)}")
    print(f"  Median     : {np.median(features, axis=0)}")
    q1 = np.percentile(features, 25, axis=0)
    q3 = np.percentile(features, 75, axis=0)
    print(f"  Q1         : {q1}")
    print(f"  Q3         : {q3}")
    print(f"  IQR        : {q3 - q1}")
    print(f"{'='*60}\n")


# ============================================================================
# 7. EJECUCIÓN COMO SCRIPT (demo)
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Demo 1D
    print("=" * 60)
    print("DEMO 1D")
    print("=" * 60)
    d1 = np.random.normal(0, 1, 1000)
    d1[[100, 500, 900]] += [15, -12, 18]
    summarize_data(d1, "raw_1d")
    clean_1d = outliers_cleaning(d1, method="iqr", threshold=1.5)
    summarize_data(clean_1d, "clean_1d")
    print(f"  Output shape: {clean_1d.shape}  (ndim={clean_1d.ndim})")

    # Demo 2D
    print("=" * 60)
    print("DEMO 2D")
    print("=" * 60)
    d1 = np.random.normal(0, 1, 1000)
    d2 = np.random.normal(0, 0.5, 1000)
    d1[[100, 500, 900]] += [15, -12, 18]
    d2[[100, 500, 900]] += [8, -20, 10]
    data_2d = np.column_stack([d1, d2])
    summarize_data(data_2d, "raw_2d")
    clean_2d = outliers_cleaning(data_2d, method="mad", threshold=3.5)
    summarize_data(clean_2d, "clean_2d")
    print(f"  Output shape: {clean_2d.shape}  (ndim={clean_2d.ndim})")

    # Demo 3D (batch, time, channels)
    print("=" * 60)
    print("DEMO 3D")
    print("=" * 60)
    data_3d = np.random.normal(0, 1, (10, 100, 3))
    data_3d[5, 50, 1] = 50   # outlier en dim 1
    data_3d[2, 30, 2] = -30  # outlier en dim 2
    summarize_data(data_3d, "raw_3d")
    clean_3d = outliers_cleaning(data_3d, method="iqr", threshold=1.5)
    summarize_data(clean_3d, "clean_3d")
    print(f"  Output shape: {clean_3d.shape}  (ndim={clean_3d.ndim})")

    # Demo augment_dimensions
    print("=" * 60)
    print("DEMO augment_dimensions")
    print("=" * 60)
    data_small = np.random.randn(1000, 2)
    data_50 = augment_dimensions(data_small, D_target=50, noise_std=0.01, seed=42)
    print(f"  Input shape : {data_small.shape}")
    print(f"  Output shape: {data_50.shape}")
    # Verificar que las originales se preservan
    print(f"  Diff orig[0] vs aug[0] (max abs): {np.max(np.abs(data_small[:,0] - data_50[:,0])):.2e}")
    print(f"  Diff orig[1] vs aug[1] (max abs): {np.max(np.abs(data_small[:,1] - data_50[:,1])):.2e}")
    # Verificar que las nuevas son cercanas a su base
    print(f"  Diff aug[2] vs aug[0] (std)    : {np.std(data_50[:,2] - data_50[:,0]):.4f}")

    # Demo augment_samples (block_bootstrap)
    print("=" * 60)
    print("DEMO augment_samples (block_bootstrap)")
    print("=" * 60)
    data_short = np.random.randn(1000, 2)
    data_long = augment_samples(data_short, n_target=10000, method="block_bootstrap", seed=42)
    print(f"  Input shape : {data_short.shape}")
    print(f"  Output shape: {data_long.shape}")
    # Comparar estadísticas
    print(f"  Mean input : {np.mean(data_short, axis=0)}")
    print(f"  Mean output: {np.mean(data_long, axis=0)}")
    print(f"  Std input  : {np.std(data_short, axis=0)}")
    print(f"  Std output : {np.std(data_long, axis=0)}")

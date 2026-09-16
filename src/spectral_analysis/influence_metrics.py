"""Extracción de pesos de influencia canal → dimensión latente.

Contiene la lógica para obtener los pesos de transformación desde la
metadata devuelta por ``extract_latent_space`` (PCA, PCA+ICA, DMD, ...),
un fallback por correlación para métodos no lineales (p. ej. Diffusion
Maps) y la métrica espectral de influencia por bandas de frecuencia.
"""

from __future__ import annotations

import logging

import numpy as np

__all__ = [
    "FREQ_BANDS",
    "extract_linear_weights",
    "compute_channel_latent_correlation",
    "compute_spectral_influence",
]

logger = logging.getLogger(__name__)

# Bandas de frecuencia estándar para la métrica espectral (§4.3.3).
# ``None`` como límite superior significa "hasta el fmax disponible".
FREQ_BANDS: dict[str, tuple[float, float | None]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, None),
}

_EPS = np.finfo(float).tiny


def _stage2_name(meta: dict) -> str:
    """Nombre del método dinámico de stage 2 (p. ej. 'pca', 'dmd')."""
    try:
        return str(meta["pipeline"]["stage2"]).lower()
    except Exception:
        return ""


def _expected_n_channels(meta: dict) -> int | None:
    """Número de canales esperado según meta['preprocessing'], si consta."""
    pre = meta.get("preprocessing")
    if not isinstance(pre, dict):
        return None
    ch = pre.get("ch_names")
    if ch is not None:
        try:
            return len(ch)
        except TypeError:
            pass
    n = pre.get("n_channels")
    return int(n) if n is not None else None


# Atributos de estimadores ajustados (MNE ICA, sklearn PCA/FastICA/TruncatedSVD)
# que pueden estar guardados dentro de meta["stage2"].
_ESTIMATOR_ATTRS = ("components_", "mixing_matrix_", "mixing_", "unmixing_matrix_")


def _matrix_from_estimator(obj) -> tuple[np.ndarray | None, str | None]:
    """Extrae una matriz 2D de un estimador ajustado (atributos conocidos)."""
    for attr in _ESTIMATOR_ATTRS:
        matrix = getattr(obj, attr, None)
        if matrix is not None and np.asarray(matrix).ndim == 2:
            return np.asarray(matrix), f"{type(obj).__name__}.{attr}"
    return None, None


def _find_matrix(stage2_meta: dict, keys: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    """Busca la primera clave con una matriz 2D en stage2_meta.

    Recorre un nivel de anidación de dicts y, si no hay coincidencia de
    clave, inspecciona los valores por si son estimadores ajustados
    (MNE ICA, sklearn PCA/ICA) con atributos matriciales conocidos.
    """
    for key in keys:
        if key in stage2_meta:
            candidate = stage2_meta[key]
            if isinstance(candidate, dict):  # p. ej. stage2['pca']['components_']
                for sub_key in keys:
                    if sub_key in candidate and np.asarray(candidate[sub_key]).ndim == 2:
                        return np.asarray(candidate[sub_key]), f"{key}.{sub_key}"
            elif np.asarray(candidate).ndim == 2:
                return np.asarray(candidate), key
    # Estimadores ajustados guardados como valores (o un nivel dentro).
    candidates = list(stage2_meta.items())
    for value in stage2_meta.values():
        if isinstance(value, dict):
            candidates.extend(value.items())
    for key, value in candidates:
        if isinstance(value, dict):
            continue
        matrix, attr = _matrix_from_estimator(value)
        if matrix is not None:
            return matrix, f"{key}:{attr}"
    return None, None


def _try_hankel_aggregation(
    matrix: np.ndarray, meta: dict, expected_n_channels: int
) -> tuple[np.ndarray | None, str]:
    """Agrega una matriz (n_basis, n_channels*depth) a (n_channels, n_basis).

    Cuando se usa embedding Hankel, la matriz de Stage 2 tiene
    ``n_channels * depth`` columnas (cada canal se replica con
    ``depth`` retardos).  Esta función reshapea a
    ``(n_basis, n_channels, depth)`` y promedia sobre el eje de
    retardos para obtener pesos a nivel de canal con su signo.

    Si el número de features Hankel no coincide exactamente con
    ``expected_n_channels * depth`` (p. ej. porque DMD aplicó
    average-reference internamente y eliminó un canal), se intenta
    inferir el número efectivo de canales como el divisor entero
    más cercano de la dimensión Hankel.
    """
    stage1 = meta.get("stage1") if isinstance(meta, dict) else None
    if not isinstance(stage1, dict) or stage1.get("embedding") != "hankel":
        return None, ""
    depth = stage1.get("depth")
    if depth is None or depth < 1:
        return None, ""

    r, c = matrix.shape
    target_size = expected_n_channels * depth

    # Orientar como (n_basis, n_channels * depth)
    if c == target_size:
        pass  # ya tiene la orientación correcta
    elif r == target_size:
        matrix = matrix.T
        r, c = c, r
    else:
        # Flexible fallback: la dimensión Hankel puede no coincidir
        # exactamente con expected_n_channels * depth si el método de
        # stage2 modificó el número de canales (p. ej. DMD con
        # average-reference + drop-last-channel).
        # Se infiere el número efectivo de canales como el divisor
        # entero más cercano.
        resolved = False
        for dim_size, needs_transpose in [(c, False), (r, True)]:
            if dim_size % depth == 0 and dim_size // depth > 0:
                n_eff_ch = dim_size // depth
                if needs_transpose:
                    matrix = matrix.T
                    r, c = c, r
                n_basis = r
                try:
                    aggregated = matrix.reshape(
                        n_basis, n_eff_ch, depth
                    ).mean(axis=2)
                    if n_eff_ch != expected_n_channels:
                        logger.info(
                            "Hankel aggregation: effective channels (%d) != "
                            "expected channels (%d); se usan %d filas de pesos.",
                            n_eff_ch, expected_n_channels, n_eff_ch,
                        )
                    return (
                        aggregated.T,
                        f"hankel_aggregated(n_eff_ch={n_eff_ch})",
                    )
                except ValueError:
                    continue
        return None, ""

    n_basis = r
    try:
        aggregated = matrix.reshape(n_basis, expected_n_channels, depth).mean(axis=2)
        return aggregated.T, "hankel_aggregated"
    except ValueError:
        return None, ""


def _orient_as_channel_by_basis(
    matrix: np.ndarray, expected_n_channels: int | None
) -> np.ndarray | None:
    """Orienta una matriz 2D como (n_channels, n_basis).

    Si se conoce el número de canales esperado se usa para decidir la
    orientación (fiable); en caso contrario se asume que la dimensión
    mayor corresponde a los canales (heurística).
    """
    if matrix.ndim != 2:
        return None
    r, c = matrix.shape
    if expected_n_channels:
        if r == expected_n_channels:
            return matrix
        if c == expected_n_channels:
            return matrix.T
        return None  # no encaja con los canales: no es una transformación canal→latente
    return matrix if r >= c else matrix.T


def extract_linear_weights(
    meta: dict,
    n_latent_dim: int,
) -> tuple[np.ndarray | None, str]:
    """Extrae pesos de influencia (n_channels, n_latent_dim) desde ``meta``.

    Maneja los métodos de ``stage2_dynamics``:

    - ``"pca"``: ``components_`` del PCA (filas = componentes).
    - ``"pca_ica"``: ``components_`` del PCA previo al ICA; si no consta,
      la matriz de mezcla ICA (``mixing_``/``A``).
    - ``"dmd"``: modos DMD (``modes``; se toma el módulo si son complejos).
    - Otros (p. ej. ``"diffusion_maps"``): búsqueda genérica de
      ``components_``/``eigenvectors``/``modes``; si no hay una
      transformación lineal canal→latente identificable, devuelve
      ``(None, motivo)`` para que el llamante use el fallback.

    Si ``meta['selected_indices']`` consta y hay más columnas que
    dimensiones latentes, se seleccionan las columnas correspondientes.

    Returns
    -------
    weights, method:
        ``weights`` es un array ``(n_channels, n_latent_dim)`` sin
        normalizar, o ``None`` si no se pudieron extraer. ``method`` es
        una cadena que describe cómo se calcularon.
    """
    if not isinstance(meta, dict):
        return None, "meta_not_dict"

    stage2_meta = meta.get("stage2")
    if not isinstance(stage2_meta, dict):
        return None, f"no_stage2_meta(stage2={_stage2_name(meta) or 'unknown'})"

    name = _stage2_name(meta)
    if "dmd" in name:
        keys = ("modes", "eigenvectors", "components_", "mixing_", "W", "A")
    elif "ica" in name:
        # PCA antes de ICA tiene prioridad; si no, matriz de mezcla ICA.
        keys = ("components_", "mixing_", "A", "W", "eigenvectors", "modes")
    else:
        keys = ("components_", "eigenvectors", "modes", "mixing_", "W", "A")

    matrix, found_key = _find_matrix(stage2_meta, keys)
    if matrix is None:
        return None, f"no_linear_weights(stage2={name or 'unknown'})"

    expected = _expected_n_channels(meta)
    weights = _orient_as_channel_by_basis(matrix, expected)
    method = f"{name or 'stage2'}:{found_key}"

    if weights is None and expected is not None:
        weights, hkl = _try_hankel_aggregation(matrix, meta, expected)
        if weights is not None:
            method += f"|{hkl}"

    if weights is None:
        return None, f"shape_mismatch(stage2={name or 'unknown'}, key={found_key})"

    if np.iscomplexobj(weights):
        weights = np.real(weights)
        method += "|real"

    # Selección de columnas según meta['selected_indices'] (stage 3).
    sel = meta.get("selected_indices")
    if (
        sel is not None
        and weights.shape[1] > n_latent_dim
        and len(sel) == n_latent_dim
        and int(np.max(sel)) < weights.shape[1]
    ):
        weights = weights[:, np.asarray(sel, dtype=int)]
        method += "|selected_indices"

    # Ajuste final del número de columnas a la dimensionalidad latente.
    if weights.shape[1] > n_latent_dim:
        weights = weights[:, :n_latent_dim]
    elif weights.shape[1] < n_latent_dim:
        logger.warning(
            "Se encontraron %d vectores de pesos pero el espacio latente tiene "
            "%d dimensiones; se rellena con ceros.",
            weights.shape[1], n_latent_dim,
        )
        pad = np.zeros((weights.shape[0], n_latent_dim - weights.shape[1]))
        weights = np.hstack([weights, pad])

    return np.asarray(weights, dtype=float), method


def compute_channel_latent_correlation(
    data: np.ndarray, latent: np.ndarray
) -> np.ndarray:
    """Fallback no lineal: correlación de Pearson canal–dimensión latente.

    Parameters
    ----------
    data:
        Señales originales ``(n_channels, n_times)``.
    latent:
        Espacio latente ``(n_samples, n_dim)``. Si es más corto que
        ``data`` (p. ej. por Hankel embedding, ``n_samples = n_times -
        depth + 1``), se toma el segmento centrado de ``data``.

    Returns
    -------
    weights:
        Array ``(n_channels, n_dim)`` con valores en ``[-1, 1]``
        (signo preservado; el llamante normaliza por máximo absoluto).
    """
    data = np.asarray(data, dtype=float)
    latent = np.asarray(latent, dtype=float)
    n_times = data.shape[1]
    n_samples = latent.shape[0]
    if n_samples > n_times:
        raise ValueError(
            f"El espacio latente ({n_samples} muestras) es más largo que "
            f"los datos originales ({n_times})."
        )
    offset = (n_times - n_samples) // 2
    segment = data[:, offset : offset + n_samples]

    def _zscore_rows(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std[std == 0.0] = 1.0
        return x / std

    z_data = _zscore_rows(segment)
    z_lat = _zscore_rows(latent.T)
    corr = (z_data @ z_lat.T) / n_samples
    return corr


def compute_spectral_influence(
    raw_psds: np.ndarray,
    latent_psds: np.ndarray,
    freqs_raw: np.ndarray,
    freqs_latent: np.ndarray,
    weights: np.ndarray,
    bands: dict[str, tuple[float, float | None]] | None = None,
) -> dict:
    """Métrica espectral de influencia por bandas de frecuencia (§4.3.3).

    Para cada dimensión latente y banda calcula:

    - ``spectral_contribution[c, d, b]``: contribución del canal ``c`` a la
      banda ``b`` de la dimensión ``d``, definida como
      ``|w[c, d]| · potencia_banda(c)`` normalizada entre canales. Responde
      a "¿qué canales contribuyen más a la potencia en alfa/beta de esta
      dimensión latente?".
    - ``spectral_correlation[c, d, b]``: correlación de Pearson entre el
      PSD del canal y el PSD de la dimensión latente dentro de la banda
      (en escala log10), útil para detectar coincidencia espectral.

    Si las rejillas de frecuencia difieren (p. ej. latente más corto por
    Hankel), el PSD latente se interpola sobre las frecuencias del raw.

    Returns
    -------
    dict con ``band_names``, ``band_ranges`` ``(n_bands, 2)``,
    ``spectral_contribution`` ``(n_channels, n_dim, n_bands)`` y
    ``spectral_correlation`` (misma forma).
    """
    bands = dict(bands or FREQ_BANDS)
    freqs_raw = np.asarray(freqs_raw, dtype=float)
    freqs_latent = np.asarray(freqs_latent, dtype=float)
    raw_psds = np.asarray(raw_psds, dtype=float)
    latent_psds = np.asarray(latent_psds, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_channels, n_dim = weights.shape
    if latent_psds.ndim == 1:
        latent_psds = latent_psds[None, :]
    if latent_psds.shape[0] != n_dim and latent_psds.shape[1] == n_dim:
        latent_psds = latent_psds.T

    # Alinear rejillas de frecuencia (interpola el latente sobre el raw).
    if freqs_latent.shape != freqs_raw.shape or not np.allclose(freqs_latent, freqs_raw):
        latent_psds = np.stack(
            [
                np.interp(freqs_raw, freqs_latent, row, left=np.nan, right=np.nan)
                for row in latent_psds
            ]
        )

    trapz = getattr(np, "trapezoid", None) or np.trapz
    band_names: list[str] = []
    band_ranges: list[tuple[float, float]] = []
    contribution = np.full((n_channels, n_dim, len(bands)), np.nan)
    correlation = np.full((n_channels, n_dim, len(bands)), np.nan)

    for b, (name, (lo, hi)) in enumerate(bands.items()):
        hi_eff = float(hi) if hi is not None else float(freqs_raw.max())
        mask = (freqs_raw >= lo) & (freqs_raw <= hi_eff)
        band_names.append(name)
        band_ranges.append((float(lo), hi_eff))
        if int(mask.sum()) < 2:
            continue

        # Contribución ponderada por potencia de banda del canal.
        band_power = trapz(np.maximum(raw_psds[:, mask], _EPS), freqs_raw[mask], axis=1)
        contrib = np.abs(weights) * band_power[:, None]
        total = contrib.sum(axis=0, keepdims=True)
        total[total == 0.0] = np.nan
        contribution[:, :, b] = contrib / total

        # Correlación espectral dentro de la banda (escala dB).
        if int(mask.sum()) >= 4:
            log_raw = np.log10(np.maximum(raw_psds[:, mask], _EPS))
            log_lat = np.log10(np.maximum(latent_psds[:, mask], _EPS))
            a = log_raw - log_raw.mean(axis=1, keepdims=True)
            for d in range(n_dim):
                row = log_lat[d]
                if not np.all(np.isfinite(row)):
                    continue
                c = row - row.mean()
                denom = np.sqrt((a**2).sum(axis=1) * float((c**2).sum()))
                with np.errstate(invalid="ignore", divide="ignore"):
                    correlation[:, d, b] = (a @ c) / denom

    return {
        "band_names": band_names,
        "band_ranges": np.asarray(band_ranges, dtype=float),
        "spectral_contribution": contribution,
        "spectral_correlation": correlation,
    }

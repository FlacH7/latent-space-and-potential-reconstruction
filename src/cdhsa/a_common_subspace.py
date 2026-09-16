"""
cdhsa/a_common_subspace.py — Steps A1-A5 of CD-HSA
=====================================================

Estima el subespacio Hankel común poblacional a partir de grabaciones
EEG multicanal de múltiples sujetos y condiciones.

NOTA CRÍTICA DE DISEÑO
------------------------
Este módulo traduce fielmente el algoritmo MATLAB cdhsa_A1_A5.m del tutor,
con las siguientes mejoras y correcciones identificadas:

1. **Normalización geométrica (A1)**: El MATLAB normaliza H por ||H||_F.
   Esto NO cambia U_sc (la base izquierda de la SVD es invariante a escalado
   escalar). Lo conservamos por compatibilidad con B/C (que sí necesita
   la norma original para calcular energía), pero documentamos que es
   irrelevante para A1-A5.

2. **Selección de rank por reproducibilidad (A2)**: El MATLAB usa un
   criterio "consecutivo desde r=1" — si R(3) cae por debajo del umbral,
   se detiene en r=2 aun cuando R(4) y R(5) puedan ser altos. Esto puede
   subestimar el rank en presencia de un componente débil temprano.
   Implementamos el mismo criterio por fidelidad, pero añadimos el
   parámetro ``gap_based=True`` que selecciona por gaps en R(r).

3. **SVD de B para M₀ (A3)**: La concatenación B = [U₁...U_N]/√N y su
   SVD izquierda es equivalente a eigendecompose M₀ = (1/N)Σ U_i U_i^T.
   El MATLAB lo hace bien. Aquí usamos np.linalg.svd en vez de svds porque
   B típicamente tiene dimensiones (pL, Σr_sc) donde Σr_sc << pL.

4. **Verificación de consistencia de canales**: Añadimos chequeo explícito
   de que todos los X[s][c] tengan el mismo número de canales p.

5. **Tipado y documentación**: Docstrings completos con tipos, shape
   annotations, y referencias a las ecuaciones del marco teórico.

Dependencias: numpy, scipy.sparse.linalg (solo para SVD truncada grande).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import svds


# ============================================================================
# 0. Block Hankel construction
# ============================================================================

def build_block_hankel(X: NDArray[np.floating], L: int) -> NDArray[np.floating]:
    """
    Construye la matriz de Hankel multicanal por bloques.

    Para X de shape (p, T), la columna t-ésima de H es::

        H[:, t] = [x(t), x(t-1), ..., x(t-L+1)]^T

    Es decir, muestra actual primero, luego muestras previas.
    El resultado tiene shape (p*L, T-L+1).

    Parameters
    ----------
    X : array, shape (p, T)
        Datos EEG canales × tiempo.
    L : int
        Número de retardos (embedding depth).

    Returns
    -------
    H : array, shape (p*L, K)  donde K = T - L + 1

    Raises
    ------
    ValueError
        Si T < L.

    CRÍTICA vs MATLAB
    ----------------
    El MATLAB (build_block_hankel) usa la misma convolución:
        H(rows,:) = X(:, L-ell+1:T-ell+1)
    con ell=1..L. Esto produce x(t) primero. Nuestra implementación
    es equivalente.

    NOTA sobre convención del proyecto existente:
    El proyecto usa ``_build_multivariate_hankel`` de hankel_dmd_extractor.
    Si esa función usa una convención diferente (ej. x(t-L+1) primero),
    los subespacios U serán los mismos (las columnas solo se reordenan
    dentro del bloque), pero los patrones espaciales-temporales reshaped
    W_j(e,τ) tendrán el eje τ invertido. Verificar antes de interpretar.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X debe ser 2-D (p × T), got shape {X.shape}")
    p, T = X.shape
    if T < L:
        raise ValueError(
            f"T ({T}) debe ser >= L ({L}) para construir Hankel"
        )
    K = T - L + 1
    H = np.empty((p * L, K), dtype=np.float64)
    for ell in range(L):
        # ell=0 → muestras más recientes (x(t)     hasta x(t-L+1)  en la primera columna)
        # ell=L-1 → muestras más antiguas  (x(t-L+1) hasta x(t-2L+2) en la primera columna)
        H[ell * p : (ell + 1) * p, :] = X[:, L - 1 - ell : T - ell]
    return H


# ============================================================================
# 1. Truncated SVD helpers
# ============================================================================

def truncated_left_svd(
    H: NDArray[np.floating],
    r: int,
) -> NDArray[np.floating]:
    """
    Primeros ``r`` vectores singulares izquierdos de H.

    Usa ``scipy.sparse.linalg.svds`` cuando r < min(m,n)-1 para eficiencia,
    con fallback a SVD densa.

    Parameters
    ----------
    H : array, shape (m, n)
    r : int
        Número de componentes a retener.

    Returns
    -------
    U : array, shape (m, r)
        Columnas ortonormales.

    CRÍTICA vs MATLAB
    ----------------
    ``scipy.sparse.linalg.svds`` devuelve valores singulares en orden
    **ascendente**, a diferencia del MATLAB ``svds(...,'largest')`` que
    devuelve descendente. Ordenamos explícitamente.

    Además, ``svds`` puede fallar con ARNOLDI para matrices con valores
    singulares degenerados (común en oscilaciones Hankel). El fallback
    a SVD densa cubre ese caso.
    """
    m, n = H.shape
    r = min(r, m, n)
    if r < 1:
        raise ValueError(f"r debe ser >= 1, got {r}")

    # Usar svds solo cuando es seguro y ventajoso
    if r < min(m, n) - 1:
        try:
            U, s_vals, _ = svds(H, k=r)
            # scipy svds → orden ascendente, corregir
            idx = np.argsort(s_vals)[::-1]
            return U[:, idx]
        except Exception:
            pass  # fallback

    # SVD densa
    U, _, _ = np.linalg.svd(H, full_matrices=False)
    return U[:, :r]


def truncated_left_svd_with_values(
    H: NDArray[np.floating],
    r: int,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Primeros ``r`` vectores singulares izquierdos y sus valores.

    Returns
    -------
    U : array, shape (m, r)
    s : array, shape (r,)
        Valores singulares en orden descendente.
    """
    m, n = H.shape
    r = min(r, m, n)
    if r < 1:
        raise ValueError(f"r debe ser >= 1, got {r}")

    if r < min(m, n) - 1:
        try:
            U, s_vals, _ = svds(H, k=r)
            idx = np.argsort(s_vals)[::-1]
            return U[:, idx], s_vals[idx]
        except Exception:
            pass

    U_full, s_full, _ = np.linalg.svd(H, full_matrices=False)
    return U_full[:, :r], s_full[:r]


# ============================================================================
# 2. Rank selection by temporal-block reproducibility (A2 helper)
# ============================================================================

def select_rank_reproducibility(
    X: NDArray[np.floating],
    L: int,
    rmax: int = 20,
    n_blocks: int = 4,
    threshold: float = 0.80,
    strategy: Literal["consecutive", "gap"] = "consecutive",
) -> tuple[int, NDArray[np.floating]]:
    """
    Estima el rank de señal Hankel local via reproducibilidad
    entre bloques temporales.

    Para cada par de bloques (a, b) y cada candidato r::

        R(r) = (1 / n_pairs) * Σ_{a<b} ||U_a(r)^T U_b(r)||_F² / r

    R(r) = 1 ⟹ subespacios idénticos de dimensión r.
    R(r) ≈ 0 ⟹ sin estructura compartida.

    Parameters
    ----------
    X : array, shape (p, T)
    L : int
        Retardos Hankel.
    rmax : int
        Máximo rank candidato.
    n_blocks : int
        Número de bloques temporales.
    threshold : float
        Umbral mínimo de reproducibilidad.
    strategy : {'consecutive', 'gap'}
        'consecutive': el MATLAB original — mayor r consecutivo desde 1.
        'gap': seleccionar por el mayor gap en R(r), más robusto ante
        un componente temprano débil.

    Returns
    -------
    r_selected : int
    R : array, shape (r_allowed,)
        Curva de reproducibilidad.

    CRÍTICA vs MATLAB
    ----------------
    1. **Criterio consecutivo**: El MATLAB selecciona el mayor r tal que
       R(1:r) >= threshold para todo r. Si R(3) = 0.79 (justo debajo),
       se para en r=2 aunque R(4) = 0.95. Esto subestima sistemáticamente
       cuando hay un componente débil interleaved. El parámetro
       ``strategy='gap'`` corrige esto.

    2. **edges con round()**: El MATLAB usa ``round(linspace(...))``, que
       puede producir bloques desiguales o vacíos. Usamos
       ``np.linspace`` con int() que es más predecible.

    3. **Bloques demasiado cortos**: Si T/n_blocks < L + 2, los bloques
       no tienen suficientes columnas Hankel. El MATLAB lanza error; aquí
       damos un warning y reducimos n_blocks automáticamente.
    """
    X = np.asarray(X, dtype=np.float64)
    _, T = X.shape

    # Validar que los bloques sean factibles
    min_block_len = L + 2
    max_feasible_blocks = T // min_block_len
    if max_feasible_blocks < 2:
        raise ValueError(
            f"Grabación demasiado corta (T={T}) para L={L} con >= 2 bloques. "
            f"Se necesitan al menos {2 * min_block_len} muestras."
        )

    effective_blocks = min(n_blocks, max_feasible_blocks)
    if effective_blocks < n_blocks:
        import warnings
        warnings.warn(
            f"Reduciendo n_blocks de {n_blocks} a {effective_blocks} "
            f"(T={T}, L={L}, se necesitan >={min_block_len} muestras por bloque)",
            stacklevel=2,
        )

    # Dividir en bloques temporales
    edges = np.linspace(0, T, effective_blocks + 1, dtype=int)

    bases: list[NDArray[np.floating]] = []
    r_allowed = rmax

    for b in range(effective_blocks):
        Xb = X[:, edges[b] : edges[b + 1]]
        if Xb.shape[1] <= L:
            raise ValueError(
                f"Bloque {b} tiene solo {Xb.shape[1]} columnas, "
                f"necesitas > {L} para Hankel con L={L}."
            )
        Hb = build_block_hankel(Xb, L)
        hnorm = np.linalg.norm(Hb, "fro")
        if hnorm < np.finfo(np.float64).eps:
            raise ValueError(f"Bloque {b} tiene Hankel casi cero.")
        Hb = Hb / hnorm
        rb = min(rmax, Hb.shape[0], Hb.shape[1])
        bases.append(truncated_left_svd(Hb, rb))
        r_allowed = min(r_allowed, rb)

    # Calcular reproducibilidad
    R = np.zeros(r_allowed, dtype=np.float64)
    n_pairs = 0

    for a in range(effective_blocks - 1):
        for b in range(a + 1, effective_blocks):
            n_pairs += 1
            Ua = bases[a]
            Ub = bases[b]
            for r in range(1, r_allowed + 1):
                G = Ua[:, :r].T @ Ub[:, :r]
                R[r - 1] += np.linalg.norm(G, "fro") ** 2 / r

    R /= n_pairs

    # Selección de rank
    if strategy == "consecutive":
        # MATLAB original: mayor r consecutivo desde 1
        r_selected = 1
        for r in range(1, r_allowed + 1):
            if np.all(R[:r] >= threshold):
                r_selected = r
            else:
                break
    elif strategy == "gap":
        # Mayor gap en R(r), buscando el primer descenso pronunciado
        # Añadimos R(0)=1.0 y R(r_allowed+1)=0 como boundary
        R_padded = np.concatenate([[1.0], R, [0.0]])
        drops = -np.diff(R_padded)
        # Excluir el primer drop (de 1.0 a R(1)) que es artificial
        drops[0] = 0.0
        best_gap_idx = np.argmax(drops[1:]) + 1  # +1 por el padding
        r_selected = max(1, best_gap_idx)
    else:
        raise ValueError(f"strategy debe ser 'consecutive' o 'gap', got '{strategy}'")

    return r_selected, R


# ============================================================================
# 3. Main: Steps A1-A5
# ============================================================================

def cdhsa_A1_A5(
    X: list[list[NDArray[np.floating]]],
    L: int,
    *,
    rank_method: Literal["fixed", "reproducibility"] = "fixed",
    fixed_rank: int = 10,
    rmax: int = 20,
    n_blocks: int = 4,
    repro_threshold: float = 0.80,
    repro_strategy: Literal["consecutive", "gap"] = "consecutive",
    max_common: int = 30,
    prevalence_quantile: float = 0.10,
) -> dict:
    """
    CD-HSA Steps A1-A5: estimación del subespacio Hankel común poblacional.

    **A1**: Construir Hankel por bloques y normalizar geométricamente.

    **A2**: Estimar el rank de señal confiable de cada grabación.
      - 'fixed': usar ``fixed_rank`` directamente.
      - 'reproducibilidad': reproducibilidad entre bloques temporales.

    **A3**: Estimar el subespacio común poblacional vía SVD de la
      concatenación de bases locales B = [U₁ ... U_N] / √N.
      Los autovalores λ_j = σ_j²(B) miden la commonalidad de cada dirección.

    **A4**: Calcular la matriz de alineación a_{sc,j} = ||U_sc^T w_j||²,
      que cuantifica cuánto de cada dirección común está presente en
      cada grabación individual.

    **A5**: Resumir commonalidad (media) y prevalencia (percentil inferior)
      de cada dirección poblacional.

    Parameters
    ----------
    X : list of list of arrays
        X[s][c] es un array (p, T) con los datos EEG del sujeto s,
        condición c. Canales en filas, tiempo en columnas.
    L : int
        Número de retardos Hankel (embedding depth).
    rank_method : {'fixed', 'reproducibility'}
    fixed_rank : int
        Rank local fijo cuando ``rank_method='fixed'``.
    rmax : int
        Máximo rank candidato para reproducibilidad.
    n_blocks : int
        Bloques temporales para reproducibilidad.
    repro_threshold : float
        Umbral de reproducibilidad.
    repro_strategy : {'consecutive', 'gap'}
        Estrategia de selección de rank (ver ``select_rank_reproducibility``).
    max_common : int
        Máximo número de direcciones comunes a estimar.
    prevalence_quantile : float
        Percentil inferior para la medida de prevalencia.

    Returns
    -------
    result : dict with keys:
        U : list of list of arrays
            U[s][c] shape (p*L, r_sc) — base local Hankel ortonormal.
        rank : ndarray, shape (S, C)
            Rank local seleccionado para cada grabación.
        hankel_norm : ndarray, shape (S, C)
            ||H_sc||_F original (antes de normalización).
        repro_curve : list of list of arrays or None
            Curvas de reproducibilidad (solo si rank_method='reproducibility').
        W : ndarray, shape (p*L, q)
            Direcciones comunes poblacionales (columnas ortonormales).
        lambda_ : ndarray, shape (q,)
            Autovalores de commonalidad λ_j ∈ [0, 1].
        alignment : ndarray, shape (S*C, q)
            a_{sc,j} = ||U_sc^T w_j||² para cada grabación y dirección.
        sc_index : ndarray, shape (S*C, 2)
            (s, c) correspondiente a cada fila de alignment.
        mean_alignment : ndarray, shape (q,)
        median_alignment : ndarray, shape (q,)
        prevalence : ndarray, shape (q,)
        min_alignment : ndarray, shape (q,)
        S : int, C : int, p : int, d : int
            Dimensiones del problema.
        L_used : int
            L tal cual se pasó.

    CRÍTICA GLOBAL vs MATLAB
    ------------------------
    - La lógica es equivalente al MATLAB cdhsa_A1_A5.m.
    - Corregimos el orden ascendente/descendente de scipy svds.
    - Añadimos ``repro_strategy='gap'`` como alternativa al criterio
      consecutivo del MATLAB.
    - Añadimos warnings cuando se reducen bloques automáticamente.
    - Normalización por ||H||_F: conservada por compatibilidad con B/C,
      aunque no afecta a U_sc.
    - No construimos los proyectores pL×pL explícitamente (igual que MATLAB).
    """
    # ---- Dimensiones del problema ----
    S = len(X)
    C = len(X[0])
    N = S * C

    # Validar dimensionalidad
    p = np.asarray(X[0][0]).shape[0]
    for s in range(S):
        for c in range(C):
            Xsc = np.asarray(X[s][c])
            if Xsc.ndim != 2 or Xsc.shape[0] != p:
                raise ValueError(
                    f"X[{s}][{c}] tiene shape {Xsc.shape}, "
                    f"se espera (p={p}, T) con T >= L+2={L + 2}"
                )
            if Xsc.shape[1] < L + 2:
                raise ValueError(
                    f"X[{s}][{c}] tiene T={Xsc.shape[1]}, "
                    f"demasiado corto para L={L} (necesita >= {L + 2})"
                )

    # ---- A1-A2: Hankel + normalización + rank local ----
    U_all: list[list[NDArray]] = [[None] * C for _ in range(S)]
    ranks = np.zeros((S, C), dtype=int)
    hankel_norms = np.zeros((S, C), dtype=np.float64)
    repro_curves: list[list[NDArray | None]] = [[None] * C for _ in range(S)]

    for s in range(S):
        for c in range(C):
            Xi = np.asarray(X[s][c], dtype=np.float64)
            H = build_block_hankel(Xi, L)
            hnorm = np.linalg.norm(H, "fro")
            if hnorm <= np.finfo(np.float64).eps:
                raise ValueError(
                    f"Hankel casi cero en sujeto {s}, condición {c}"
                )
            hankel_norms[s, c] = hnorm

            # A1: normalización geométrica (no cambia U_sc, pero la
            # guardamos para B/C que necesita la escala original)
            Hgeom = H / hnorm

            # A2: rank local
            if rank_method == "fixed":
                r = min(fixed_rank, Hgeom.shape[0], Hgeom.shape[1])
                repro = None
            elif rank_method == "reproducibility":
                r, repro = select_rank_reproducibility(
                    Xi, L,
                    rmax=rmax,
                    n_blocks=n_blocks,
                    threshold=repro_threshold,
                    strategy=repro_strategy,
                )
            else:
                raise ValueError(
                    f"rank_method debe ser 'fixed' o 'reproducibility', got '{rank_method}'"
                )

            Ui = truncated_left_svd(Hgeom, r)
            U_all[s][c] = Ui
            ranks[s, c] = r
            repro_curves[s][c] = repro

    # ---- A3: subespacio común poblacional ----
    # M₀ = (1/N) Σ U_sc U_sc^T
    # Equivale a SVD de B = [U₁...U_N]/√N
    Ucat = np.concatenate(
        [U_all[s][c] for s in range(S) for c in range(C)],
        axis=1,
    )
    B = Ucat / np.sqrt(N)

    qmax = min(max_common, B.shape[0], B.shape[1])
    W, sigma_B = truncated_left_svd_with_values(B, qmax)
    lambda_ = sigma_B ** 2
    lambda_ = np.clip(lambda_, 0.0, 1.0)  # clipping numérico

    # ---- A4: alineación por grabación ----
    alignment = np.zeros((N, qmax), dtype=np.float64)
    sc_index = np.zeros((N, 2), dtype=int)
    row = 0
    for s in range(S):
        for c in range(C):
            Ui = U_all[s][c]
            G = W.T @ Ui  # shape (qmax, r_sc)
            alignment[row, :] = np.sum(G ** 2, axis=1)
            sc_index[row, :] = [s, c]
            row += 1

    # ---- A5: commonalidad y prevalencia ----
    mean_alignment = np.mean(alignment, axis=0)
    median_alignment = np.median(alignment, axis=0)
    prevalence = np.quantile(alignment, prevalence_quantile, axis=0)
    min_alignment = np.min(alignment, axis=0)

    return {
        "U": U_all,
        "rank": ranks,
        "hankel_norm": hankel_norms,
        "repro_curve": repro_curves,
        "W": W,
        "lambda_": lambda_,
        "alignment": alignment,
        "sc_index": sc_index,
        "mean_alignment": mean_alignment,
        "median_alignment": median_alignment,
        "prevalence": prevalence,
        "min_alignment": min_alignment,
        "S": S,
        "C": C,
        "p": p,
        "d": p * L,
        "L_used": L,
        "N": N,
    }

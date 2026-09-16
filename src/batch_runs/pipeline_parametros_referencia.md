# Referencia Completa de Parámetros del Pipeline Orquestador de Espacio Latente EEG

## Descripción General

El orquestador principal es la función **`extract_latent_space()`** (definida en `extract_latent_subspace.py`), que implementa un pipeline modular de **3 etapas** para extraer un subespacio latente de baja dimensionalidad a partir de grabaciones EEG:

```
Stage 1 — Embedding    : None | "hankel"
Stage 2 — Dinámica    : "pca_ica" | "pca" | "dmd" | "diffusion_maps"
Stage 3 — Selección   : "top_n" | "markov_fastest" | "markov_slowest"
```

El flujo de datos interno sigue la convención de forma **(n_features, n_samples)** — las filas son variables/modos/canales y las columnas son tiempo. Solo al final el orquestador transpone el resultado a **(n_samples, n_dim)** para la salida.

---

## Tabla de Contenidos

1. [Parámetros de la API Nueva (3 Etapas)](#1-parámetros-de-la-api-nueva-3-etapas)
2. [Parámetros Legacy (deprecados)](#2-parámetros-legacy-deprecados)
3. [Stage 1 — Embedding: Parámetros por Variante](#3-stage-1--embedding-parámetros-por-variante)
4. [Stage 2 — Dinámica: Parámetros por Variante](#4-stage-2--dinámica-parámetros-por-variante)
5. [Stage 3 — Selección: Parámetros por Variante](#5-stage-3--selección-parámetros-por-variante)
6. [Tabla de Combinaciones Válidas](#6-tabla-de-combinaciones-válidas)
7. [Mapeo Legacy → Nueva API](#7-mapeo-legacy--nueva-api)
8. [Parámetros del Script CLI (`test_iga_from_eeg...`)](#8-parámetros-del-script-cli)
9. [Estructura del Diccionario `meta` Devuelto](#9-estructura-del-diccionario-meta-devuelto)
10. [Funciones Auxiliares del Orquestador](#10-funciones-auxiliares-del-orquestador)

---

## 1. Parámetros de la API Nueva (3 Etapas)

Estos son los parámetros de la firma principal de `extract_latent_space()` correspondientes a la nueva API modular. Todos los parámetros de etapa se pasan como **argumentos de palabra clave** (`kwargs`).

### 1.1 Parámetro de Entrada

| Parámetro | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `raw_input` | `mne.io.Raw \| str \| Path` | *(requerido)* | Objeto MNE Raw en memoria, o ruta a un fichero de datos EEG crudo (`.fif`, `.set`). Es la única entrada posicional de la función; todo lo demás se pasa como `kwargs`. |

### 1.2 Dimensión del Subespacio

| Parámetro | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `n_dim` | `int` | `2` | Dimensionalidad objetivo del subespacio latente. Es el número de modos/componentes que se seleccionarán al final. Debe ser `>= 1` y `<= D` (donde `D` es el número de modos que produce la Etapa 2). Por ejemplo, `n_dim=2` produce un espacio latente 2D para visualización de trayectorias. |

### 1.3 Etapa 1 — Embedding

| Parámetro | Tipo | Por Defecto | Valores Válidos | Descripción |
|---|---|---|---|---|
| `stage1_embedding` | `Literal[None, "hankel"]` | `None` | `None`, `"hankel"` | Tipo de *embedding* temporal. `None` significa passthrough (la señal filtrada pasa sin cambios). `"hankel"` construye la matriz de Hankel por bloques multivariada `H ∈ R^{(N_c·T) × (N_t-T+1)}` donde `T` es la profundidad de retardo. |
| `stage1_params` | `dict \| None` | `None` | — | Diccionario de parámetros específicos del embedding elegido. Se pasa directamente al constructor de la clase correspondiente. Si `stage1_embedding=None`, este diccionario **debe** estar vacío (lanza `ValueError` si no lo está). |

**Parámetros dentro de `stage1_params` para `"hankel"`:**

| Clave | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `depth` | `int \| None` | `None` | Profundidad del *embedding* de retardo `T` (número de *snapshots* temporales apilados por columna). `None` → auto-calculado como `clip(sfreq * 0.25, 50, 200)` acotado además por `n_times // 10`. Debe ser estrictamente `< n_times` (si no, lanza `ValueError`). Valores típicos: 30-250. Cuanto mayor, mayor memoria temporal capturada, pero menos muestras efectivas. |

### 1.4 Etapa 2 — Dinámica

| Parámetro | Tipo | Por Defecto | Valores Válidos | Descripción |
|---|---|---|---|---|
| `stage2_dynamics` | `Literal["pca_ica", "pca", "dmd", "diffusion_maps"]` | `"pca_ica"` | `"pca"`, `"pca_ica"`, `"dmd"`, `"diffusion_maps"` | Método de descomposición dinámica. Ver la [Sección 4](#4-stage-2--dinámica-parámetros-por-variante) para los detalles de cada variante y sus parámetros específicos. |
| `stage2_params` | `dict \| None` | `None` | — | Diccionario de parámetros para la dinámica elegida. Las claves válidas dependen de `stage2_dynamics`. Ver la [Sección 4](#4-stage-2--dinámica-parámetros-por-variante) para la lista completa. |

### 1.5 Etapa 3 — Selección

| Parámetro | Tipo | Por Defecto | Valores Válidos | Descripción |
|---|---|---|---|---|
| `stage3_selection` | `Literal["top_n", "markov_fastest", "markov_slowest"]` | `"top_n"` | `"top_n"`, `"markov_fastest"`, `"markov_slowest"` | Método de selección de modos finales. `"top_n"` mantiene los primeros `n_dim` modos (ya ordenados por la Etapa 2). `"markov_fastest"` busca el subespacio con el **menor** tiempo de relajación de Markov τ (dinámicas más ricas). `"markov_slowest"` busca el **mayor** τ. Ver la [Sección 5](#5-stage-3--selección-parámetros-por-variante) para más detalles. |
| `stage3_params` | `dict \| None` | `None` | — | Diccionario de parámetros para la selección elegida. Ver la [Sección 5](#5-stage-3--selección-parámetros-por-variante). |

### 1.6 Preprocesamiento

| Parámetro | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `l_freq` | `float` | `1.0` | Frecuencia de corte inferior del filtro *band-pass* en Hz. Se aplica antes de cualquier etapa (excepto cuando `stage2="pca_ica"` sin Hankel, donde se delega a `run_full_preprocessing`). Un valor típico es 1.0 Hz para eliminar drifts lentos. |
| `h_freq` | `float` | `40.0` | Frecuencia de corte superior del filtro *band-pass* en Hz. Un valor típico es 40.0 Hz que elimina ruido de línea (50/60 Hz) y artefactos musculares de alta frecuencia. |

### 1.7 Computación y Paralelización

| Parámetro | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `n_workers` | `int \| None` | `None` | Número de procesos paralelos para la búsqueda combinatoria en Etapa 3 (solo aplicable a `markov_fastest`, `markov_slowest` y las estrategias legacy). `None` usa todos los núcleos disponibles. `1` fuerza ejecución serial. |
| `verbose` | `bool \| str \| None` | `None` | Nivel de verbosidad MNE. `True`/`"INFO"` activa logs; `False`/`None` los silencia. Se propaga al filtrado MNE y al preprocesamiento ICA. |

---

## 2. Parámetros Legacy (deprecados)

Estos parámetros existen **solo para compatibilidad hacia atrás**. Cuando se pasa `scoring_method`, se emite un `DeprecationWarning` y todos los argumentos legacy se mapean internamente a la nueva API de 3 etapas mediante la función `map_legacy_scoring_method()`. **No deben usarse en código nuevo.**

| Parámetro | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `scoring_method` | `str \| None` | `None` | Método de *scoring* legacy. Valores válidos: `"markov"`, `"markov_inverted"`, `"conservative"`, `"weighted"`, `"sequential"`, `"pareto"`, `"independent"`, `"hankel_dmd"`, `"diffusion_maps"`. Ver la [Sección 7](#7-mapeo-legacy--nueva-api) para la tabla de mapeo completa. |
| `fc_metric` | `str` | `"variance_sum"` | Métrica de varianza para la *conservative fraction*. Valores: `"variance_sum"`, `"first_pc_var"`, `"total_variance"`. Solo se usa con `scoring_method` en `{"conservative", "weighted", "sequential", "pareto", "independent"}`. |
| `n_bins` | `int` | `5` | Número de *bins* de cuantiles para la discretización de Markov. Solo se usa con métodos que involucran Markov. |
| `alpha` | `float \| None` | `None` | Parámetro de trade-off para la estrategia `"weighted"` (combinación convexa normalizada de FC y 1/τ). Requerido cuando `scoring_method="weighted"`. |
| `primary_criterion` | `Literal["fc", "markov"]` | `"markov"` | Criterio primario para la estrategia `"sequential"` (filtra primero por este criterio, luego reordena por el otro). |
| `sequential_K` | `int \| None` | `None` | Número de candidatos a retener en la primera fase de `"sequential"`. Si `None`, se auto-calcula como `max(5, int(total_combs * 0.05))` acotado a `[5, 20]`. |
| `hankel_embedding_depth` | `int \| None` | `None` | Profundidad de *embedding* Hankel para `scoring_method="hankel_dmd"`. `None` → auto (ver descripción de `depth` en la Etapa 1). |
| `diffusion_sigma` | `float \| None` | `None` | *Bandwidth* σ del kernel Gaussiano para Diffusion Maps. `None` → auto-estimación BGH. |
| `diffusion_k` | `int` | `100` | Número de vecinos más cercanos para la matriz de afinidad dispersa en Diffusion Maps. |
| `diffusion_time` | `float` | `0.0` | Tiempo de difusión t ≥ 0. Valores mayores atenúan componentes de fina escala. `t=0` significa sin filtrado (autovectores puros). |
| `diffusion_alpha` | `float` | `0.5` | Parámetro de normalización de densidad (Coifman-Lafon). `0.0` = Laplacian Eigenmaps, `0.5` = Diffusion Maps clásico, `1.0` = Fokker-Planck. |
| `n_components` | `int \| float \| None` | `None` | Número de componentes ICA/SVD. En la rama MNE-ICA, `None` = todos los canales − 1; un `float` en (0,1] indica fracción de varianza. En la rama Hankel-ICA, debe ser `int`. |
| `ica_method` | `str` | `"picard"` | Algoritmo ICA de MNE (solo rama sin Hankel). Valores típicos: `"picard"`, `"fastica"`, `"infomax"`. |
| `ica_random_state` | `int \| None` | `42` | Semilla aleatoria para la reproducibilidad del ICA. |
| `retained_labels` | `list[str] \| None` | `None` | Clases ICLabel a retener tras el rechazo de artefactos (solo rama MNE-ICA sin Hankel). Por ejemplo `["brain", "other"]`. Si `None`, se usan las etiquetas por defecto. |
| `search_strategy` | `Literal["exhaustive", "greedy"]` | `"exhaustive"` | Estrategia de búsqueda combinatoria. `"greedy"` se fuerza automáticamente cuando `n_dim >= 4` (por costo combinatorio). |

---

## 3. Stage 1 — Embedding: Parámetros por Variante

### 3.1 `None` (IdentityEmbedding)

La señal de canales filtrada pasa sin cambios. **No acepta parámetros.** Si se pasa algún parámetro en `stage1_params`, se lanza un `ValueError`.

- **Input**: `X` con shape `(N_c, N_t)` — matriz de canales filtrados.
- **Output**: la misma `X`, con `n_time_lost = 0`.
- **Meta clave**: `meta["stage1"]["output_shape"] == (N_c, N_t)`.

### 3.2 `"hankel"` (HankelEmbedding)

Construye la matriz de Hankel por bloques multivariada. Cada bloque corresponde a un canal, y las columnas de H son vectores de retardo que concatenan `T` *snapshots* temporales de todos los canales.

**Parámetros de `stage1_params`:**

| Clave | Tipo | Requerido | Por Defecto | Descripción |
|---|---|---|---|---|
| `depth` | `int \| None` | No | `None` | Profundidad de retardo T. `None` → auto-calculado como `int(clip(sfreq * 0.25, 50, 200))`, acotado además por `n_times // 10` (mínimo absoluto 10). Debe satisfacer `depth < n_times`. Para sfreq=250 Hz, el auto-cálculo da 62 (250×0.25=62.5, clip→62). |

- **Input**: `X` con shape `(N_c, N_t)`.
- **Output**: `H` con shape `(N_c * T, N_t - T + 1)`. Por ejemplo, 8 canales con depth=30 sobre 3000 muestras produce una matriz `(240, 2971)`.
- **Meta clave**: `meta["stage1"]["depth"]`, `meta["stage1"]["n_time_lost"] = depth - 1`.
- **Nota importante**: No se aplica referencia promedio ni eliminación de canales aquí — esas operaciones son específicas de la rama DMD y viven dentro de `eeg_hankel_dmd_core`.

### 3.3 Cálculo Automático de `depth`

La función `_auto_embedding_depth(sfreq, n_times)` implementa la siguiente lógica:

```
t_samples = int(sfreq * 0.25)      # ~250 ms de muestras
t_samples = max(50, min(t_samples, 200))  # acotar a [50, 200]
t_samples = min(t_samples, n_times // 10)     # no consumir >10% del signal
t_samples = max(t_samples, 10)               # mínimo absoluto
```

---

## 4. Stage 2 — Dinámica: Parámetros por Variante

Cada variante de Etapa 2 se instancia con las claves de `stage2_params`. La resolución del rango de trabajo (número de modos D de la salida) sigue la regla: `n_components >= n_dim` siempre, para que la Etapa 3 pueda seleccionar.

### 4.1 `"pca"` (PCADynamics)

PCA truncada (SVD) de la matriz de entrada. Devuelve los *scores* temporales de los componentes principales `Σ Vᵀ`.

**Parámetros de `stage2_params`:**

| Clave | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `n_components` | `int \| None` | `50` | Número de componentes principales retenidos. `None` → `min(50, max_allowed)` donde `max_allowed = min(m, n) - 1` de la matriz de entrada. Debe ser `>= n_dim`. |

- **Sin Hankel**: SVD de `X` (canales × tiempo) → `scores` shape `(n_components, N_t)`.
- **Con Hankel**: SVD de `H` (Hankel) → `scores` shape `(n_components, N_t - depth + 1)`.
- **Meta**: incluye `singular_values` (lista de valores singulares descendentes).
- **Convención de signo**: se aplica `svd_flip` (el loading de mayor magnitud en cada vector singular izquierdo se fuerza positivo) para garantizar determinismo *run-to-run*.

### 4.2 `"pca_ica"` (PCAICADynamics)

Descomposición ICA con dos ramas mutuamente excluyentes dependiendo de si hay o no Hankel.

**Parámetros de `stage2_params`:**

| Clave | Tipo | Por Defecto | Rama | Descripción |
|---|---|---|---|---|
| `n_components` | `int \| float \| None` | `None` | Ambas | **Sin Hankel**: número de componentes ICA de MNE. `None` = todos los canales − 1. Un `float` en (0,1] indica fracción de varianza (solo soportado en la rama MNE). **Con Hankel**: rango SVD antes de FastICA (entero, default 50). Debe ser `>= n_dim`. |
| `ica_method` | `str` | `"picard"` | Sin Hankel | Algoritmo ICA de MNE: `"picard"`, `"fastica"`, `"infomax"`. No se usa en la rama Hankel. |
| `ica_random_state` | `int \| None` | `42` | Ambas | Semilla aleatoria. Para la rama Hankel se pasa a `sklearn.decomposition.FastICA`. |
| `retained_labels` | `list[str] \| None` | `None` | Sin Hankel | Etiquetas ICLabel a conservar (ej. `["brain"]`). Solo aplicable a la rama MNE-ICA. |
| `ica_solver` | `str` | `"fastica"` | Con Hankel | Solver para la rama Hankel. Actualmente solo se acepta `"fastica"` (sklearn). Cualquier otro valor lanza `ValueError`. |

#### Rama A — Sin Hankel (MNE ICA + ICLabel)

- **Requiere**: `ctx.raw` debe ser un objeto `mne.Raw` real (MNE ICA necesita canales con nombres y posiciones). Si se pasa una matriz numérica, lanza `ValueError`.
- **Flujo**: llama a `run_full_preprocessing(raw, ...)` internamente, que realiza: filtrado *band-pass* → ICA de MNE → clasificación ICLabel → rechazo de artefactos →矩阵 `Y` (componentes limpios, media cero).
- **Meta**: incluye `branch="mne_ica_iclabel"`, `excluded_indices`, `kept_indices`, `labels`, y el diccionario completo de preprocesamiento legacy.

#### Rama B — Con Hankel (SVD + whitening + sklearn FastICA)

- **No usa MNE ICA ni ICLabel** (los canales virtuales de retardo no tienen topografía en el cuero cabelludo).
- **Flujo**: (a) SVD truncada de H → (b) *whitening*: estandarización de los *scores* PCA a varianza unitaria por fila (`Z = (Σ Vᵀ) / std(axis=1)`) → (c) `sklearn.decomposition.FastICA` sobre las PCs *whitened* (transpuestas a formato sklearn: `(n_samples, n_features)`).
- **FastICA config interno**: `whiten="unit-variance"`, `max_iter=1000`, `tol=1e-4`.
- **Meta**: incluye `branch="svd_whitening_fastica_sklearn"`, `svd_rank`, `singular_values`, `n_iter_` (iteraciones convergencia).

### 4.3 `"dmd"` (DMDDynamics)

Dynamic Mode Decomposition con/sin el *embedding* Hankel.

**Parámetros de `stage2_params`:**

| Clave | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `rank` | `int \| None` | `10` | Número de modos DMD/POD retenidos (D de la salida Etapa 2). `None` → 10. Para reproducir exactamente el comportamiento legacy `hankel_dmd`, el orquestador mapea `rank = n_dim`. Debe ser `>= n_dim`. |

#### Rama A — Con Hankel (Empate 4.1)

- Llama directamente a `eeg_hankel_dmd_core()` sobre la matriz de canales original (almacenada en `ctx.stage1_meta["input_data"]`).
- **Preserva la numerics legacy exactamente**: referencia promedio + eliminación del último canal ocurren *dentro* de `eeg_hankel_dmd_core`, no en la Etapa 1.
- **Flujo interno de `eeg_hankel_dmd_core(V, T, P, dt)`**:
  1. Referencia promedio sobre todos los electrodos: `V = V - mean(V, axis=0)`.
  2. Descartar el último canal: `X = V[:-1, :]` (evita déficit de rango).
  3. Construir la matriz Hankel `H` de shape `(Ne * T, Nt - T + 1)`.
  4. SVD truncada de `H` a `P` componentes (usando `scipy.sparse.linalg.svds`).
  5. Operador DMD reducido en la base POD: `A = (Lᵀ H₂ R₁) diag(1/s)`, donde `H₁ = H[:, :-1]`, `H₂ = H[:, 1:]`.
  6. Descomposición espectral: eigenvalores discretos `λ`, continuos `ω = log(λ)/dt`, frecuencias `f = Im(ω)/(2π)`, tasas de amortiguamiento `d = Re(ω)`.
  7. *Scores* latentes: `Lᵀ H` de shape `(P, Nt-T+1)`.
- **Modos ordenados por** `|frecuencia|` ascendente (modos lentos dominantes primero).
- **Meta**: incluye `frequencies_hz`, `damping_rates`, `eigenvalues`, `continuous_eigenvalues`, `singular_values`, `hankel_shape`.

#### Rama B — Sin Hankel (Empate 4.2, DMD directo sobre X)

- DMD directo sobre la matriz de canales `X`. Matemáticamente es un modelo autorregresivo de orden 1 en el espacio de canales ("PCA dinámico").
- **Flujo**: `X₁ = X[:, :-1]`, `X₂ = X[:, 1:]`. Operador de propagación reducido: `A = X₂ X₁⁺` (pseudoinversa, shape `N_c × N_c`). Descomposición espectral de `A` → coordenadas modales.
- **Limitación**: **No** captura memoria temporal ni atractores no lineales (a diferencia de la rama Hankel).
- **Meta**: incluye `branch="direct_ar1_channel_space"`, `note` (advertencia sobre las limitaciones), `frequencies_hz`, `damping_rates`, `eigenvalues`.

### 4.4 `"diffusion_maps"` (DiffusionMapsDynamics)

Diffusion Maps (Coifman & Lafon 2006) vía el paquete `pyDiffMap`.

**Parámetros de `stage2_params`:**

| Clave | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `n_components` | `int \| None` | `10` | Número de coordenadas de difusión no triviales retenidas (D de la salida Etapa 2). pyDiffMap ya excluye el autovalor trivial λ₀ = 1. Debe ser `>= n_dim` y `< n_samples - 1`. |
| `svd_rank` | `int \| None` | `50` | **Solo rama Hankel (NLSA)**: rango de truncación SVD de H antes de aplicar DM. El grafo de DM se construye sobre los puntos `V_r Σ_r` en `R^{svd_rank}`. Ignorado en la rama sin Hankel. |
| `sigma` | `float \| None` | `None` | *Bandwidth* del kernel Gaussiano. `None` → estimación automática vía el método Berry-Giannakis-Harlim (BGH) implementado en pyDiffMap. Relación con epsilon de pyDiffMap: `epsilon = sigma² / 2`. |
| `k` | `int` | `100` | Número de vecinos más cercanos para la matriz de afinidad dispersa. Si `n_samples < k + 1`, se reduce automáticamente a `n_samples - 1`. |
| `diffusion_time` | `float` | `0.0` | Tiempo de difusión `t >= 0`. Aplica filtrado multiescala: `Ψ_t(x) = [λ₁ᵗ φ₁(x), ...]`. `t = 0` = autovectores puros (sin filtrado). Valores mayores atenúan componentes de pequeña escala y resaltan dinámicas macroscópicas. |
| `alpha` | `float` | `0.5` | Parámetro de normalización de densidad (Coifman-Lafon): `0.0` = Laplacian Eigenmaps, `0.5` = Diffusion Maps clásico (por defecto), `1.0` = Fokker-Planck. Controla cómo se compensa la densidad de muestreo no uniforme. |

#### Rama A — Sin Hankel (Empate 3)

- La DM opera sobre **instantes temporales**: cada columna de `X` es un punto en `R^{N_c}`. La dimensionalidad es baja (~50 canales), así que k-NN es eficiente sin pre-PCA.
- **Meta**: `branch="dm_over_time_instants"`.

#### Rama B — Con Hankel (Empate 2, NLSA)

- La matriz Hankel se trunca primero vía SVD a `svd_rank` componentes. La DM se construye sobre los puntos `V_r Σ_r` en `R^{svd_rank}` (Análisis Espectral de Laplaciano No Lineal = NLSA: SSA lineal sobre H + DM no lineal sobre los modos principales).
- **Meta**: `branch="nlsa_svd_then_dm"`, incluye `svd_rank` y `singular_values` del paso SVD previo.

#### Meta adicional de Diffusion Maps (ambas ramas)

| Clave Meta | Descripción |
|---|---|
| `sigma_used` | Valor σ efectivamente usado (estimado o manual). |
| `epsilon_fitted` | Valor ε ajustado por pyDiffMap (solo cuando `sigma=None`). |
| `k_neighbors` | Número de vecinos usados. |
| `eigenvalues` | Todos los autovalores no triviales (ordenados descendente). |
| `eigenvalues_latent` | Solo los `n_components` primeros autovalores. |
| `spectral_gap` | `|λ₁| - |λ₂|` (brecha espectral; mayor = separación más clara). |

#### Advertencias de Diffusion Maps

- Si `|λ₁| > 0.999`, se emite un `UserWarning` indicando que el grafo puede estar sobre-conectado (sigma demasiado grande o k demasiado pequeño).
- Si sigma estimado = 0 (datos duplicados/constantes), se usa `epsilon = 0.1` como *fallback*.

---

## 5. Stage 3 — Selección: Parámetros por Variante

La Etapa 3 recibe la matriz `Y` de shape `(D, T')` de la Etapa 2 (filas = modos candidatos, columnas = tiempo) y selecciona `n_dim` filas.

### 5.1 `"top_n"` (TopNSelection)

Selección simple por ranking: mantiene las primeras `n_dim` filas de `Y`. Los modos ya vienen ordenados por relevancia de la Etapa 2 (valor singular, frecuencia, autovalor de difusión, etc.).

**Parámetros de `stage3_params`:**

- **Ninguno**. Si se pasa algún parámetro, se lanza `ValueError` con el mensaje `"takes no parameters"`.

- **Resultado**: `selected_indices = [0, 1, ..., n_dim-1]`.
- **`meta["stage3"]["scores"]`**: diccionario vacío `{}`.

### 5.2 `"markov_fastest"` / `"markov_slowest"` (MarkovSelection)

Selección de subespacio basada en el **tiempo de relajación de Markov** τ. El procedimiento es:

1. Discretizar cada fila de `Y` en `n_bins` *bins* de cuantiles.
2. Enumerar estados conjuntos vía indexación de radices mixtas.
3. Estimar la matriz de transición de Markov contando transiciones.
4. Calcular la brecha espectral y el tiempo de relajación: `τ = -1 / ln(|λ₂|)`.
5. Buscar (exhaustiva o greedy) el subespacio de `n_dim` componentes con el τ óptimo.

**Parámetros de `stage3_params`:**

| Clave | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `n_bins` | `int` | `5` | Número de *bins* de cuantiles por componente. Más *bins* = mayor resolución de la dinámica, pero más estados conjuntos (`n_bins^n_dim`) y mayor riesgo de dispersión. Valores típicos: 4-10. |
| `search_strategy` | `"exhaustive" \| "greedy"` | `"exhaustive"` | `"exhaustive"` evalúa todas las combinaciones `C(D, n_dim)`. `"greedy"` selecciona iterativamente el componente que mejora más el criterio. **Importante**: `"greedy"` se fuerza automáticamente cuando `n_dim >= 4` (por costo combinatorio). |

**Diferencia entre `markov_fastest` y `markov_slowest`:**

- `markov_fastest`: `maximize=False` → busca τ **mínimo** (dinámicas más rápidas, exploración rica del espacio de estados).
- `markov_slowest`: `maximize=True` → busca τ **máximo** (dinámicas más lentas, subespacio más estable).

**Meta**: incluye `scores.tau` (valor óptimo de τ), `scores.n_bins`, `scores.search` ("greedy" o "exhaustive"), `scores.maximize` (booleano).

**Detalles de la implementación de Markov:**

| Función | Descripción |
|---|---|
| `discretize_series(data, n_bins)` | Discretiza cada fila en `n_bins` *bins* de cuantiles. Los bordes son los percentiles empíricos. |
| `compute_markov_time(state_seq, n_states)` | Estima τ desde una secuencia de estados discretos. Si la cadena es no-ergódica (estados no visitados) o periódica (`|λ₂| >= 1 - tol`), devuelve `inf`. |
| `evaluate_markov_combination(comb, bins_idx, n_bins)` | Calcula τ para una combinación de componentes. Usa indexación de radices mixtas para el estado conjunto. |
| `find_best_subspace_markov(data, N, ...)` | Búsqueda exhaustiva sobre todas las `C(D, N)` combinaciones. Soporta paralelización vía `n_workers`. |
| `greedy_forward_selection_markov(data, N, ...)` | Selección greedy: en cada paso añade el componente que minimiza (o maximiza) τ. |

### 5.3 `"legacy_fc"` (LegacyFCSelection)

**Shim de compatibilidad** — no es parte de la API nueva. Solo existe para que el CLI legacy siga funcionando. Se usa internamente cuando se pasa `scoring_method` con los valores `"conservative"`, `"weighted"`, `"sequential"`, `"pareto"` o `"independent"`.

**Parámetros de `stage3_params`:**

| Clave | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `method` | `str` | *(requerido)* | Uno de: `"conservative"`, `"weighted"`, `"sequential"`, `"pareto"`, `"independent"`. |
| `fc_metric` | `str` | `"variance_sum"` | Métrica de varianza: `"variance_sum"`, `"first_pc_var"`, `"total_variance"`. |
| `alpha` | `float \| None` | `None` | Requerido para `method="weighted"`. Trade-off en `[0, 1]`. |
| `primary_criterion` | `str` | `"markov"` | Para `method="sequential"`: `"fc"` o `"markov"`. |
| `sequential_K` | `int \| None` | `None` | Para `method="sequential"`: candidatos a retener. |
| `n_bins` | `int` | `5` | *Bins* de cuantiles Markov. |
| `search_strategy` | `str` | `"exhaustive"` | `"exhaustive"` o `"greedy"`. |

#### Descripción de los 5 métodos legacy

| Método | Descripción |
|---|---|
| `conservative` | Subespacio con la máxima *conservative fraction* (fracción de varianza conservada proyectada al subespacio). Búsqueda exhaustiva o greedy. |
| `weighted` | Combinación convexa normalizada: `S_α = α·f̃c + (1-α)·(1/τ)̃`. Se evalúa una grilla de α. Requiere `alpha`. |
| `sequential` | Filtra top-K bajo un criterio primario, luego reordena bajo el secundario. Soporta `primary_criterion` y `K`. |
| `pareto` | Frente de Pareto en el plano `(f_c, 1/τ)`: subespacios no dominados. Devuelve todos los de la frontera. |
| `independent` | Ejecuta FC y Markov de forma independiente y devuelve ambos resultados (combinación y valor óptimos para cada uno). |

---

## 6. Tabla de Combinaciones Válidas

La siguiente tabla resume el comportamiento de cada combinación de Etapa 2 × (con/sin Hankel):

| Etapa 2 | Sin Hankel (Stage 1 = None) | Con Hankel (Stage 1 = "hankel") |
|---|---|---|
| `"pca"` | SVD truncada de `X` (N_c × N_t) | SVD truncada de `H` |
| `"pca_ica"` | MNE ICA sobre canales reales + ICLabel | SVD de H → whitening → sklearn FastICA |
| `"dmd"` | DMD directo sobre X (AR-1 espacial) | DMD sobre H (pipeline legacy completo) |
| `"diffusion_maps"` | DM sobre instantes temporales de X (puntos en R^{N_c}) | NLSA: SVD de H → DM sobre V_r Σ_r (puntos en R^{svd_rank}) |

Y para la Etapa 3:

| Etapa 3 | Requiere parámetros | Notas |
|---|---|---|
| `"top_n"` | No | Los modos ya están ordenados por la Etapa 2. |
| `"markov_fastest"` | Sí: `n_bins`, `search_strategy` | Búsqueda combinatoria; costo crece como C(D, n_dim). |
| `"markov_slowest"` | Sí: `n_bins`, `search_strategy` | Igual que anterior pero maximiza τ. |
| `"legacy_fc"` | Sí: `method`, `fc_metric`, etc. | Solo accesible vía `scoring_method` legacy. |

### Combinaciones que lanzan `ValueError`

| Combinación inválida | Mensaje de error |
|---|---|
| `stage1_embedding="wavelet"` (o cualquier valor no registrado) | `"Unknown embedding ..."` |
| `stage2_dynamics="emd"` (o cualquier valor no registrado) | `"Unknown dynamics ..."` |
| `stage3_selection="random"` (o cualquier valor no registrado) | `"Unknown selection ..."` |
| `stage1_embedding=None` con `stage1_params` no vacío | `"takes no parameters"` |
| `stage1_embedding="hankel"` con `depth >= n_times` | `"embedding depth ... must be < n_times"` |
| `stage2_dynamics="pca"` con `n_components < n_dim` | `"must be >= n_dim"` |
| `stage2_dynamics="pca_ica"` + Hankel con `ica_solver="picard"` | `"ica_solver=... not supported in the Hankel branch"` |
| `stage2_dynamics="pca_ica"` sin Hankel con `raw=None` | `"requires an mne.Raw object"` |

---

## 7. Mapeo Legacy → Nueva API

La siguiente tabla muestra cómo cada `scoring_method` legacy se mapea a la nueva API de 3 etapas:

| `scoring_method` | Stage 1 | Stage 2 | Stage 3 | Notas |
|---|---|---|---|---|
| `"markov"` | `None` | `"pca_ica"` | `"markov_fastest"` | `n_bins` y `search_strategy` se pasan a Stage 3. |
| `"markov_inverted"` | `None` | `"pca_ica"` | `"markov_slowest"` | Invierte la optimización de Markov. |
| `"hankel_dmd"` | `"hankel"` | `"dmd"` | `"top_n"` | `rank = n_dim` (reproduce numerics legacy). `hankel_embedding_depth` se pasa a Stage 1. |
| `"diffusion_maps"` | `None` | `"diffusion_maps"` | `"top_n"` | `n_components = n_dim`. `diffusion_sigma/k/time/alpha` se pasan a Stage 2. |
| `"conservative"` | `None` | `"pca_ica"` | `"legacy_fc"` | `method="conservative"` en Stage 3. |
| `"weighted"` | `None` | `"pca_ica"` | `"legacy_fc"` | `method="weighted"`, requiere `alpha`. |
| `"sequential"` | `None` | `"pca_ica"` | `"legacy_fc"` | `method="sequential"`. |
| `"pareto"` | `None` | `"pca_ica"` | `"legacy_fc"` | `method="pareto"`. |
| `"independent"` | `None` | `"pca_ica"` | `"legacy_fc"` | `method="independent"`. |

---

## 8. Parámetros del Script CLI

El script `test_iga_from_eeg_latent_test_retest_gedai.py` expone todos los parámetros del pipeline vía argumentos de línea de comandos, además de parámetros propios del flujo KM+IgA. Los argumentos de la nueva API tienen prioridad sobre los legacy.

### 8.1 Fuente EEG

| Argumento | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `--subject` | `str` | *(requerido)* | ID del sujeto (ej. `sub-01`). |
| `--session` | `str` | *(requerido)* | ID de sesión (ej. `session1`). |
| `--task` | `str` | *(requerido)* | Etiqueta de tarea: `eyesclosed`, `eyesopen`, `mathematic`, `memory`, `music`. |
| `--db-path` | `str` | `None` | Ruta raíz del dataset GEDAI (sobrescribe la config por defecto). |
| `--t-start` | `float` | `0.0` | Tiempo de inicio del segmento EEG (segundos). |
| `--t-end` | `float` | `60.0` | Tiempo de fin del segmento EEG (segundos). |

### 8.2 Pipeline de 3 Etapas (nueva API)

| Argumento | Tipo | Choices | Descripción |
|---|---|---|---|
| `--stage1-embedding` | `str` | `none`, `hankel` | Etapa 1. Si se omite junto con los demás `--stageX`, se usa el mapeo legacy. |
| `--stage1-params` | `str` (JSON) | — | Parámetros de Etapa 1 como JSON, ej. `{"depth": 250}`. |
| `--stage2-dynamics` | `str` | `pca_ica`, `pca`, `dmd`, `diffusion_maps` | Etapa 2. |
| `--stage2-params` | `str` (JSON) | — | Parámetros de Etapa 2 como JSON, ej. `{"svd_rank": 50, "k": 100}`. |
| `--stage3-selection` | `str` | `top_n`, `markov_fastest`, `markov_slowest` | Etapa 3. |
| `--stage3-params` | `str` (JSON) | — | Parámetros de Etapa 3 como JSON, ej. `{"n_bins": 10}`. |

**Prioridad**: si se pasa **cualquier** argumento `--stageX`, se usa la nueva API directamente. Los *flags* de conveniencia como `--diffusion-sigma`, `--hankel-embedding-depth` o `--n-bins` se inyectan en los JSON params cuando no están ya presentes.

### 8.3 API Legacy

| Argumento | Tipo | Por Defecto | Choices | Descripción |
|---|---|---|---|---|
| `--scoring-method` | `str` | `"hankel_dmd"` | `markov`, `markov_inverted`, `conservative`, `weighted`, `sequential`, `pareto`, `independent`, `hankel_dmd`, `diffusion_maps` | Método legacy. Ignorado si se pasa algún `--stageX`. |
| `--fc-metric` | `str` | `"variance_sum"` | `variance_sum`, `first_pc_var`, `total_variance` | Métrica de varianza para conservative fraction. |
| `--n-bins` | `int` | `10` | — | Bins de cuantiles Markov. |
| `--workers` | `int` | `None` | — | Procesos paralelos. |
| `--search-strategy` | `str` | `"exhaustive"` | `exhaustive`, `greedy` | Estrategia de búsqueda combinatoria. |

### 8.4 Parámetros de Preprocesamiento y Configuración

| Argumento | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `--latent-dim` | `int` | `2` | Dimensión del subespacio latente (equivalente a `n_dim`). |
| `--l-freq` | `float` | `1.0` | Corte inferior del band-pass (Hz). |
| `--h-freq` | `float` | `40.0` | Corte superior del band-pass (Hz). |
| `--ica-method` | `str` | `"picard"` | Algoritmo ICA (solo rama MNE). |
| `--verbose` | `flag` | `True` | Activar verbosidad. |
| `--stage1-params` | `str` (JSON) | — | Parámetros de Etapa 1 como JSON, ej. `{"depth": 250}`. |
| `--diffusion-sigma` | `float` | `None` | Bandwidth DM (también inyectable vía `--stage2-params`). |
| `--diffusion-k` | `int` | `100` | Vecinos DM. |
| `--diffusion-time` | `float` | `0.0` | Tiempo de difusión. |
| `--diffusion-alpha` | `float` | `0.5` | Normalización de densidad DM. |

### 8.5 Parámetros de Caché y Salida

| Argumento | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `--cache-file` | `str` | `None` | Ruta al fichero de caché `.npz`. Si `None`, se auto-genera. |
| `--ignore-cache` | `flag` | `False` | Forzar recompute aunque exista caché. |
| `--out-dir` | `str` | `None` | Directorio de salida para plots y resultados. |
| `--save-potential` | `flag` | `True` | Guardar datos del potencial en `.npz`. |
| `--no-save-potential` | `flag` | `False` | Deshabilitar guardado del potencial. |

### 8.6 Parámetros de Análisis KM/IgA (posteriores al pipeline)

| Argumento | Tipo | Por Defecto | Descripción |
|---|---|---|---|
| `--analysis-dim` | `int` | `None` | Dimensiones latentes para Kramers-Moyal. `None` = todas. |
| `--column` | `int` | `None` | Si `analysis-dim=1`, qué columna latente usar (0-based). |

---

## 9. Estructura del Diccionario `meta` Devuelto

La función `extract_latent_space()` devuelve `(latent, meta)`. El diccionario `meta` contiene:

```python
meta = {
    # --- Traza del pipeline (nueva API) ---
    "pipeline": {
        "stage1": None | "hankel",
        "stage2": "pca_ica" | "pca" | ...,
        "stage3": "top_n" | "markov_fastest" | ...,
        "params": {
            "stage1_params": { ... },
            "stage2_params": { ... },
            "stage3_params": { ... },
        },
        "chain": "none+pca_ica+markov_fastest",
    },
    # --- Metadata por etapa ---
    "stage1": { ... },
    "stage2": { ... },
    "stage3": { ... },
    # --- Claves legacy ---
    "selected_indices": [0, 1],
    "scoring_method": "hankel_dmd",
    "latent_scores": { ... },
    "preprocessing": { ... },
    "Y": np.ndarray,
    "Y_shape": (D, T'),
    "elapsed_time": 3.45,
}
```

### 9.1 Claves en `meta["latent_scores"]`

| Clave | Etapa 2 | Descripción |
|---|---|---|
| `frequencies_hz` | `dmd` | Frecuencias oscilatorias DMD [Hz]. |
| `damping_rates` | `dmd` | Tasas de amortiguamiento [1/s]. |
| `eigenvalues` | `dmd` | Autovalores discretos del operador DMD. |
| `continuous_eigenvalues` | `dmd` (Hankel) | Autovalores continuos `w = log(lambda)/dt`. |
| `singular_values` | `pca`, `pca_ica` (Hankel), `dmd` (Hankel) | Valores singulares. |
| `sigma_used` | `diffusion_maps` | Bandwidth del kernel. |
| `epsilon_fitted` | `diffusion_maps` | Epsilon ajustado por pyDiffMap. |
| `k_neighbors` | `diffusion_maps` | Vecinos usados. |
| `diffusion_time` | `diffusion_maps` | Tiempo de difusión. |
| `alpha` | `diffusion_maps` | Normalización de densidad. |
| `eigenvalues` | `diffusion_maps` | Autovalores de difusión. |
| `eigenvalues_latent` | `diffusion_maps` | Autovalores de coordenadas latentes. |
| `spectral_gap` | `diffusion_maps` | Brecha espectral. |
| `tau` | `markov_fastest` / `markov_slowest` | Tiempo de relajación Markov optimo. |

### 9.2 Claves en `meta["preprocessing"]`

| Clave | Tipo | Descripción |
|---|---|---|
| `l_freq` | `float` | Corte inferior del filtro [Hz]. |
| `h_freq` | `float` | Corte superior del filtro [Hz]. |
| `sfreq` | `float` | Frecuencia de muestreo [Hz]. |
| `dt` | `float` | Intervalo de muestreo [s]. |
| `D` | `int` | Modos de la Etapa 2 (filas de Y). |
| `T` | `int` | Muestras temporales de Y (columnas). |
| `n_samples_latent` | `int` | Muestras del espacio latente final. |
| `n_time_lost_by_embedding` | `int` | Muestras perdidas por embedding (0 o `depth-1`). |
| `n_channels` | `int` | Canales EEG de entrada. |
| `n_times` | `int` | Muestras temporales originales. |
| `embedding_depth` | `int` | Profundidad Hankel (solo con Hankel). |
| `hankel_shape` | `tuple` | Shape de H (solo con Hankel). |
| `singular_values` | `list` | Valores singulares (cuando disponible). |
| `excluded_indices` | `list` | Componentes excluidos (solo MNE-ICA). |
| `kept_indices` | `list` | Componentes conservados. |
| `X_filtered` | `ndarray` | Matriz filtrada (solo si no es MNE-ICA). |

---

## 10. Funciones Auxiliares del Orquestador

### 10.1 `map_legacy_scoring_method(scoring_method, *, ...)`

Mapea un `scoring_method` legacy a la especificacion de 3 etapas. Devuelve un `dict` con las claves: `stage1_embedding`, `stage1_params`, `stage2_dynamics`, `stage2_params`, `stage3_selection`, `stage3_params`. Util para inspeccionar a que cadena se mapea un metodo legacy sin ejecutar el pipeline completo.

### 10.2 `sample_conservative_fraction(Y, n_dim, *, n_samples=500, metric="variance_sum", n_workers=None)`

Diagnostico legacy: evalua la conservative fraction en las primeras `n_samples` combinaciones de `n_dim` componentes. Devuelve estadisticas (min, max, mean, std, CV, `degenerate`). Util para evaluar si la fraccion conservadora discrimina entre subespacios.

### 10.3 `diagnose_subspace_discrimination(Y, n_dim=2, *, fc_metrics=None, n_bins=5, n_workers=None, cv_threshold=0.05)`

Diagnostico completo: evalua exhaustivamente todas las combinaciones bajo multiples metricas de FC y Markov, y emite una recomendacion de metodo (`"markov"`, `"conservative"`, `"weighted"`). Devuelve resultados detallados y la recomendacion con justificacion.

---

## Apéndice: Constantes por Defecto

| Constante | Valor | Definida en | Usada por |
|---|---|---|---|
| `DEFAULT_L_FREQ` | `1.0` | `extract_latent_subspace.py` | Corte inferior del filtro. |
| `DEFAULT_H_FREQ` | `40.0` | `extract_latent_subspace.py` | Corte superior del filtro. |
| `DEFAULT_ICA_METHOD` | `"picard"` | `extract_latent_subspace.py` | Algoritmo ICA MNE. |
| `DEFAULT_ICA_RANDOM_STATE` | `42` | `extract_latent_subspace.py` | Semilla ICA. |
| `DEFAULT_N_BINS` | `5` | `extract_latent_subspace.py` | Bins Markov. |
| `DEFAULT_FC_METRIC` | `"variance_sum"` | `extract_latent_subspace.py` | Metrica conservative fraction. |
| `DEFAULT_EMBEDDING_DEPTH` | `60` | `hankel_dmd_extractor.py` | Profundidad Hankel (solo extractor standalone). |

---
*AI生成*

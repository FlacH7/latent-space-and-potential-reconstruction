"""Plots de influencia de canales sobre un grafo estructural (no-EEG).

Para datasets sin montaje topográfico (p. ej. Ludovico_01, series
temporales multivariadas sobre una red de nodos), la influencia de los
canales originales sobre cada dimensión latente se puede visualizar
sobre el **grafo estructural** que genera los datos, en lugar de sobre
una cabeza (``mne`` topoplot) o barras horizontales:

- Nodos = canales; aristas = conexiones estructurales (matriz de
  adyacencia binaria, p. ej. ``structural.csv`` de Ludovico_01).
- Color del nodo = peso de influencia en ``[-1, 1]`` (azul = positivo,
  rojo = negativo, blanco ≈ 0; colormap ``RdBu``).
- Orden de superposición por ``|peso|``: en las zonas donde los nodos
  se solapan (núcleo denso), los de influencia más fuerte se dibujan
  al final y quedan **encima** — un nodo relevante nunca queda
  enterrado debajo de nodos casi-blancos.
- Un panel por dimensión latente (LD1, LD2, ...), cada uno con su
  colorbar, en la misma disposición que ``plot_topomap_grid``.

Este módulo es un plotter **puro**: no lee configuración ni rutas del
proyecto. La resolución del path de la matriz estructural
(``DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH``) la hace quien lo invoca
(``src.spectral_analysis.psd_analysis``).

Dependencias: numpy + matplotlib. ``networkx`` es opcional: si está
instalado se usa para los layouts (spring/kamada_kawai/...); si no, se
usa un layout espectral calculado con numpy (autovectores 2.º y 3.º del
Laplaciano del grafo), determinista y sin dependencias extra.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "compute_graph_layout",
    "load_structural_matrix",
    "plot_structural_graph_influence",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carga y validación de la matriz estructural
# ---------------------------------------------------------------------------

def load_structural_matrix(path: str | Path) -> np.ndarray:
    """Carga la matriz de adyacencia estructural desde un CSV sin cabecera.

    El fichero es una matriz cuadrada de 0s y 1s (filas y columnas =
    nodos/canales, en el mismo orden que los canales de los datos).
    Se binariza por seguridad (``!= 0``), se simetriza
    (``max(A, A.T)``) y se fuerza diagonal cero (sin auto-lazos).

    Parameters
    ----------
    path:
        Ruta al CSV (p. ej. ``DB_LUDOVICO_01_STRUCTURAL_MATRIX_PATH``).

    Returns
    -------
    np.ndarray
        Matriz de adyacencia ``(n_nodes, n_nodes)`` binaria, simétrica
        y con diagonal cero.

    Raises
    ------
    ValueError
        Si la matriz no es cuadrada.
    """
    path = Path(path)
    adjacency = np.loadtxt(path, delimiter=",", dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            f"La matriz estructural debe ser cuadrada; "
            f"recibida shape={adjacency.shape} ({path})"
        )
    adjacency = (adjacency != 0).astype(float)
    if not np.array_equal(adjacency, adjacency.T):
        logger.info(
            "La matriz estructural %s no es simétrica; se simetriza con "
            "max(A, A.T).", path.name,
        )
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


# ---------------------------------------------------------------------------
# Layout del grafo
# ---------------------------------------------------------------------------

def compute_graph_layout(
    adjacency: np.ndarray,
    layout: str = "spring",
    seed: int = 42,
    k: float | None = None,
) -> np.ndarray:
    """Calcula posiciones 2D de los nodos del grafo.

    Con ``networkx`` instalado soporta ``"spring"`` (Fruchterman-
    Reingold, determinista vía ``seed``), ``"kamada_kawai"``,
    ``"spectral"`` y ``"circular"``. Sin ``networkx``, ``"circular"``
    se calcula igualmente con numpy puro (nodos equiespaciados sobre la
    circunferencia unidad, en orden de índice — la misma convención de
    ``nx.circular_layout``); cualquier otro valor de ``layout`` cae en
    un layout espectral puro-numpy (autovectores asociados al 2.º y 3.º
    autovalor del Laplaciano), también determinista.

    Parameters
    ----------
    adjacency:
        Matriz de adyacencia ``(n, n)``.
    layout:
        Algoritmo de layout (ver arriba). Con ``"circular"`` los nodos
        se colocan sobre una circunferencia y se dibujan **todas** las
        aristas de la matriz (no se poda ninguna), lo que es útil para
        grafos densos donde los layouts de fuerzas solapan nodos.
    seed:
        Semilla para layouts estocásticos (``spring``).
    k:
        Distancia óptima entre nodos para ``spring``. ``None`` usa
        ``3.5 / sqrt(n)``, más abierto que el default de networkx para
        grafos densos como el de Ludovico_01 (grado medio ~23), de modo
        que las etiquetas internas de los nodos queden legibles.

    Returns
    -------
    np.ndarray
        Posiciones ``(n, 2)``.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    try:
        import networkx as nx
    except ImportError:
        nx = None

    if nx is not None:
        graph = nx.from_numpy_array(adjacency)
        if layout == "spring":
            pos = nx.spring_layout(graph, seed=seed, k=k if k is not None else 3.5 / np.sqrt(n))
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        else:
            raise ValueError(
                f"Layout desconocido: {layout!r}. "
                "Opciones: 'spring', 'kamada_kawai', 'spectral', 'circular'."
            )
        return np.array([pos[i] for i in range(n)], dtype=float)

    if layout == "circular":
        # Fallback numpy para "circular": nodos equiespaciados sobre la
        # circunferencia unidad, en orden de índice (0, 1, ..., n-1),
        # empezando en el eje +x y en sentido antihorario — la misma
        # convención que nx.circular_layout. Determinista y sin
        # dependencias.
        angles = 2.0 * np.pi * np.arange(n) / max(n, 1)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    logger.info(
        "networkx no está instalado; se usa layout espectral (numpy) "
        "en lugar de %r.", layout,
    )
    degree = adjacency.sum(axis=1)
    laplacian = np.diag(degree) - adjacency
    _, eigvecs = np.linalg.eigh(laplacian)
    if n >= 3:
        coords = eigvecs[:, 1:3]
    else:  # grafos triviales de 1-2 nodos
        coords = np.hstack([eigvecs[:, 1:2], np.zeros((n, 1))])
    return np.asarray(coords, dtype=float)


# ---------------------------------------------------------------------------
# Figura principal
# ---------------------------------------------------------------------------

def plot_structural_graph_influence(
    weights: np.ndarray,
    adjacency: np.ndarray,
    dim_labels: list[str] | None = None,
    node_labels: list[str] | None = None,
    show_node_labels: bool = True,
    node_label_fontsize: float = 7.0,
    sort_by_abs_weight: bool = True,
    layout: str = "spring",
    seed: int = 42,
    layout_k: float | None = None,
    cmap: str = "RdBu",
    vlim: tuple[float, float] = (-1.0, 1.0),
    title: str = "Channel influence on latent space (structural graph)",
    node_size: float = 300.0,
    edge_color: str = "0.6",
    edge_alpha: float = 0.35,
    edge_width: float = 0.5,
    figsize_per_dim: tuple[float, float] = (5.2, 5.4),
    max_cols: int = 4,
) -> plt.Figure:
    """Influencia de canales por dimensión latente sobre el grafo estructural.

    Un panel por dimensión latente: nodos coloreados por el peso de
    influencia (azul = positivo, rojo = negativo, intensidad ∝ |valor|,
    escala fija ``vlim`` compartida) y aristas estructurales en gris
    tenue debajo. La disposición de paneles (etiqueta ``LD{d}`` +
    colorbar por panel) replica el estilo de los topoplots EEG.

    Parameters
    ----------
    weights:
        Pesos de influencia ``(n_channels, n_dim)``, ya normalizados a
        ``[-1, 1]`` (signo preservado), tal como los devuelve
        ``compute_channel_influence_on_latent``. La fila ``i`` se
        corresponde con el nodo/canal ``i`` de la matriz estructural.
    adjacency:
        Matriz de adyacencia estructural ``(n_channels, n_channels)``.
    dim_labels:
        Etiquetas de cada dimensión (default: ``LD1..LDn``).
    node_labels:
        Etiquetas de los nodos. Si es ``None`` se usa el índice del
        nodo (``0..n-1``): el nodo ``i`` se corresponde con la
        fila/columna ``i`` de la matriz estructural y con el canal
        ``i`` de los datos. Solo se dibujan si ``show_node_labels=True``.
    show_node_labels:
        Dibujar la etiqueta **centrada dentro** de cada nodo (el color
        de texto —blanco o negro— se elige por la luminancia del nodo
        para mantener el contraste con cualquier valor de influencia).
    node_label_fontsize:
        Tamaño de fuente de las etiquetas internas. Con el
        ``node_size`` por defecto (300) caben 1–2 dígitos; si reduces
        ``node_size``, reduce este valor en consecuencia.
    sort_by_abs_weight:
        Si es ``True`` (default), el orden de dibujo de los nodos sigue
        el valor absoluto del peso: los de influencia más fuerte se
        dibujan al final y quedan **encima** donde hay solapamiento
        (núcleo denso), en vez de quedar enterrados bajo nodos casi
        blancos. Con ``False`` se dibujan en orden de índice de nodo.
    layout, seed, layout_k:
        Ver :func:`compute_graph_layout`.
    cmap, vlim:
        Colormap divergente y rango fijo. Default ``RdBu`` con
        ``[-1, 1]``: azul = positivo, rojo = negativo, blanco ≈ 0.
        (Nota: es la convención **inversa** a la de los topoplots EEG
        de ``plot_topomap_grid``, que usan ``RdBu_r``.)
    title:
        Título global de la figura (``None`` o ``""`` para omitir).
    node_size, edge_color, edge_alpha, edge_width:
        Estética de nodos y aristas.
    figsize_per_dim:
        Tamaño por panel; la figura final escala con el n.º de
        dimensiones latentes.
    max_cols:
        Máximo de paneles por fila (se envuelve a más filas si
        ``n_dim > max_cols``).

    Returns
    -------
    plt.Figure
        La figura (sin guardar); quien llama decide path y dpi.
    """
    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0)
    adjacency = np.asarray(adjacency, dtype=float)
    n_channels, n_dim = weights.shape
    if adjacency.shape != (n_channels, n_channels):
        raise ValueError(
            f"adjacency debe ser ({n_channels}, {n_channels}); "
            f"recibida shape={adjacency.shape}"
        )
    # Defensivo: binarizar, simetrizar y sin auto-lazos.
    adjacency = (adjacency != 0).astype(float)
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)

    if dim_labels is None:
        dim_labels = [f"LD{d + 1}" for d in range(n_dim)]
    if len(dim_labels) != n_dim:
        raise ValueError(
            f"dim_labels tiene {len(dim_labels)} entradas pero hay {n_dim} dimensiones."
        )

    pos = compute_graph_layout(adjacency, layout=layout, seed=seed, k=layout_k)
    edge_i, edge_j = np.nonzero(np.triu(adjacency, k=1))

    n_cols = min(n_dim, max_cols)
    n_rows = int(np.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_dim[0] * n_cols, figsize_per_dim[1] * n_rows),
        squeeze=False,
    )
    vmin, vmax = vlim

    for d in range(n_dim):
        ax = axes[d // n_cols][d % n_cols]
        # Aristas estructurales debajo de los nodos.
        for i, j in zip(edge_i, edge_j):
            ax.plot(
                [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=edge_color, lw=edge_width, alpha=edge_alpha, zorder=1,
            )
        # Orden de superposición: por |peso| de influencia. Los nodos
        # casi blancos (|peso| ≈ 0) se dibujan primero y los de color
        # más fuerte al final → donde hay solapamiento (núcleo denso)
        # los nodos relevantes quedan ENCIMA y no quedan enterrados.
        draw_order = (
            np.argsort(np.abs(weights[:, d]), kind="stable")
            if sort_by_abs_weight
            else np.arange(n_channels)
        )
        pos_d = pos[draw_order]
        weights_d = weights[draw_order, d]
        scatter = ax.scatter(
            pos_d[:, 0], pos_d[:, 1],
            c=weights_d, cmap=cmap, vmin=vmin, vmax=vmax,
            s=node_size, edgecolors="k", linewidths=0.8, zorder=3,
        )
        if show_node_labels:
            # Etiqueta centrada dentro del nodo, dibujada en el mismo
            # orden: la etiqueta del nodo fuerte también queda encima.
            # Default: índice del nodo (0-based, orden de la matriz
            # estructural / canales). El color del texto se elige por
            # luminancia del marcador para que sea legible sobre nodos
            # oscuros y claros.
            labels = (
                [str(name) for name in node_labels]
                if node_labels is not None
                else [str(i) for i in range(n_channels)]
            )
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap)
            for idx in draw_order:
                rgba = cmap_obj(norm(weights[idx, d]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(
                    pos[idx, 0], pos[idx, 1], labels[idx],
                    ha="center", va="center",
                    fontsize=node_label_fontsize,
                    color="white" if luminance < 0.5 else "black",
                    zorder=4,
                )
        ax.set_title(dim_labels[d], fontsize=13)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.margins(0.15)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    # Apagar axes sobrantes cuando n_dim no llena la rejilla.
    for d in range(n_dim, n_rows * n_cols):
        axes[d // n_cols][d % n_cols].axis("off")

    if title:
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    return fig

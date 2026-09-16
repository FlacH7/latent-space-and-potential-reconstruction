"""
io.py
======================
Helper de I/O para guardar y cargar potenciales reconstruidos via IgA.
Permite serializar/deserializar toda la informacion necesaria para recrear
un potencial reconstruido (B-spline space, coeficientes, edges, mascaras).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Lazy import de BSplineTensorSpace para evitar fallos cuando el modulo
# no esta disponible en el entorno de import.
# ---------------------------------------------------------------------------
_BSplineTensorSpace = None

def _get_bspline_space_class():
    """Importa y devuelve BSplineTensorSpace de forma lazy."""
    global _BSplineTensorSpace
    if _BSplineTensorSpace is not None:
        return _BSplineTensorSpace
    try:
        from src.potential_reconstruction.iga_reconstructor import BSplineTensorSpace
        _BSplineTensorSpace = BSplineTensorSpace
        return BSplineTensorSpace
    except ImportError:
        try:
            from src.potential_reconstruction.iga_reconstructor import BSplineTensorSpace
            _BSplineTensorSpace = BSplineTensorSpace
            return BSplineTensorSpace
        except ImportError:
            import sys
            _HERE = Path(__file__).resolve().parent
            if str(_HERE) not in sys.path:
                sys.path.insert(0, str(_HERE))
            try:
                from src.potential_reconstruction.iga_reconstructor import BSplineTensorSpace
                _BSplineTensorSpace = BSplineTensorSpace
                return BSplineTensorSpace
            except ImportError:
                return None


def save_potential(
    result_iga: dict,
    edges: list[np.ndarray],
    density: np.ndarray,
    config: dict,
    out_dir: str | Path,
    metadata: dict | None = None,
    filename: str = "potential_data.npz",
) -> Path:
    """
    Guarda toda la informacion necesaria para recrear un potencial.

    Parameters
    ----------
    result_iga : dict
        Diccionario devuelto por ``reconstruct_potential(..., return_full=True)``.
        Debe contener al menos: ``potential``, ``coefficients``, ``g_field``,
        ``density_mask``.  Opcionalmente: ``gradient``, ``residual``, ``eta``,
        ``is_equilibrium``.
    edges : list of np.ndarray
        Centros de los bins por dimension (de la etapa KM).
    density : np.ndarray
        Densidad empirica en la grilla.
    config : dict
        Configuracion del pipeline (model_name, D, bins, degree, etc.).
    out_dir : str or Path
        Directorio de salida (se crea si no existe).
    metadata : dict, optional
        Metadatos adicionales: subject, stage_label, t_start, t_end, epoch_idx, etc.
    filename : str, default "potential_data.npz"
        Nombre del fichero a escribir.

    Returns
    -------
    output_path : Path
        Ruta al fichero ``.npz`` guardado.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / filename

    # --- arrays obligatorios ---
    potential = result_iga["potential"]
    coefficients = result_iga["coefficients"]
    g_field = result_iga["g_field"]
    density_mask = result_iga["density_mask"]

    # --- arrays opcionales ---
    gradient = result_iga.get("gradient")
    residual = result_iga.get("residual")
    eta = result_iga.get("eta")
    is_equilibrium = result_iga.get("is_equilibrium")
    rank_D = result_iga.get("rank_D")
    condition_D = result_iga.get("condition_D")

    # Serializar edges: cada edge es un array 1D de longitud variable
    edge_dict = {}
    for d, e in enumerate(edges):
        edge_dict[f"edge_dim_{d}"] = e.astype(float)

    # Serializar config y metadata como JSON empaquetado en bytes
    config_bytes = json.dumps(config, default=str).encode("utf-8")
    meta_bytes = json.dumps(metadata or {}, default=str).encode("utf-8")

    # Construir dict de guardado
    save_dict = {
        "potential": potential.astype(float),
        "coefficients": coefficients.astype(float),
        "g_field": g_field.astype(float),
        "density_mask": density_mask.astype(bool),
        "density": density.astype(float),
        "config_json": config_bytes,
        "metadata_json": meta_bytes,
        **edge_dict,
    }

    # Anadir opcionales si existen
    if gradient is not None:
        save_dict["gradient"] = gradient.astype(float)
    if residual is not None:
        save_dict["residual"] = residual.astype(float)
    if eta is not None:
        save_dict["eta"] = float(eta)
    if is_equilibrium is not None:
        save_dict["is_equilibrium"] = bool(is_equilibrium)
    if rank_D is not None:
        save_dict["rank_D"] = rank_D.astype(int)
    if condition_D is not None:
        save_dict["condition_D"] = condition_D.astype(float)

    np.savez_compressed(output_path, **save_dict)
    print(f"  [IO] Potential data saved to: {output_path}")
    return output_path


def load_potential(filepath: str | Path) -> dict[str, Any]:
    """
    Carga un potencial serializado por ``save_potential``.

    Parameters
    ----------
    filepath : str or Path
        Ruta al fichero ``.npz``.

    Returns
    -------
    data : dict
        Diccionario con:
        - ``potential`` : np.ndarray, potencial evaluado.
        - ``coefficients`` : np.ndarray, coeficientes B-spline.
        - ``g_field`` : np.ndarray, campo efectivo g = -D^+ f.
        - ``density_mask`` : np.ndarray bool.
        - ``density`` : np.ndarray, densidad empirica.
        - ``edges`` : list[np.ndarray], centros de bins por dimension.
        - ``config`` : dict, configuracion del pipeline.
        - ``metadata`` : dict, metadatos (subject, stage, etc.).
        - ``bspline_space`` : BSplineTensorSpace recreado (o None).
        - Opcionalmente: ``gradient``, ``residual``, ``eta``, ``is_equilibrium``,
          ``rank_D``, ``condition_D``.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Potential file not found: {filepath}")

    loaded = np.load(filepath, allow_pickle=True)

    # --- arrays principales ---
    data: dict[str, Any] = {
        "potential": loaded["potential"],
        "coefficients": loaded["coefficients"],
        "g_field": loaded["g_field"],
        "density_mask": loaded["density_mask"].astype(bool),
        "density": loaded["density"],
    }

    # --- recuperar edges ---
    edges = []
    d = 0
    while f"edge_dim_{d}" in loaded:
        edges.append(loaded[f"edge_dim_{d}"])
        d += 1
    data["edges"] = edges

    # --- config y metadata ---
    data["config"] = json.loads(loaded["config_json"].item().decode("utf-8"))
    data["metadata"] = json.loads(loaded["metadata_json"].item().decode("utf-8"))

    # --- opcionales ---
    for key in ("gradient", "residual", "rank_D", "condition_D"):
        if key in loaded:
            data[key] = loaded[key]
    if "eta" in loaded:
        data["eta"] = float(loaded["eta"])
    if "is_equilibrium" in loaded:
        data["is_equilibrium"] = bool(loaded["is_equilibrium"])

    # --- recrear BSplineTensorSpace (lazy) ---
    BSplineClass = _get_bspline_space_class()
    if BSplineClass is not None:
        degree = data["config"].get("degree", 2)
        try:
            bspline_space = BSplineClass(edges, degree=degree)
            data["bspline_space"] = bspline_space
        except Exception as exc:
            warnings.warn(f"[IO] No se pudo recrear BSplineTensorSpace: {exc}")
            data["bspline_space"] = None
    else:
        data["bspline_space"] = None

    return data


def recreate_potential_from_file(filepath: str | Path) -> tuple:
    """
    Carga y devuelve la tupla mas comumente usada:
    (potential, bspline_space, coefficients, edges, density_mask, density).
    """
    data = load_potential(filepath)
    return (
        data["potential"],
        data["bspline_space"],
        data["coefficients"],
        data["edges"],
        data["density_mask"],
        data["density"],
    )


def find_potential_files(root_dir: str | Path, filename: str = "potential_data.npz") -> list[Path]:
    """Busca recursivamente ficheros de potenciales."""
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []
    return sorted(root_dir.rglob(filename))


def group_potentials_by_subject(root_dir: str | Path, filename: str = "potential_data.npz") -> dict[str, list[Path]]:
    """Agrupa ficheros por sujeto."""
    files = find_potential_files(root_dir, filename)
    grouped: dict[str, list[Path]] = {}
    for f in files:
        parts = f.parts
        if "anphy" in parts:
            idx = parts.index("anphy")
            if idx + 1 < len(parts):
                subject = parts[idx + 1]
                grouped.setdefault(subject, []).append(f)
        else:
            subject = f.parent.name
            grouped.setdefault(subject, []).append(f)
    return grouped


def group_potentials_by_subject_and_stage(
    root_dir: str | Path, filename: str = "potential_data.npz"
) -> dict[tuple[str, str], list[Path]]:
    """Agrupa ficheros por (subject, stage)."""
    files = find_potential_files(root_dir, filename)
    grouped: dict[tuple[str, str], list[Path]] = {}
    for f in files:
        try:
            data = load_potential(f)
            meta = data.get("metadata", {})
            subject = meta.get("subject", "unknown")
            stage = meta.get("stage_label", "unknown")
            grouped.setdefault((subject, stage), []).append(f)
        except Exception:
            grouped.setdefault(("unknown", "unknown"), []).append(f)
    return grouped


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test I/O de potenciales IgA")
    parser.add_argument("file", type=str, help="Path a un .npz de potencial")
    args = parser.parse_args()

    print(f"Cargando: {args.file}")
    data = load_potential(args.file)

    print("\n--- Metadatos ---")
    for k, v in data["metadata"].items():
        print(f"  {k}: {v}")
    print(f"\nPotential shape: {data['potential'].shape}")
    print(f"Coeffs shape: {data['coefficients'].shape}")
    print(f"BSpline recreado: {data['bspline_space'] is not None}")
    if "eta" in data:
        print(f"Eta: {data['eta']:.4f}")

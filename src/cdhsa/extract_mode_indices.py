"""
extract_mode_indices.py
======================
Lee los archivos de salida de CD-HSA (hankel_info.json, config.json,
cdhsa_arrays.npz) y genera un JSON que mapea cada par
(super-sujeto, condicion) a los indices de los modos especificos
mas relevantes de esa condicion.

Uso::

    python -m src.cdhsa.extract_mode_indices \
        --results-dir results/cdhsa/session1/nSS5_.../.../eyesclosed_music
    python -m src.cdhsa.extract_mode_indices --results-dir <PATH> --top-n 2

El JSON de salida (mode_map.json) contiene, para cada condicion,
los indices de los modos especificos ordenados por eigenvalor,
y para cada (super-sujeto, condicion) la clave de la Hankel
asociada y la formula de proyeccion.

Contexto matematico
--------------------
Step D de CD-HSA calcula, para cada condicion c:

  1. Residuos: R(s,c) = (I - U0 U0^T) H(s,c)  para cada SS s
  2. Covarianza residual pool: R_bar(c) = (1/S) sum_s R(s,c) R(s,c)^T
  3. Descomposicion: R_bar(c) = W(c) Lambda(c) W(c)^T
  4. Los modos especificos son las columnas de W(c) en R^p.

Los modos son COMUNES a todos los super-sujetos dentro de una
condicion (se calculan haciendo pooling sobre S super-sujetos).

Para obtener las series temporales de dimension r_c de un par (s,c)::

    alpha = W(c)^T @ H(s,c)   ->  shape (r_c, T)

Las series se ordenan por eigenvalor (Lambda(c)) de mayor a menor.
Tomar los top-n da las n series que mas capturan la dinamica
especifica de la condicion c.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _safe_scalar(x) -> int | float | None:
    """Convertir un valor numpy (de cualquier dimensionalidad) a escalar Python.

    Maneja: None, 0-d arrays, 1D arrays de un elemento, y arrays
    multidimensionales (devuelve la lista conversion via tolist).
    """
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.flat[0].item()
    return arr.tolist()


def _json_safe(obj):
    """Convertir tipos numpy a tipos nativos de Python para JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _extract_per_condition(
    W: np.ndarray,
    lam: np.ndarray | None,
    r_specific: np.ndarray,
    C: int,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Extraer W(c) y lambda(c) por condicion desde los arrays planos.

    El array W_specific guardado por _save_result_arrays puede tener
    distintas formas dependiendo de como lo devuelva el pipeline:

    - 3D (C, p, r_max): stacked, cada condicion ocupa W[c, :, :r_c]
    - 2D (p, r_total): concatenado, se divide por sumas acumuladas de r_specific

    Parameters
    ----------
    W : ndarray
        Array de modos especificos (W_specific).
    lam : ndarray or None
        Array de eigenvalores (lambda_specific).
    r_specific : ndarray, shape (C,)
        Numero de modos por condicion.
    C : int
        Numero de condiciones.

    Returns
    -------
    list of (W_c, lam_c) tuples
        W_c has shape (p, r_c), lam_c has shape (r_c,) or None.
    """
    results = []

    if W.ndim == 3 and W.shape[0] == C:
        # --- Caso 3D: (C, p, r_max) ---
        for c in range(C):
            rc = int(r_specific[c])
            W_c = W[c, :, :rc]
            results.append((W_c, None))  # lam se procesa abajo

        if lam is not None:
            if lam.ndim == 2 and lam.shape[0] == C:
                for c in range(C):
                    rc = int(r_specific[c])
                    results[c] = (results[c][0], lam[c, :rc])
            elif lam.ndim == 1:
                # Flat concatenado igual que W
                offset = 0
                for c in range(C):
                    rc = int(r_specific[c])
                    results[c] = (results[c][0], lam[offset:offset + rc])
                    offset += rc

    elif W.ndim == 2:
        # --- Caso 2D: (p, r_total) concatenado ---
        offset = 0
        for c in range(C):
            rc = int(r_specific[c])
            W_c = W[:, offset:offset + rc]
            results.append((W_c, None))
            offset += rc

        if lam is not None:
            if lam.ndim == 1:
                offset = 0
                for c in range(C):
                    rc = int(r_specific[c])
                    results[c] = (results[c][0], lam[offset:offset + rc])
                    offset += rc
            elif lam.ndim == 2 and lam.shape[0] == C:
                for c in range(C):
                    rc = int(r_specific[c])
                    results[c] = (results[c][0], lam[c, :rc])

    else:
        raise ValueError(
            f'Shape de W_specific no reconocido: {W.shape}. '
            f'Se esperaba 2D (p, r_total) o 3D (C, p, r_max).'
        )

    return results


def build_mode_map(
    results_dir: str | Path,
    top_n: int = 2,
) -> dict:
    """Construir el mapeo de modos especificos desde resultados de CD-HSA.

    Parameters
    ----------
    results_dir : path
        Directorio que contiene hankel_info.json, config.json,
        y cdhsa_arrays.npz.
    top_n : int
        Cuantos modos especificos conservar por condicion (ordenados
        por eigenvalor descendente).

    Returns
    -------
    mode_map : dict
        Estructura JSON completa.
    """
    results_dir = Path(results_dir)

    # ------------------------------------------------------------------
    # 1. Cargar metadata
    # ------------------------------------------------------------------
    with open(results_dir / 'hankel_info.json') as f:
        hankel_info = json.load(f)

    with open(results_dir / 'config.json') as f:
        config = json.load(f)

    tasks = hankel_info.get('tasks', [])
    S = hankel_info.get('n_super_subjects',
                        hankel_info.get('S', 1))
    C = len(tasks)
    p_ch = hankel_info.get('n_channels_common',
                          hankel_info.get('n_global_channels', None))
    depth = hankel_info.get('depth_common',
                            hankel_info.get('hankel_depth_requested', None))
    p_hankel = (p_ch * depth) if (p_ch is not None and depth is not None) else None

    # ------------------------------------------------------------------
    # 2. Cargar cdhsa_arrays.npz
    # ------------------------------------------------------------------
    npz_path = results_dir / 'cdhsa_arrays.npz'
    npz_data = np.load(npz_path, allow_pickle=True)

    # ------------------------------------------------------------------
    # 3. Extraer arrays de Step D (keys flat D__*)
    # ------------------------------------------------------------------
    d_keys = [k for k in npz_data.files if k.startswith('D__')]

    # --- r_specific: (C,) numero de modos por condicion ---
    if 'D__r_specific' not in npz_data:
        raise ValueError(
            'No se encontro D__r_specific en cdhsa_arrays.npz. '
            'Step D puede no haberse ejecutado.\n'
            f'Keys D__ disponibles: {d_keys}'
        )
    r_specific = npz_data['D__r_specific']
    C_npz = len(r_specific)
    if C_npz != C:
        raise ValueError(
            f'Inconsistencia: hankel_info dice C={C} pero '
            f'D__r_specific tiene {C_npz} elementos'
        )

    # --- prevalence_contrast: (C,) ---
    prev_contrast = (npz_data['D__prevalence_contrast']
                     if 'D__prevalence_contrast' in npz_data else None)

    # --- prevalence_own: (C,) ---
    prev_own = (npz_data['D__prevalence_own']
                if 'D__prevalence_own' in npz_data else None)

    # --- W_specific: modos especificos (3D o 2D) ---
    if 'D__W_specific' not in npz_data:
        raise ValueError(
            'No se encontro D__W_specific en cdhsa_arrays.npz. '
            'Step D puede no haberse ejecutado.\n'
            f'Keys D__ disponibles: {d_keys}'
        )
    W_all = npz_data['D__W_specific']

    # --- lambda_specific: eigenvalores (opcional) ---
    lam_all = (npz_data['D__lambda_specific']
               if 'D__lambda_specific' in npz_data else None)

    # --- residual_rank: (C,) o escalar ---
    residual_rank = (npz_data['D__residual_rank']
                     if 'D__residual_rank' in npz_data else None)

    # ------------------------------------------------------------------
    # 4. Extraer W(c) y lambda(c) por condicion
    # ------------------------------------------------------------------
    per_condition = _extract_per_condition(W_all, lam_all, r_specific, C)

    # ------------------------------------------------------------------
    # 5. Para cada condicion: ordenar por eigenvalor, seleccionar top-N
    # ------------------------------------------------------------------
    conditions_data = {}

    for c_idx in range(C):
        task_name = tasks[c_idx]
        W_c, lam_c = per_condition[c_idx]
        r_c = W_c.shape[1]

        # Ordenar por eigenvalor descendente
        if lam_c is not None and len(lam_c) == r_c:
            order = np.argsort(lam_c)[::-1]
        else:
            order = np.arange(r_c)

        n_actual = min(top_n, r_c)
        top_indices = [int(order[i]) for i in range(n_actual)]

        cond_info = {
            'task': task_name,
            'condition_index': c_idx,
            'total_specific_modes': int(r_c),
            'top_n_requested': top_n,
            'top_n_actual': n_actual,
            'mode_indices_in_W': top_indices,
            'W_npz_key': 'D__W_specific',
            'W_shape_full': list(W_all.shape),
            'W_c_shape': list(W_c.shape),
            'eigenvalues_npz_key': ('D__lambda_specific'
                                   if lam_all is not None else None),
            'eigenvalues': _json_safe(lam_c) if lam_c is not None else None,
            'eigenvalues_sorted': (_json_safe(lam_c[order])
                                   if lam_c is not None else None),
            'prevalence_contrast': (float(prev_contrast[c_idx])
                                    if prev_contrast is not None else None),
            'prevalence_own': (float(prev_own[c_idx])
                               if prev_own is not None else None),
            'r_specific_from_pipeline': int(r_specific[c_idx]),
            'residual_rank': _safe_scalar(residual_rank),
        }

        # Info detallada de cada modo seleccionado
        cond_info['modes'] = []
        for rank_pos, mode_idx in enumerate(top_indices):
            mode_info = {
                'rank_position': rank_pos + 1,
                'column_index_in_W_c': mode_idx,
                'eigenvalue': (float(lam_c[mode_idx])
                               if lam_c is not None else None),
                'eigenvalue_rank': rank_pos + 1,
            }
            cond_info['modes'].append(mode_info)

        conditions_data[task_name] = cond_info

    # ------------------------------------------------------------------
    # 6. Para cada (super-sujeto, condicion): mapeo a Hankel + proyeccion
    # ------------------------------------------------------------------
    per_pair = []

    for s_idx in range(S):
        for c_idx in range(C):
            task_name = tasks[c_idx]
            hankel_key = f'H_ss{s_idx + 1}_c{c_idx + 1}'

            cond = conditions_data[task_name]
            mode_indices = cond['mode_indices_in_W']

            # Construir la formula de proyeccion concreta
            # Depende de si W es 3D o 2D
            if W_all.ndim == 3:
                w_slice = f'W_specific[{c_idx}, :, :{cond["total_specific_modes"]}][:, {mode_indices}]'
            else:
                rc = cond['total_specific_modes']
                # Calcular offset para esta condicion
                offset = int(np.sum(r_specific[:c_idx]))
                w_slice = f'W_specific[:, {offset}:{offset + rc}][:, {mode_indices}]'

            entry = {
                'super_subject': s_idx + 1,
                'super_subject_label': f'SS{s_idx + 1}',
                'task': task_name,
                'task_index': c_idx,
                'condition_index_in_W': c_idx if W_all.ndim == 3 else None,
                'hankel_npz_key': hankel_key,
                'hankel_npz_file': 'hankel_matrices.npz',
                'W_npz_key': 'D__W_specific',
                'W_npz_file': 'cdhsa_arrays.npz',
                'columns_to_extract': mode_indices,
                'n_output_dimensions': cond['top_n_actual'],
                'projection_code': (
                    f'W = cdhsa["D__W_specific"]  # {list(W_all.shape)}\n'
                    f'H = hankel["{hankel_key}"]      # (p, T)\n'
                    f'W_sel = {w_slice}             # (p, {n_actual})\n'
                    f'alpha = W_sel.T @ H            # ({n_actual}, T)'
                ),
            }
            per_pair.append(entry)

    # ------------------------------------------------------------------
    # 7. Info del espacio comun (A6)
    # ------------------------------------------------------------------
    common_info = {}
    for k in npz_data.files:
        if k.startswith('A6__'):
            short_name = k.replace('A6__', '')
            common_info[short_name] = _json_safe(npz_data[k])

    # ------------------------------------------------------------------
    # 8. Ensamblar JSON final
    # ------------------------------------------------------------------
    mode_map = {
        'metadata': {
            'source_dir': str(results_dir),
            'n_super_subjects': S,
            'n_conditions': C,
            'tasks': tasks,
            'p_hankel': p_hankel,
            'n_channels': p_ch,
            'hankel_depth': depth,
            'top_n_requested': top_n,
            'pipeline_config': {
                'fixed_rank': config.get('fixed_rank'),
                'L': config.get('L'),
                'd_max_specific': config.get('d_max_specific', 10),
                'a6_n_null': config.get('a6_n_null'),
                'bc_n_perm': config.get('bc_n_perm'),
            },
        },
        'common_subspace': common_info,
        'conditions': conditions_data,
        'per_super_subject_task': per_pair,
        'npz_structure': {
            'W_specific_shape': list(W_all.shape),
            'lambda_specific_shape': (list(lam_all.shape)
                                      if lam_all is not None else None),
            'r_specific': _json_safe(r_specific),
            'all_D_keys': d_keys,
        },
        'usage': {
            'description': (
                'Para obtener las series temporales de dimension top_n '
                'del par (super_subject, task), proyectar la Hankel '
                'sobre los modos especificos indicados en columns_to_extract.'
            ),
            'python_example': (
                'import numpy as np\n'
                'hankel = np.load("hankel_matrices.npz")\n'
                'cdhsa = np.load("cdhsa_arrays.npz")\n'
                'W = cdhsa["D__W_specific"]  # ver shape en npz_structure\n'
                'H = hankel["H_ss1_c1"]       # (p, T)\n'
                'idx = [0, 1]  # columns_to_extract del JSON\n'
                'alpha = W[:, idx].T @ H        # (2, T)\n'
                '# alpha[0] = serie temporal del modo mas importante\n'
                '# alpha[1] = serie temporal del segundo modo'
            ),
        },
    }

    return mode_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Extraer indices de modos especificos de CD-HSA y generar '
            'mode_map.json'
        ),
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directorio con los archivos de salida de CD-HSA',
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=2,
        help=(
            'Cuantos modos especificos conservar por condicion '
            '(default: 2)'
        ),
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help=(
            'Ruta del JSON de salida. Default: mode_map.json '
            'en el mismo results-dir'
        ),
    )

    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        print(f'[ERROR] Directorio no encontrado: {results_dir}',
              file=sys.stderr)
        return 1

    # Verificar archivos necesarios
    required = ['hankel_info.json', 'config.json', 'cdhsa_arrays.npz']
    for fname in required:
        if not (results_dir / fname).exists():
            print(f'[ERROR] Archivo requerido no encontrado: {fname}',
                  file=sys.stderr)
            return 1

    print(f'Leyendo resultados de: {results_dir}')
    print(f'Top-N modos por condicion: {args.top_n}')
    print()

    mode_map = build_mode_map(results_dir, top_n=args.top_n)

    # Guardar
    out_path = (Path(args.output) if args.output
                else results_dir / 'mode_map.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(mode_map, f, indent=2, ensure_ascii=False, default=str)

    print(f'[OK] mode_map.json guardado en: {out_path}')
    print()

    # Resumen
    meta = mode_map['metadata']
    print(f'  Super-sujetos: {meta["n_super_subjects"]}')
    print(f'  Condiciones:   {meta["n_conditions"]} ({meta["tasks"]})')
    print(f'  p (Hankel):    {meta["p_hankel"]}')
    print()

    print('  Modos especificos por condicion:')
    for task_name, cond in mode_map['conditions'].items():
        r_c = cond['total_specific_modes']
        n_actual = cond['top_n_actual']
        indices = cond['mode_indices_in_W']
        pc = cond['prevalence_contrast']
        print(f'    {task_name}: {r_c} modos, '
              f'top-{n_actual} = cols {indices}, '
              f'prev_contrast={pc}')
        for m in cond['modes']:
            ev = m['eigenvalue']
            ev_str = f'ev={ev:.6f}' if ev is not None else 'ev=N/A'
            print(f'      rank {m["rank_position"]}: col {m["column_index_in_W_c"]} ({ev_str})')

    print()
    print('  Mapeo (SS, task) -> Hankel key -> cols:')
    for entry in mode_map['per_super_subject_task']:
        print(f'    SS{entry["super_subject"]}/{entry["task"]}: '
              f'{entry["hankel_npz_key"]} -> {entry["columns_to_extract"]}')

    print()
    npz_struct = mode_map['npz_structure']
    print(f'  W_specific shape: {npz_struct["W_specific_shape"]}')
    if npz_struct['lambda_specific_shape']:
        print(f'  lambda_specific shape: {npz_struct["lambda_specific_shape"]}')
    print(f'  r_specific: {npz_struct["r_specific"]}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

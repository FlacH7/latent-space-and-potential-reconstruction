"""
km_tools_v2.py
================
Pipeline completo para estimación de coeficientes de Kramers-Moyal,
reconstrucción de potenciales via IgA (Galerkin-B-spline), y simulación
de datos estocásticos.

Versión 4.0 — Reconstrucción por Análisis Isogeométrico.

Cambios principales respecto a v3.0:
  - Reemplaza integración BFS/path por Galerkin con B-splines tensoriales.
  - Inversión robusta de D via SVD truncado adaptativo (pseudo-inversa).
  - Descomposición de Helmholtz explícita con métrica de no-equilibrio η.
  - Compatible D=1..5, bins no equidistantes, sin ground truth.
"""

import numpy as np
from kramersmoyal import km
from kramersmoyal.kernels import epanechnikov, gaussian
import matplotlib.pyplot as plt

import hashlib
import json
import os
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from scipy.stats import pearsonr
import scipy.ndimage as ndi

# ---------------------------------------------------------------------------
# Importar núcleo IgA
# ---------------------------------------------------------------------------
from src.potential_reconstruction.iga_reconstructor import (
    reconstruct_potential_iga,
    compute_g_robust,
    compute_helmholtz_residual,
    BSplineTensorSpace
)

from src.utils.config import(
    BASE_CACHE_PATH
)

# =============================================================================
# SECCIÓN 0: UTILIDADES GENERALES
# =============================================================================

def _param_to_str(param):
    if np.isscalar(param):
        return f"scalar:{float(param):.12g}"
    else:
        arr = np.asarray(param)
        return f"array:{arr.shape}:{np.array2string(arr, precision=12, separator=',', suppress_small=True)}"

def _build_cache_path(config, cache_dir='sim_cache'):
    param_dict = {
        'model': config['model'],
        'D': config['D'],
        'dt': config['dt'],
        'T': config['T'],
        'burn_ratio': config.get('burn_ratio', 0.1),
        'seed': config.get('seed', None),
        'params': {k: _param_to_str(v) for k, v in config.get('params', {}).items()}
    }
    param_str = json.dumps(param_dict, sort_keys=True, ensure_ascii=True)
    param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
    cache_filename = f"sim_D{config['D']}_{config['model']}_{param_hash}.npz"
    return os.path.join(BASE_CACHE_PATH, cache_dir, cache_filename), param_dict

def _rebuild_theoretical(model_info, D):
    """Reconstruye el dict 'theoretical' a partir de model_info (serializable)."""
    name = model_info['name']
    if name == 'ou':
        theta_mat = model_info['theta']
        sigma_mat = model_info['sigma']
        D_diff = model_info['D_diff']
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1: return -theta_mat @ x
            else: return (-theta_mat @ x.T).T
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            D_inv = np.linalg.inv(D_diff + 1e-10 * np.eye(D))
            A = D_inv @ theta_mat
            A_sym = (A + A.T) / 2
            if x.ndim == 1: return 0.5 * x @ A_sym @ x
            else: return 0.5 * np.sum(x * (A_sym @ x.T).T, axis=1)
    elif name == 'double_well':
        a_vec = model_info['a']; b_vec = model_info['b']; c = model_info['c']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = 2 * a_vec * x - 4 * b_vec * x**3
                if c != 0:
                    for i in range(D): d[i] -= 2 * c * np.sum(x[i] - x)
                return d
            else:
                d = 2 * a_vec * x - 4 * b_vec * x**3
                if c != 0:
                    for i in range(D): d[:, i] -= 2 * c * np.sum(x[:, i:i+1] - x, axis=1)
                return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = np.sum(-a_vec * x**2 + b_vec * x**4)
                if c != 0:
                    for i in range(D):
                        for j in range(i+1, D): U += c * (x[i] - x[j])**2
                return U
            else:
                U = np.sum(-a_vec * x**2 + b_vec * x**4, axis=1)
                if c != 0:
                    for i in range(D):
                        for j in range(i+1, D): U += c * (x[:, i] - x[:, j])**2
                return U
    elif name == 'ring_attractor':
        alpha = model_info['alpha']; r0 = model_info['r0']; omega = model_info['omega']
        theta_rest_vec = model_info['theta_rest']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = np.zeros(D)
                if D >= 2:
                    x1, x2 = x[0], x[1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
                    d[0] = -alpha * (r - r0) * (x1 / r) - omega * x2
                    d[1] = -alpha * (r - r0) * (x2 / r) + omega * x1
                for i in range(2, D): d[i] = -theta_rest_vec[i-2] * x[i]
                return d
            else:
                d = np.zeros_like(x)
                if D >= 2:
                    x1, x2 = x[:, 0], x[:, 1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
                    d[:, 0] = -alpha * (r - r0) * (x1 / r) - omega * x2
                    d[:, 1] = -alpha * (r - r0) * (x2 / r) + omega * x1
                for i in range(2, D): d[:, i] = -theta_rest_vec[i-2] * x[:, i]
                return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = 0.0
                if D >= 2: r = np.sqrt(x[0]**2 + x[1]**2); U += 0.5 * alpha * (r - r0)**2
                for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[i]**2
                return U
            else:
                U = np.zeros(x.shape[0])
                if D >= 2: r = np.sqrt(x[:, 0]**2 + x[:, 1]**2); U += 0.5 * alpha * (r - r0)**2
                for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[:, i]**2
                return U
    elif name == 'multi_stable':
        a = model_info['a']; b = model_info['b']; c = model_info['c']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = 2 * a * x - 4 * b * x**3; sum_sq = np.sum(x**2)
                d -= 2 * c * x * (sum_sq - x**2); return d
            else:
                d = 2 * a * x - 4 * b * x**3; sum_sq = np.sum(x**2, axis=1, keepdims=True)
                d -= 2 * c * x * (sum_sq - x**2); return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = np.sum(-a * x**2 + b * x**4)
                for i in range(D):
                    for j in range(i+1, D): U += c * x[i]**2 * x[j]**2
                return U
            else:
                U = np.sum(-a * x**2 + b * x**4, axis=1)
                for i in range(D):
                    for j in range(i+1, D): U += c * x[:, i]**2 * x[:, j]**2
                return U
    elif name == 'stochastic_oscillator':
        lam = model_info['lambda']; omega = model_info['omega']
        theta_rest_vec = model_info['theta_rest']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = np.zeros(D)
                if D >= 2:
                    x1, x2 = x[0], x[1]; r2 = x1**2 + x2**2
                    d[0] = lam * x1 - omega * x2 - r2 * x1
                    d[1] = lam * x2 + omega * x1 - r2 * x2
                for i in range(2, D): d[i] = -theta_rest_vec[i-2] * x[i]
                return d
            else:
                d = np.zeros_like(x)
                if D >= 2:
                    x1, x2 = x[:, 0], x[:, 1]; r2 = x1**2 + x2**2
                    d[:, 0] = lam * x1 - omega * x2 - r2 * x1
                    d[:, 1] = lam * x2 + omega * x1 - r2 * x2
                for i in range(2, D): d[:, i] = -theta_rest_vec[i-2] * x[:, i]
                return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = 0.0
                if D >= 2: r2 = x[0]**2 + x[1]**2; U += -0.5 * lam * r2 + 0.25 * r2**2
                for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[i]**2
                return U
            else:
                U = np.zeros(x.shape[0])
                if D >= 2: r2 = x[:, 0]**2 + x[:, 1]**2; U += -0.5 * lam * r2 + 0.25 * r2**2
                for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[:, i]**2
                return U
    elif name == 'single_well':
        k = model_info['k']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            return -k * x if x.ndim == 1 else -k * x
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            return 0.5 * k * np.sum(x**2) if x.ndim == 1 else 0.5 * k * np.sum(x**2, axis=1)
    elif name == 'asymmetric_double_well':
        a = model_info['a']; b = model_info['b']; c = model_info['c']; d = model_info['d']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            return -(4*a*x**3 - 3*b*x**2 + 2*c*x + d) if x.ndim == 1 else -(4*a*x**3 - 3*b*x**2 + 2*c*x + d)
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            return a*x**4 - b*x**3 + c*x**2 + d*x if x.ndim == 1 else a*x**4 - b*x**3 + c*x**2 + d*x
    elif name == 'triple_well_3d':
        a = model_info['a']; b = model_info['b']; c = model_info['c']; k_rest = model_info['k_rest']
        sigma_mat = model_info['sigma']; D_diff = (sigma_mat @ sigma_mat.T) / 2.0
        def drift_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                d = np.zeros(D)
                for i in range(D):
                    d[i] = 2*a*x[i] - 4*b*x[i]**3
                if D >= 2:
                    for i in range(D):
                        for j in range(i+1, D):
                            if j < 3:
                                d[i] -= 2*c*x[i]*x[j]**2
                                d[j] -= 2*c*x[j]*x[i]**2
                for i in range(2, D):
                    d[i] -= 2*k_rest*x[i]
                return d
            else:
                d = np.zeros_like(x)
                for i in range(D):
                    d[:, i] = 2*a*x[:, i] - 4*b*x[:, i]**3
                if D >= 2:
                    for i in range(D):
                        for j in range(i+1, D):
                            if j < 3:
                                d[:, i] -= 2*c*x[:, i]*x[:, j]**2
                                d[:, j] -= 2*c*x[:, j]*x[:, i]**2
                for i in range(2, D):
                    d[:, i] -= 2*k_rest*x[:, i]
                return d
        def diffusion_func(x): return D_diff
        def potential_func(x):
            x = np.asarray(x)
            if x.ndim == 1:
                U = np.sum(-a*x**2 + b*x**4)
                for i in range(D):
                    for j in range(i+1, D):
                        if j < 3:
                            U += c*x[i]**2*x[j]**2
                for i in range(2, D):
                    U += k_rest*x[i]**2
                return U
            else:
                U = np.sum(-a*x**2 + b*x**4, axis=1)
                for i in range(D):
                    for j in range(i+1, D):
                        if j < 3:
                            U += c*x[:, i]**2*x[:, j]**2
                for i in range(2, D):
                    U += k_rest*x[:, i]**2
                return U
    else:
        raise ValueError(f"Modelo {name} no reconocido para reconstrucción")
    return {
        'drift_func': drift_func,
        'diffusion_func': diffusion_func,
        'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }

def _eval_drift_on_grid(drift_func, edges):
    grid_shape = tuple(len(e) for e in edges)
    N = np.prod(grid_shape)
    mesh = np.meshgrid(*edges, indexing='ij')
    points = np.stack([m.ravel() for m in mesh], axis=1)
    try:
        vals = drift_func(points)
    except Exception:
        vals = np.array([drift_func(p) for p in points])
    vals = np.asarray(vals)
    if vals.shape[:1] != (N,):
        vals = np.broadcast_to(vals, (N,) + vals.shape).copy()
    return vals.reshape(grid_shape + (vals.shape[-1],)).transpose(-1, *range(len(grid_shape)))

def _eval_diffusion_on_grid(diffusion_func, edges):
    grid_shape = tuple(len(e) for e in edges)
    N = np.prod(grid_shape)
    mesh = np.meshgrid(*edges, indexing='ij')
    points = np.stack([m.ravel() for m in mesh], axis=1)
    try:
        vals = diffusion_func(points)
    except Exception:
        vals = np.array([diffusion_func(p) for p in points])
    vals = np.asarray(vals)
    if vals.shape[:1] != (N,):
        vals = np.broadcast_to(vals, (N,) + vals.shape).copy()
    D_out = vals.shape[-2]
    return vals.reshape(grid_shape + (D_out, D_out)).transpose(-2, -1, *range(len(grid_shape)))

def _eval_potential_on_grid(potential_func, edges):
    grid_shape = tuple(len(e) for e in edges)
    N = np.prod(grid_shape)
    mesh = np.meshgrid(*edges, indexing='ij')
    points = np.stack([m.ravel() for m in mesh], axis=1)
    try:
        vals = potential_func(points)
    except Exception:
        vals = np.array([potential_func(p) for p in points])
    vals = np.asarray(vals)
    if vals.shape[:1] != (N,):
        return np.broadcast_to(vals, grid_shape).copy()
    return vals.reshape(grid_shape)

# =============================================================================
# SECCIÓN 1: SIMULACIÓN DE DATOS (sin cambios funcionales)
# =============================================================================

def simulate_data(config, cache_dir='sim_cache', use_cache=True):
    """Orquestador de simulación (sin cambios respecto a v3.0)."""
    model = config['model']
    D = config['D']
    dt = config['dt']
    T = config['T']
    burn_ratio = config.get('burn_ratio', 0.1)
    seed = config.get('seed', None)
    params = config.get('params', {})

    if not (1 <= D <= 5):
        raise ValueError(f"D={D} no soportado. Use 1 <= D <= 5.")

    cache_path, param_dict = _build_cache_path(config, cache_dir)
    if use_cache and os.path.exists(cache_path):
        try:
            loaded = np.load(cache_path, allow_pickle=True)
            data = loaded['data']
            model_info = loaded['model_info'].item()
            theoretical = _rebuild_theoretical(model_info, D)
            print(f"[CACHE] Cargando simulación existente: {os.path.basename(cache_path)}")
            return {'data': data, 'theoretical': theoretical, 'config': config, 'model_info': model_info}
        except Exception as e:
            print(f"[CACHE] Archivo corrupto ({e}), regenerando...")

    print(f"[SIM] Generando {model} (D={D})...")
    if model == 'ou':
        data, theoretical, model_info = _simulate_ou(D, dt, T, burn_ratio, seed, params)
    elif model == 'double_well':
        data, theoretical, model_info = _simulate_double_well(D, dt, T, burn_ratio, seed, params)
    elif model == 'single_well':
        data, theoretical, model_info = _simulate_single_well(D, dt, T, burn_ratio, seed, params)
    elif model == 'asymmetric_double_well':
        data, theoretical, model_info = _simulate_asymmetric_double_well(D, dt, T, burn_ratio, seed, params)
    elif model == 'triple_well_3d':
        data, theoretical, model_info = _simulate_triple_well_3d(D, dt, T, burn_ratio, seed, params)
    elif model == 'ring_attractor':
        data, theoretical, model_info = _simulate_ring_attractor(D, dt, T, burn_ratio, seed, params)
    elif model == 'multi_stable':
        data, theoretical, model_info = _simulate_multi_stable(D, dt, T, burn_ratio, seed, params)
    elif model == 'stochastic_oscillator':
        data, theoretical, model_info = _simulate_stochastic_oscillator(D, dt, T, burn_ratio, seed, params)
    else:
        raise ValueError(f"Modelo '{model}' no reconocido.")

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, data=data, model_info=model_info)
        print(f"[CACHE] Guardada en: {cache_path}")

    return {'data': data, 'theoretical': theoretical, 'config': config, 'model_info': model_info}

# --- Modelos privados (sin cambios) ---

def _simulate_ou(D, dt, T, burn_ratio, seed, params):
    theta = params.get('theta', 1.0)
    sigma = params.get('sigma', 0.5)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    theta_mat = theta * np.eye(D) if np.isscalar(theta) else np.asarray(theta)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = X[t-1] - theta_mat @ X[t-1] * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        return -theta_mat @ x if x.ndim == 1 else (-theta_mat @ x.T).T
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        D_inv = np.linalg.inv(D_diff + 1e-10 * np.eye(D))
        A_sym = (D_inv @ theta_mat + theta_mat.T @ D_inv.T) / 2
        return 0.5 * x @ A_sym @ x if x.ndim == 1 else 0.5 * np.sum(x * (A_sym @ x.T).T, axis=1)
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'ou', 'theta': theta_mat, 'sigma': sigma_mat, 'D_diff': D_diff}

def _simulate_single_well(D, dt, T, burn_ratio, seed, params):
    """
    Potencial armónico simple: U(x) = 0.5 * k * x^2.
    Ideal para validación 1D básica (equilibrio detallado, J=0).
    """
    k = params.get('k', 1.0)
    sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if sigma_mat.shape != (D, D):
        raise ValueError(f"sigma debe ser escalar o matriz {D}x{D}")
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(T / dt)
    burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1]
        drift = -k * x
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        return -k * x if x.ndim == 1 else -k * x
    def diffusion_func(x):
        return D_diff
    def potential_func(x):
        x = np.asarray(x)
        return 0.5 * k * np.sum(x**2) if x.ndim == 1 else 0.5 * k * np.sum(x**2, axis=1)
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'single_well', 'k': k, 'sigma': sigma_mat}


def _simulate_asymmetric_double_well(D, dt, T, burn_ratio, seed, params):
    """
    Potencial doble pozo asimétrico: U(x) = a*x^4 - b*x^3 + c*x^2 + d*x.
    Los pozos tienen profundidades diferentes (test de sensibilidad a asimetría).
    """
    a = params.get('a', 1.0)
    b = params.get('b', 0.5)
    c = params.get('c', -1.0)
    d = params.get('d', 0.2)
    sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if sigma_mat.shape != (D, D):
        raise ValueError(f"sigma debe ser escalar o matriz {D}x{D}")
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(T / dt)
    burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1]
        drift = -(4*a*x**3 - 3*b*x**2 + 2*c*x + d)
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        return -(4*a*x**3 - 3*b*x**2 + 2*c*x + d) if x.ndim == 1 else -(4*a*x**3 - 3*b*x**2 + 2*c*x + d)
    def diffusion_func(x):
        return D_diff
    def potential_func(x):
        x = np.asarray(x)
        return a*x**4 - b*x**3 + c*x**2 + d*x if x.ndim == 1 else a*x**4 - b*x**3 + c*x**2 + d*x
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'asymmetric_double_well', 'a': a, 'b': b, 'c': c, 'd': d, 'sigma': sigma_mat}


def _simulate_triple_well_3d(D, dt, T, burn_ratio, seed, params):
    """
    Potencial triple pozo en 3D: pozos en (±1,0,0) y (0,±1,0) con acoplamiento.
    Extensible a D>=3 (últimas D-2 dimensiones son armónicas).
    """
    a = params.get('a', 1.0)
    b = params.get('b', 1.0)
    c = params.get('c', 0.3)
    k_rest = params.get('k_rest', 1.0)
    sigma = params.get('sigma', 0.4)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if sigma_mat.shape != (D, D):
        raise ValueError(f"sigma debe ser escalar o matriz {D}x{D}")
    if seed is not None:
        np.random.seed(seed)
    n_steps = int(T / dt)
    burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1].copy()
        drift = np.zeros(D)
        for i in range(D):
            drift[i] = 2*a*x[i] - 4*b*x[i]**3
        if D >= 2:
            for i in range(D):
                for j in range(i+1, D):
                    if j < 3:
                        drift[i] -= 2*c*x[i]*x[j]**2
                        drift[j] -= 2*c*x[j]*x[i]**2
        for i in range(2, D):
            drift[i] -= 2*k_rest*x[i]
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.zeros(D)
            for i in range(D):
                d[i] = 2*a*x[i] - 4*b*x[i]**3
            if D >= 2:
                for i in range(D):
                    for j in range(i+1, D):
                        if j < 3:
                            d[i] -= 2*c*x[i]*x[j]**2
                            d[j] -= 2*c*x[j]*x[i]**2
            for i in range(2, D):
                d[i] -= 2*k_rest*x[i]
            return d
        else:
            d = np.zeros_like(x)
            for i in range(D):
                d[:, i] = 2*a*x[:, i] - 4*b*x[:, i]**3
            if D >= 2:
                for i in range(D):
                    for j in range(i+1, D):
                        if j < 3:
                            d[:, i] -= 2*c*x[:, i]*x[:, j]**2
                            d[:, j] -= 2*c*x[:, j]*x[:, i]**2
            for i in range(2, D):
                d[:, i] -= 2*k_rest*x[:, i]
            return d
    def diffusion_func(x):
        return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = np.sum(-a*x**2 + b*x**4)
            for i in range(D):
                for j in range(i+1, D):
                    if j < 3:
                        U += c*x[i]**2*x[j]**2
            for i in range(2, D):
                U += k_rest*x[i]**2
            return U
        else:
            U = np.sum(-a*x**2 + b*x**4, axis=1)
            for i in range(D):
                for j in range(i+1, D):
                    if j < 3:
                        U += c*x[:, i]**2*x[:, j]**2
            for i in range(2, D):
                U += k_rest*x[:, i]**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'triple_well_3d', 'a': a, 'b': b, 'c': c, 'k_rest': k_rest, 'sigma': sigma_mat}


def _simulate_double_well(D, dt, T, burn_ratio, seed, params):
    a = params.get('a', 1.0); b = params.get('b', 1.0); c = params.get('c', 0.1); sigma = params.get('sigma', 0.5)
    a_vec = np.full(D, a) if np.isscalar(a) else np.asarray(a)
    b_vec = np.full(D, b) if np.isscalar(b) else np.asarray(b)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1]
        drift = 2 * a_vec * x - 4 * b_vec * x**3
        if c != 0:
            for i in range(D): drift[i] -= 2 * c * np.sum(x[i] - x)
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = 2 * a_vec * x - 4 * b_vec * x**3
            if c != 0:
                for i in range(D): d[i] -= 2 * c * np.sum(x[i] - x)
            return d
        else:
            d = 2 * a_vec * x - 4 * b_vec * x**3
            if c != 0:
                for i in range(D): d[:, i] -= 2 * c * np.sum(x[:, i:i+1] - x, axis=1)
            return d
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = np.sum(-a_vec * x**2 + b_vec * x**4)
            if c != 0:
                for i in range(D):
                    for j in range(i+1, D): U += c * (x[i] - x[j])**2
            return U
        else:
            U = np.sum(-a_vec * x**2 + b_vec * x**4, axis=1)
            if c != 0:
                for i in range(D):
                    for j in range(i+1, D): U += c * (x[:, i] - x[:, j])**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'double_well', 'a': a_vec, 'b': b_vec, 'c': c, 'sigma': sigma_mat}

def _simulate_ring_attractor(D, dt, T, burn_ratio, seed, params):
    alpha = params.get('alpha', 1.0); r0 = params.get('r0', 1.0); omega = params.get('omega', 1.0)
    theta_rest = params.get('theta_rest', 1.0); sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    theta_rest_vec = np.full(max(0, D-2), theta_rest) if np.isscalar(theta_rest) else np.asarray(theta_rest)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1].copy(); drift = np.zeros(D)
        if D >= 2:
            x1, x2 = x[0], x[1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
            drift[0] = -alpha * (r - r0) * (x1 / r) - omega * x2
            drift[1] = -alpha * (r - r0) * (x2 / r) + omega * x1
        for i in range(2, D): drift[i] = -theta_rest_vec[i-2] * x[i]
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.zeros(D)
            if D >= 2:
                x1, x2 = x[0], x[1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
                d[0] = -alpha * (r - r0) * (x1 / r) - omega * x2
                d[1] = -alpha * (r - r0) * (x2 / r) + omega * x1
            for i in range(2, D): d[i] = -theta_rest_vec[i-2] * x[i]
            return d
        else:
            d = np.zeros_like(x)
            if D >= 2:
                x1, x2 = x[:, 0], x[:, 1]; r = np.sqrt(x1**2 + x2**2) + 1e-12
                d[:, 0] = -alpha * (r - r0) * (x1 / r) - omega * x2
                d[:, 1] = -alpha * (r - r0) * (x2 / r) + omega * x1
            for i in range(2, D): d[:, i] = -theta_rest_vec[i-2] * x[:, i]
            return d
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = 0.0
            if D >= 2: r = np.sqrt(x[0]**2 + x[1]**2); U += 0.5 * alpha * (r - r0)**2
            for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[i]**2
            return U
        else:
            U = np.zeros(x.shape[0])
            if D >= 2: r = np.sqrt(x[:, 0]**2 + x[:, 1]**2); U += 0.5 * alpha * (r - r0)**2
            for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[:, i]**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'ring_attractor', 'alpha': alpha, 'r0': r0, 'omega': omega, 'theta_rest': theta_rest_vec, 'sigma': sigma_mat}

def _simulate_multi_stable(D, dt, T, burn_ratio, seed, params):
    a = params.get('a', 1.0); b = params.get('b', 1.0); c = params.get('c', 0.5); sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1]
        drift = 2 * a * x - 4 * b * x**3
        sum_sq = np.sum(x**2)
        drift -= 2 * c * x * (sum_sq - x**2)
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = 2 * a * x - 4 * b * x**3; sum_sq = np.sum(x**2)
            d -= 2 * c * x * (sum_sq - x**2); return d
        else:
            d = 2 * a * x - 4 * b * x**3; sum_sq = np.sum(x**2, axis=1, keepdims=True)
            d -= 2 * c * x * (sum_sq - x**2); return d
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = np.sum(-a * x**2 + b * x**4)
            for i in range(D):
                for j in range(i+1, D): U += c * x[i]**2 * x[j]**2
            return U
        else:
            U = np.sum(-a * x**2 + b * x**4, axis=1)
            for i in range(D):
                for j in range(i+1, D): U += c * x[:, i]**2 * x[:, j]**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'multi_stable', 'a': a, 'b': b, 'c': c, 'sigma': sigma_mat}

def _simulate_stochastic_oscillator(D, dt, T, burn_ratio, seed, params):
    lam = params.get('lambda', 1.0); omega = params.get('omega', 1.0); theta_rest = params.get('theta_rest', 1.0); sigma = params.get('sigma', 0.5)
    sigma_mat = sigma * np.eye(D) if np.isscalar(sigma) else np.asarray(sigma)
    theta_rest_vec = np.full(max(0, D-2), theta_rest) if np.isscalar(theta_rest) else np.asarray(theta_rest)
    if seed is not None: np.random.seed(seed)
    n_steps = int(T / dt); burn = int(burn_ratio * n_steps)
    X = np.zeros((n_steps, D))
    for t in range(1, n_steps):
        x = X[t-1].copy(); drift = np.zeros(D)
        if D >= 2:
            x1, x2 = x[0], x[1]; r2 = x1**2 + x2**2
            drift[0] = lam * x1 - omega * x2 - r2 * x1
            drift[1] = lam * x2 + omega * x1 - r2 * x2
        for i in range(2, D): drift[i] = -theta_rest_vec[i-2] * x[i]
        dW = np.sqrt(dt) * np.random.randn(D)
        X[t] = x + drift * dt + sigma_mat @ dW
    data = X[burn:]
    D_diff = (sigma_mat @ sigma_mat.T) / 2.0
    def drift_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            d = np.zeros(D)
            if D >= 2:
                x1, x2 = x[0], x[1]; r2 = x1**2 + x2**2
                d[0] = lam * x1 - omega * x2 - r2 * x1
                d[1] = lam * x2 + omega * x1 - r2 * x2
            for i in range(2, D): d[i] = -theta_rest_vec[i-2] * x[i]
            return d
        else:
            d = np.zeros_like(x)
            if D >= 2:
                x1, x2 = x[:, 0], x[:, 1]; r2 = x1**2 + x2**2
                d[:, 0] = lam * x1 - omega * x2 - r2 * x1
                d[:, 1] = lam * x2 + omega * x1 - r2 * x2
            for i in range(2, D): d[:, i] = -theta_rest_vec[i-2] * x[:, i]
            return d
    def diffusion_func(x): return D_diff
    def potential_func(x):
        x = np.asarray(x)
        if x.ndim == 1:
            U = 0.0
            if D >= 2: r2 = x[0]**2 + x[1]**2; U += -0.5 * lam * r2 + 0.25 * r2**2
            for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[i]**2
            return U
        else:
            U = np.zeros(x.shape[0])
            if D >= 2: r2 = x[:, 0]**2 + x[:, 1]**2; U += -0.5 * lam * r2 + 0.25 * r2**2
            for i in range(2, D): U += 0.5 * theta_rest_vec[i-2] * x[:, i]**2
            return U
    theoretical = {
        'drift_func': drift_func, 'diffusion_func': diffusion_func, 'potential_func': potential_func,
        'drift_grid': lambda edges: _eval_drift_on_grid(drift_func, edges),
        'diffusion_grid': lambda edges: _eval_diffusion_on_grid(diffusion_func, edges),
        'potential_grid': lambda edges: _eval_potential_on_grid(potential_func, edges),
    }
    return data, theoretical, {'name': 'stochastic_oscillator', 'lambda': lam, 'omega': omega, 'theta_rest': theta_rest_vec, 'sigma': sigma_mat}

# =============================================================================
# SECCIÓN 2: ESTIMACIÓN KM (sin cambios funcionales)
# =============================================================================

def extract_km_coefficients(data, bins, p=2, bw=None, kernel='epanechnikov', dt=1.0,
                            sigma_smooth=0.0, density_threshold=0.0):
    """
    Estima coeficientes de Kramers-Moyal con post-suavizado opcional
    para eliminar artefactos de borde.
    """
    if isinstance(kernel, str):
        if kernel == 'epanechnikov':
            kernel_func = epanechnikov
        elif kernel == 'gaussian':
            kernel_func = gaussian
        else:
            raise ValueError(f"Kernel {kernel} no reconocido")
    else:
        kernel_func = kernel

    kmc, edges, bw_used, powers = km(
        data, bins=bins, powers=p, kernel=kernel_func,
        bw=bw, center_edges=True, full=True
    )
    kmc = kmc.copy()
    for idx in range(1, len(kmc)):
        kmc[idx] /= dt

    D = data.shape[1]
    sum_powers = np.sum(powers, axis=1)
    first_order_idx = np.where(sum_powers == 1)[0]
    comp_idx = [np.argmax(powers[i]) for i in first_order_idx]
    order = np.argsort(comp_idx)
    first_order_idx = first_order_idx[order]
    drift = np.array([kmc[idx] for idx in first_order_idx])

    second_order_idx = np.where(sum_powers == 2)[0]
    diffusion = np.full((D, D), None, dtype=object)
    for idx in second_order_idx:
        comb = powers[idx]
        nonzero = np.where(comb > 0)[0]
        if len(nonzero) == 1:
            i = nonzero[0]
            diffusion[i, i] = kmc[idx]
        elif len(nonzero) == 2:
            i, j = nonzero[0], nonzero[1]
            diffusion[i, j] = kmc[idx]
            diffusion[j, i] = kmc[idx]

    diff_numeric = np.full((D, D) + drift.shape[1:], np.nan, dtype=float)
    for i in range(D):
        for j in range(D):
            if diffusion[i, j] is not None:
                diff_numeric[i, j] = diffusion[i, j]
    diffusion = diff_numeric

    # ================================================================
    # POST-SUAVIZADO: elimina picos de borde por boundary bias
    # ================================================================
    if sigma_smooth > 0:
        # 'reflect' simula datos simétricos fuera del dominio, evitando
        # el sesgo de "corte" que produce los picos.
        for i in range(D):
            drift[i] = ndi.gaussian_filter(
                np.nan_to_num(drift[i], nan=0.0),
                sigma=sigma_smooth, mode='reflect'
            )
        for i in range(D):
            for j in range(D):
                if not np.all(np.isnan(diffusion[i, j])):
                    diffusion[i, j] = ndi.gaussian_filter(
                        np.nan_to_num(diffusion[i, j], nan=0.0),
                        sigma=sigma_smooth, mode='reflect'
                    )

    # ================================================================
    # MÁSCARA DE DENSIDAD: anular celdas con soporte estadístico bajo
    # ================================================================
    if density_threshold > 0:
        hist_edges = []
        for d in range(D):
            c = edges[d]
            half = (c[1] - c[0]) / 2.0 if len(c) > 1 else 0.5
            e = np.concatenate([[c[0] - half], c + half])
            hist_edges.append(e)

        density, _ = np.histogramdd(data, bins=hist_edges)
        density = density.astype(float)
        mask = density >= density_threshold * density.max()

        # Dilatamos la máscara unas pocas celdas para que el borde "sano"
        # no se contamine al difundirse hacia las celdas anuladas.
        pad = int(np.ceil(sigma_smooth)) + 1
        if D == 1:
            struct = np.ones(3)
            mask = ndi.binary_dilation(mask, structure=struct, iterations=pad)
        elif D >= 2:
            mask = ndi.binary_dilation(mask, iterations=pad)

        for i in range(D):
            drift[i] = np.where(mask, drift[i], np.nan)
        for i in range(D):
            for j in range(D):
                if not np.all(np.isnan(diffusion[i, j])):
                    diffusion[i, j] = np.where(mask, diffusion[i, j], np.nan)

    return drift, diffusion, edges

# =============================================================================
# SECCIÓN 3: RECONSTRUCCIÓN DE POTENCIAL (wrappers a IgA)
# =============================================================================

def reconstruct_potential_1D(drift, diffusion, edges, method='iga',
                             sigma_smooth=1.0, mask_density=None, density=None,
                             **iga_kwargs):
    """
    Reconstruye potencial 1D por componente (cortes del potencial D-dimensional).

    Ahora usa IgA internamente. El parámetro density es obligatorio para method='iga'.
    """
    if density is None:
        raise ValueError("reconstruct_potential_1D requiere 'density' (histograma empírico)")
    if sigma_smooth > 0 and method == 'iga':
        warnings.warn("sigma_smooth se ignora en reconstrucción IgA (el suavizado es global)", UserWarning)

    decompose_helmholtz = iga_kwargs.pop('decompose_helmholtz', False)
    result = reconstruct_potential_iga(
        drift, diffusion, edges, density,
        decompose_helmholtz=decompose_helmholtz,
        **iga_kwargs
    )
    U = result['potential']
    D = drift.shape[0]
    U_list = []
    for i in range(D):
        slice_idx = []
        for d in range(D):
            if d == i:
                slice_idx.append(slice(None))
            else:
                idx0 = np.argmin(np.abs(edges[d]))
                slice_idx.append(idx0)
        U_list.append(U[tuple(slice_idx)])
    return U_list


def reconstruct_potential(drift, diffusion, edges, sigma_smooth=1.0,
                          method='iga', mask_density=None, n_iter=5000,
                          center=True, post_smooth=0.5, density=None,
                          return_full=False, **iga_kwargs):
    """
    Reconstruye el potencial escalar U via Galerkin-B-spline (IgA).

    Parámetros legacy (sigma_smooth, post_smooth, n_iter, center) se mantienen
    para compatibilidad backward pero son ignorados por el método IgA.

    Parameters
    ----------
    drift, diffusion, edges : arrays
    density : ndarray (n1, ..., nD)
        Histograma empírico de densidad. OBLIGATORIO para method='iga'.
    return_full : bool
        Si True, retorna el dict completo del pipeline IgA (incluye eta, rank_D, etc.)
    compute_stream_function : bool, optional (via **iga_kwargs)
        Si True y D==2, calcula la función de corriente ψ, el campo
        reconstruido g_recon = ∇U + ∇⊥ψ y la fuerza no-conservativa v.
    **iga_kwargs : passed to reconstruct_potential_iga

    Returns
    -------
    U : ndarray o dict
    """
    if method != 'iga':
        warnings.warn(f"method='{method}' no soportado en v4.0. Usando 'iga'.", UserWarning)

    if density is None:
        raise ValueError("reconstruct_potential requiere 'density' (histograma empírico) para method='iga'")

    if sigma_smooth > 0:
        warnings.warn("sigma_smooth se ignora en IgA (suavizado global implícito)", UserWarning)
    if post_smooth > 0:
        warnings.warn("post_smooth se ignora en IgA (resultado ya es C^{p-1})", UserWarning)

    # Extraer decompose_helmholtz de iga_kwargs o usar default True
    decompose_helmholtz = iga_kwargs.pop('decompose_helmholtz', True)
    # Extraer compute_stream_function (default False: backward compat)
    compute_stream_function = iga_kwargs.pop('compute_stream_function', False)
    result = reconstruct_potential_iga(
        drift, diffusion, edges, density,
        decompose_helmholtz=decompose_helmholtz,
        compute_stream_function=compute_stream_function,
        **iga_kwargs
    )

    U = result['potential']
    if center:
        U_valid = U[~np.isnan(U)]
        if len(U_valid) > 0:
            U = U - np.nanmean(U)
            result['potential'] = U

    if return_full:
        return result
    return U


# =============================================================================
# SECCIÓN 4: BW ÓPTIMO (actualizado para IgA)
# =============================================================================

def _worker_bw(args):
    """Worker para optimal_bw usando IgA."""
    data, bins, bw, dt, p, kernel, sigma_smooth, density = args
    drift, diffusion, edges = extract_km_coefficients(
        data, bins=bins, p=p, bw=bw, kernel=kernel, dt=dt
    )
    result = reconstruct_potential_iga(
        drift, diffusion, edges, density=density,
        degree=3, decompose_helmholtz=False
    )
    return result['potential'], edges, bw


def optimal_bw(data, bins, dt=1.0, p=2, kernel='epanechnikov',
               sigma_smooth=1.0, n_candidates=30, n_jobs=1,
               plot=True, figsize=(16, 4.5),
               w_corr=0.35, w_stability=0.35, w_smoothness=0.30):
    """
    Busca el bw óptimo usando un criterio híbrido:
      - Correlación con -log(densidad empírica)  [35%]
      - Estabilidad del potencial ante variaciones de bw  [35%]
      - Suavidad / penalización de rangos extremos  [30%]

    El criterio no requiere ground truth y es robusto ante sobreajuste.

    Parameters
    ----------
    w_corr, w_stability, w_smoothness : float
        Pesos del score híbrido (deben sumar 1.0).
    n_candidates : int
        Número de candidatos (default 30, log-espaciados).
    """
    D = data.shape[1]
    bins_arr = np.atleast_1d(bins)

    # ── Rango de búsqueda mejorado ──────────────────────────────────
    data_range = np.ptp(data, axis=0)
    data_range[data_range == 0] = 1.0
    dx = data_range / bins_arr
    # bw mínimo: kernel debe cubrir al menos ~1.5 celdas en la dim más gruesa
    bw_min = np.max(dx) * 1.5

    std_data = np.std(data, axis=0)
    std_data = std_data[std_data > 0]
    if len(std_data) == 0:
        std_data = np.array([1.0])
    # bw máximo: usar std MÁXIMA (no mínima) para no restringir artificialmente
    bw_max = np.max(std_data) * 0.6
    if bw_max <= bw_min:
        bw_max = bw_min * 5.0

    # Candidatos log-espaciados (mejor cobertura de rangos amplios)
    bw_candidates = np.geomspace(bw_min, bw_max, n_candidates)
    print(f"[optimal_bw] Rango log-espaciado: [{bw_min:.4f}, {bw_max:.4f}] ({n_candidates} candidatos)")

    # ── Histograma base (con primer candidato) ──────────────────────
    drift0, diffusion0, edges = extract_km_coefficients(
        data, bins=bins, p=p, bw=bw_candidates[0], kernel=kernel, dt=dt
    )

    hist_edges = []
    for d in range(D):
        c = edges[d]
        if len(c) > 1:
            half = (c[1] - c[0]) / 2.0
        else:
            half = 0.5
        e = np.concatenate([[c[0] - half], c + half])
        hist_edges.append(e)

    rho, _ = np.histogramdd(data, bins=hist_edges)
    rho = rho.astype(float)
    rho_floor = np.percentile(rho[rho > 0], 1) * 0.1
    U_density = -np.log(rho + rho_floor)
    mask_density = rho > (0.05 * rho.max())
    range_logrho = np.ptp(U_density[mask_density]) if np.any(mask_density) else 1.0

    args_list = [(data, bins, bw, dt, p, kernel, sigma_smooth, rho)
                 for bw in bw_candidates]

    potentials = []
    edges_list = []
    bw_out = []

    if n_jobs != 1 and len(bw_candidates) > 1:
        max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_bw, arg): i
                       for i, arg in enumerate(args_list)}
            with tqdm(total=len(bw_candidates), desc="Scanning bw", unit="bw") as pbar:
                for future in as_completed(futures):
                    U, edges_i, bw_i = future.result()
                    potentials.append(U)
                    edges_list.append(edges_i)
                    bw_out.append(bw_i)
                    pbar.update(1)
    else:
        with tqdm(total=len(bw_candidates), desc="Scanning bw", unit="bw") as pbar:
            for arg in args_list:
                U, edges_i, bw_i = _worker_bw(arg)
                potentials.append(U)
                edges_list.append(edges_i)
                bw_out.append(bw_i)
                pbar.update(1)

    results = sorted(zip(bw_out, potentials, edges_list), key=lambda x: x[0])
    bw_candidates = np.array([r[0] for r in results])
    potentials = [r[1] for r in results]
    edges = results[0][2]

    # ── Métricas por candidato ──────────────────────────────────────
    n = len(bw_candidates)
    correlations = np.full(n, np.nan)
    stability = np.full(n, np.nan)
    smoothness = np.full(n, np.nan)
    score = np.full(n, np.nan)

    for i, U in enumerate(potentials):
        if U.shape != U_density.shape:
            continue

        valid = mask_density & ~np.isnan(U)
        n_valid = np.sum(valid)
        if n_valid < 10:
            continue

        # 1. Correlación con -log ρ
        if np.std(U[valid]) > 1e-12 and np.std(U_density[valid]) > 1e-12:
            corr, _ = pearsonr(U[valid], U_density[valid])
            correlations[i] = corr

        # 2. Estabilidad: variación respecto a vecinos
        if 0 < i < n - 1:
            # Comparar con vecino anterior y posterior (ambos deben existir)
            U_prev = potentials[i - 1]
            U_next = potentials[i + 1]
            if (U_prev.shape == U.shape and U_next.shape == U.shape and
                not np.all(np.isnan(U_prev)) and not np.all(np.isnan(U_next))):
                # Diferencias en región válida común
                valid3 = valid & ~np.isnan(U_prev) & ~np.isnan(U_next)
                if np.sum(valid3) > 10:
                    diff_prev = np.mean((U[valid3] - U_prev[valid3])**2)
                    diff_next = np.mean((U[valid3] - U_next[valid3])**2)
                    mean_diff = 0.5 * (diff_prev + diff_next)
                    # Normalizar por varianza de U para hacer adimensional
                    var_U = np.var(U[valid3])
                    if var_U > 1e-12:
                        stability[i] = np.exp(-mean_diff / var_U)
                    else:
                        stability[i] = 1.0
        elif n == 1:
            stability[i] = 1.0

        # 3. Suavidad: penalizar rangos extremos
        range_U = np.ptp(U[valid])
        if range_logrho > 1e-12 and range_U > 1e-12:
            ratio = range_U / range_logrho
            # Ideal: ratio ≈ 1 (misma escala que -log ρ)
            smoothness[i] = 1.0 / (1.0 + np.abs(np.log(ratio + 1e-12)))
        else:
            smoothness[i] = 1.0

    # ─- Score híbrido ──────────────────────────────────────────────
    # Normalizar cada métrica a [0, 1] sobre candidatos válidos
    def _normalize(arr):
        valid = ~np.isnan(arr)
        if not np.any(valid):
            return np.zeros_like(arr)
        a_min, a_max = np.nanmin(arr), np.nanmax(arr)
        if a_max - a_min < 1e-12:
            return np.where(valid, 1.0, np.nan)
        return np.where(valid, (arr - a_min) / (a_max - a_min), np.nan)

    corr_norm = _normalize(correlations)
    stab_norm = _normalize(stability)
    smooth_norm = _normalize(smoothness)

    # Combinar
    for i in range(n):
        if not (np.isnan(corr_norm[i]) or np.isnan(stab_norm[i]) or np.isnan(smooth_norm[i])):
            score[i] = (w_corr * corr_norm[i] +
                        w_stability * stab_norm[i] +
                        w_smoothness * smooth_norm[i])

    if np.any(~np.isnan(score)):
        optimal_idx = int(np.nanargmax(score))
    else:
        optimal_idx = n // 2
        warnings.warn("Ningún candidato produjo score válido. Usando candidato central.")

    # ── Plotting ────────────────────────────────────────────────────
    fig = None
    if plot:
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        ax = axes[0]
        valid = ~np.isnan(correlations)
        ax.plot(bw_candidates[valid], correlations[valid], 'bo-', markersize=5, zorder=3)
        ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Óptimo = {bw_candidates[optimal_idx]:.4f}', zorder=4)
        ax.set_xlabel('bw')
        ax.set_ylabel(r'$\rho(U, -\log \rho_{\rm emp})$')
        ax.set_title('Correlación con densidad')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.05, 1.05])

        ax = axes[1]
        valid = ~np.isnan(stability)
        ax.plot(bw_candidates[valid], stability[valid], 'gs-', markersize=5, zorder=3)
        ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2, zorder=4)
        ax.set_xlabel('bw')
        ax.set_ylabel('Estabilidad')
        ax.set_title('Estabilidad del potencial')
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        valid = ~np.isnan(score)
        ax.plot(bw_candidates[valid], score[valid], 'm^-', markersize=6, zorder=3, linewidth=2)
        ax.axvline(bw_candidates[optimal_idx], color='r', linestyle='--', linewidth=2,
                   label=f'Score máx = {score[optimal_idx]:.3f}', zorder=4)
        ax.set_xlabel('bw')
        ax.set_ylabel('Score híbrido')
        ax.set_title(f'Score = {w_corr:.0%}corr + {w_stability:.0%}stab + {w_smoothness:.0%}smooth')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        U_opt = potentials[optimal_idx]
        if D == 2:
            X, Y = np.meshgrid(edges[0], edges[1], indexing='ij')
            U_plot = np.nan_to_num(U_opt, nan=np.nanmedian(U_opt))
            levels = np.linspace(U_plot.min(), U_plot.max(), 20)
            cnt = ax.contourf(X, Y, U_plot, levels=levels, cmap='viridis')
            plt.colorbar(cnt, ax=ax, label='U')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        elif D == 1:
            ax.plot(edges[0], U_opt, 'b-', label='Potencial óptimo')
            ax.set_xlabel('x')
            ax.set_ylabel('U')
        ax.set_title(f'Potencial óptimo (bw={bw_candidates[optimal_idx]:.4f})')

        fig.tight_layout()
        plt.show()

    return {
        'bw_candidates': bw_candidates,
        'potentials': potentials,
        'edges': edges,
        'correlations': correlations,
        'stability': stability,
        'smoothness': smoothness,
        'score': score,
        'optimal_bw': float(bw_candidates[optimal_idx]),
        'optimal_idx': int(optimal_idx),
        'fig': fig
    }


# =============================================================================
# SECCIÓN 5: PLOTEO (sin cambios funcionales)
# =============================================================================

def _bbox_from_valid(valid):
    """Bounding box (x0, x1, y0, y1) en índices de las celdas True.
    Retorna None si no hay ninguna."""
    if not np.any(valid):
        return None
    xb = np.any(valid, axis=1)
    yb = np.any(valid, axis=0)
    x0 = int(np.argmax(xb))
    x1 = int(len(xb) - np.argmax(xb[::-1]))
    y0 = int(np.argmax(yb))
    y1 = int(len(yb) - np.argmax(yb[::-1]))
    return x0, x1, y0, y1


def _plot_drift_field_panel(fig, pos, F, edges, title):
    """
    Panel de deriva D¹ como campo de fuerzas.

    - D == 1: línea (fallback, no hay campo vectorial en 1D).
    - D == 2: heatmap de |F| + quiver en el plano.
    - D == 3: quiver tridimensional.
    - D >= 4: no soportado (se indica en el panel).
    """
    D = F.shape[0]

    if D == 1:
        ax = fig.add_subplot(pos)
        ax.plot(edges[0], F[0], 'b-', linewidth=2)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('Drift $D^{(1)}$')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax

    if D == 2:
        ax = fig.add_subplot(pos)
        x_c, y_c = edges
        X, Y = np.meshgrid(x_c, y_c, indexing='ij')
        mag = np.sqrt(F[0]**2 + F[1]**2)
        valid = ~np.isnan(mag)
        im = ax.contourf(X, Y, np.ma.masked_invalid(mag), levels=20,
                         cmap='viridis', alpha=0.75)
        plt.colorbar(im, ax=ax, label='$|D^{(1)}|$')
        skip = max(1, int(np.ceil(max(len(x_c), len(y_c)) / 18)))
        # Quiver solo en celdas válidas (sin puntos fantasma en la zona NaN)
        valid_s = valid[::skip, ::skip]
        ax.quiver(X[::skip, ::skip][valid_s], Y[::skip, ::skip][valid_s],
                  np.nan_to_num(F[0])[::skip, ::skip][valid_s],
                  np.nan_to_num(F[1])[::skip, ::skip][valid_s],
                  color='white', alpha=0.9)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        # Recortar al bounding box de la región con información
        bbox = _bbox_from_valid(valid)
        if bbox is not None:
            x0, x1, y0, y1 = bbox
            ax.set_xlim(x_c[x0], x_c[x1 - 1])
            ax.set_ylim(y_c[y0], y_c[y1 - 1])
        else:
            ax.set_xlim(x_c[0], x_c[-1])
            ax.set_ylim(y_c[0], y_c[-1])
        ax.set_title(title, pad=22)
        ax.set_aspect('equal', adjustable='box')
        return ax

    if D == 3:
        ax = fig.add_subplot(pos, projection='3d')
        steps = [max(1, len(e) // 6) for e in edges]
        s = tuple(slice(None, None, st) for st in steps)
        X, Y, Z = np.meshgrid(*edges, indexing='ij')
        valid = ~(np.isnan(F[0]) | np.isnan(F[1]) | np.isnan(F[2]))
        valid_s = valid[s]
        u = np.nan_to_num(F[0][s], nan=0.0)[valid_s]
        v = np.nan_to_num(F[1][s], nan=0.0)[valid_s]
        w = np.nan_to_num(F[2][s], nan=0.0)[valid_s]
        mag = np.sqrt(u**2 + v**2 + w**2)
        # Longitud de flechas proporcional a |F| (escala robusta p95)
        span = max(np.ptp(edges[0]), np.ptp(edges[1]), np.ptp(edges[2]))
        mag_ref = np.percentile(mag[mag > 0], 95) if np.any(mag > 0) else 1.0
        q = ax.quiver(X[s][valid_s], Y[s][valid_s], Z[s][valid_s], u, v, w,
                      length=0.15 * span / mag_ref, normalize=False,
                      arrow_length_ratio=0.35)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$x_3$')
        # Recortar al bounding box de la región con información
        if np.any(valid):
            for d, setlim in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
                ax_mask = np.any(valid, axis=tuple(k for k in range(3) if k != d))
                idx = np.where(ax_mask)[0]
                setlim(edges[d][idx[0]], edges[d][idx[-1]])
        ax.set_title(title)
        return ax

    ax = fig.add_subplot(pos)
    ax.text(0.5, 0.5, f'Campo de deriva no soportado para D={D}',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title)
    ax.axis('off')
    return ax


def _plot_diffusion_map_panel(fig, pos, DIFF, edges, i, j, fixed_coords,
                              title):
    """
    Panel de una componente de difusión D²(i,j) como mapa 2D.

    - D == 1: línea (fallback).
    - D == 2: mapa completo sobre (x1, x2).
    - D >= 3: corte en las dimensiones (x1, x2) fijando el resto según
      fixed_coords (default 0.0).
    """
    D = DIFF.ndim - 2
    ax = fig.add_subplot(pos)

    if D == 1:
        ax.plot(edges[0], DIFF[i, j], 'g-', linewidth=2)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel(f'$D^{{({i+1},{j+1})}}$')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return ax

    slice_idx = [slice(None), slice(None)]
    fixed_str = ''
    if D >= 3:
        fixed_vals = []
        for d in range(2, D):
            fixed_val = fixed_coords.get(d, 0.0)
            idx = int(np.argmin(np.abs(edges[d] - fixed_val)))
            slice_idx.append(idx)
            fixed_vals.append(f'$x_{d+1}$={edges[d][idx]:.2f}')
        fixed_str = '\n(' + ', '.join(fixed_vals) + ')'

    M = DIFF[i, j][tuple(slice_idx)]
    x_c, y_c = edges[0], edges[1]
    X, Y = np.meshgrid(x_c, y_c, indexing='ij')
    im = ax.contourf(X, Y, np.ma.masked_invalid(M), levels=20, cmap='magma')
    # format='%.4f' evita el texto de offset científico de la colorbar,
    # que solapa el título del panel cuando el campo es casi constante
    plt.colorbar(im, ax=ax, label=f'$D^{{({i+1},{j+1})}}$', format='%.4f')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    # Recortar al bounding box de la región con información
    bbox = _bbox_from_valid(~np.isnan(M))
    if bbox is not None:
        x0, x1, y0, y1 = bbox
        ax.set_xlim(x_c[x0], x_c[x1 - 1])
        ax.set_ylim(y_c[y0], y_c[y1 - 1])
    else:
        ax.set_xlim(x_c[0], x_c[-1])
        ax.set_ylim(y_c[0], y_c[-1])
    ax.set_title(title + fixed_str, pad=22)
    ax.set_aspect('equal', adjustable='box')
    return ax


def plot_km_components(drift, diffusion, edges,
                       drift_components=None,
                       diff_components=None,
                       fixed_coords=None,
                       theoretical=None,
                       figsize=None):
    """
    Visualización de los coeficientes de Kramers-Moyal.

    - Deriva D¹: campo de fuerzas (quiver). En 2D, flechas en el plano
      sobre un heatmap de |D¹|; en 3D, quiver tridimensional. Toda la
      información de la deriva en un solo gráfico (no por componentes).
    - Difusión D²: un mapa 2D por cada componente seleccionada en
      `diff_components` (p.ej. (1,1), (2,2), (1,2)). En D >= 3 se plotea
      el corte en (x1, x2) fijando las demás coordenadas con
      `fixed_coords` (default 0.0).

    Genera DOS figuras separadas (imágenes independientes): una para la
    deriva y otra para las difusiones. Si `theoretical` está disponible,
    cada figura tiene dos filas: arriba lo estimado y abajo el ground truth.

    Parameters
    ----------
    drift : ndarray (D, n1, ..., nD)
    diffusion : ndarray (D, D, n1, ..., nD)
    edges : list of D arrays
    drift_components : list of int or None
        Si es None o vacío, no se plotea la deriva. (Se mantiene por
        compatibilidad: el campo usa siempre todas las componentes.)
    diff_components : list of (i, j) or None
        Componentes de difusión a plotear como mapas.
    fixed_coords : dict, optional
        {dim_index: valor} para fijar dimensiones >= 2 en los cortes de
        difusión (default 0.0).
    theoretical : dict, optional
        Ground truth con 'drift_func'/'diffusion_func' (y opcionalmente
        'drift_grid'/'diffusion_grid' para evaluar directo en `edges`).
    figsize : tuple, optional
        DEPRECADO en la versión de imágenes separadas: se ignora y cada
        figura se auto-escala según el número de paneles. Se mantiene en
        la firma solo por compatibilidad.

    Returns
    -------
    figs : dict
        {'drift': Figure o None, 'diffusion': Figure o None}. Cada figura
        es independiente y debe guardarse/cerrarse por separado. La key es
        None si no se pidió ese grupo de componentes.
    """
    if fixed_coords is None:
        fixed_coords = {}
    D = drift.shape[0]

    show_drift = bool(drift_components)
    diff_components = diff_components or []
    n_diff = len(diff_components)
    if not show_drift and n_diff == 0:
        raise ValueError("Debe especificarse al menos drift_components o diff_components")

    # ------------------------------------------------------------------
    # Ground truth evaluado sobre la misma grilla (si disponible)
    # ------------------------------------------------------------------
    drift_theo = None
    diff_theo = None
    if theoretical is not None:
        try:
            if isinstance(theoretical, dict) and 'drift_grid' in theoretical:
                drift_theo = theoretical['drift_grid'](edges)
            elif isinstance(theoretical, dict) and 'drift_func' in theoretical:
                drift_theo = _eval_drift_on_grid(theoretical['drift_func'], edges)
        except Exception as e:
            warnings.warn(f"No se pudo evaluar drift teórico: {e}", UserWarning)
        try:
            if isinstance(theoretical, dict) and 'diffusion_grid' in theoretical:
                diff_theo = theoretical['diffusion_grid'](edges)
            elif isinstance(theoretical, dict) and 'diffusion_func' in theoretical:
                diff_theo = _eval_diffusion_on_grid(theoretical['diffusion_func'], edges)
        except Exception as e:
            warnings.warn(f"No se pudo evaluar difusión teórica: {e}", UserWarning)

    # ------------------------------------------------------------------
    # Recortar el ground truth a la región con información del estimado:
    # se enmascara con los NaN del reconstruido para que los tramos sin
    # datos (donde el campo teórico puede ser muy grande) no distorsionen
    # las escalas de color al comparar.
    # ------------------------------------------------------------------
    if drift_theo is not None:
        valid_drift = ~np.any(np.isnan(drift), axis=0)
        if np.any(valid_drift):
            for d in range(D):
                drift_theo[d] = np.where(valid_drift, drift_theo[d], np.nan)
    if diff_theo is not None:
        for (i, j) in diff_components:
            valid_ij = ~np.isnan(diffusion[i, j])
            if np.any(valid_ij):
                diff_theo[i, j] = np.where(valid_ij, diff_theo[i, j], np.nan)

    # ------------------------------------------------------------------
    # Layout: DOS figuras separadas — una para la deriva, otra para las
    # difusiones. En cada una: fila 0 = estimado, fila 1 = teórico
    # (si hay ground truth).
    # ------------------------------------------------------------------
    nrows = 2 if theoretical is not None else 1
    rows = [(0, drift, diffusion, 'Estimado')]
    if theoretical is not None:
        rows.append((1, drift_theo, diff_theo, 'Teórico'))

    figs = {'drift': None, 'diffusion': None}

    # ---- Figura 1: deriva (estimada vs teórica) ----
    if show_drift:
        fig_drift = plt.figure(figsize=(7.0, 6.0 * nrows))
        gs = fig_drift.add_gridspec(nrows, 1)
        for row_idx, drift_g, _, label in rows:
            if drift_g is not None:
                _plot_drift_field_panel(
                    fig_drift, gs[row_idx, 0], drift_g, edges,
                    title=f'Deriva $D^{{(1)}}$ — {label}'
                )
            else:
                ax = fig_drift.add_subplot(gs[row_idx, 0])
                ax.text(0.5, 0.5, 'No disponible', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(f'Deriva $D^{{(1)}}$ — {label}')
                ax.axis('off')
        fig_drift.tight_layout()
        figs['drift'] = fig_drift

    # ---- Figura 2: difusiones (estimadas vs teóricas) ----
    if n_diff > 0:
        fig_diff = plt.figure(figsize=(5.5 * n_diff, 5.0 * nrows))
        gs = fig_diff.add_gridspec(nrows, n_diff)
        for row_idx, _, diff_g, label in rows:
            for col, (i, j) in enumerate(diff_components):
                if diff_g is not None:
                    _plot_diffusion_map_panel(
                        fig_diff, gs[row_idx, col], diff_g, edges, i, j,
                        fixed_coords,
                        title=f'Difusión $D^{{({i+1},{j+1})}}$ — {label}'
                    )
                else:
                    ax = fig_diff.add_subplot(gs[row_idx, col])
                    ax.text(0.5, 0.5, 'No disponible', ha='center', va='center',
                            transform=ax.transAxes)
                    ax.set_title(f'Difusión $D^{{({i+1},{j+1})}}$ — {label}')
                    ax.axis('off')
        fig_diff.tight_layout()
        figs['diffusion'] = fig_diff

    return figs


def plot_potential(U_reconstructed, edges, theoretical=None, component_labels=None,
                   figsize=(10, 6), fixed_coords=None):
    if fixed_coords is None:
        fixed_coords = {}
    D = len(U_reconstructed)
    if component_labels is None:
        component_labels = [f'x{i+1}' for i in range(D)]
    fig, axes = plt.subplots(1, D, figsize=figsize, squeeze=False)
    axes = axes[0]
    for i in range(D):
        ax = axes[i]
        x = edges[i]
        U_i = U_reconstructed[i]
        ax.plot(x, U_i, 'b-', label='Reconstructed', linewidth=2)
        if theoretical is not None:
            x_eval = np.zeros((len(x), D))
            x_eval[:, i] = x
            for d in range(D):
                if d != i:
                    x_eval[:, d] = fixed_coords.get(d, 0.0)
            U_theo = theoretical['potential_func'](x_eval)
            ax.plot(x, U_theo, 'r--', label='Teórico', linewidth=2)
        ax.set_xlabel(component_labels[i])
        ax.set_ylabel('Potential U')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Potential at {component_labels[i]}')
    fig.tight_layout()
    return fig


def _crop_to_valid(U_arr, edge_list):
    """Recorta un grid 2D y sus edges a la bounding box de celdas no-NaN."""
    valid = ~np.isnan(U_arr)
    if not np.any(valid):
        return U_arr, edge_list
    x_mask = np.any(valid, axis=1)
    y_mask = np.any(valid, axis=0)
    x_start = np.argmax(x_mask)
    x_end   = len(x_mask) - np.argmax(x_mask[::-1])
    y_start = np.argmax(y_mask)
    y_end   = len(y_mask) - np.argmax(y_mask[::-1])
    U_crop = U_arr[x_start:x_end, y_start:y_end]
    edges_crop = [edge_list[0][x_start:x_end], edge_list[1][y_start:y_end]]
    return U_crop, edges_crop


def _valid_bbox(U_arr):
    """Bounding box (x_start, x_end, y_start, y_end) de celdas no-NaN."""
    valid = ~np.isnan(U_arr)
    if not np.any(valid):
        return 0, U_arr.shape[0], 0, U_arr.shape[1]
    x_mask = np.any(valid, axis=1)
    y_mask = np.any(valid, axis=0)
    x_start = np.argmax(x_mask)
    x_end = len(x_mask) - np.argmax(x_mask[::-1])
    y_start = np.argmax(y_mask)
    y_end = len(y_mask) - np.argmax(y_mask[::-1])
    return x_start, x_end, y_start, y_end


def _overlay_streamlines(ax, x_centers, y_centers, g1, g2, stream_color='white',
                         stream_density=2.0, linewidth=1.2):
    """Superpone streamlines del campo (g1, g2) sobre un eje, tolerando NaN.

    g1, g2 usan orientación (nx, ny) — igual que las mallas X, Y creadas con
    meshgrid(indexing='ij') en este módulo; streamplot requiere (ny, nx),
    de ahí la transposición.
    """
    valid = ~np.isnan(g1) & ~np.isnan(g2)
    if not np.any(valid):
        return
    ax.streamplot(x_centers, y_centers,
                  np.nan_to_num(g1).T, np.nan_to_num(g2).T,
                  color=stream_color, density=stream_density,
                  linewidth=linewidth, arrowstyle='->', arrowsize=1.0)


# def plot_potential_2d(U, edges, U_theoretical=None, title_est="Reconstructed Potential",
#                       title_theo="Theoretical Potential", levels=50, cmap='viridis',
#                       figsize=(12, 5), unify_colorbar=False, align_minima=False,
#                       align_to_zero=False, crop_to_valid=False):
#     """
#     Plot 2D potential. If crop_to_valid=True, automatically zooms to the
#     region where U is not NaN (useful when density_threshold masks large areas).
#     """
#     if U.ndim != 2:
#         raise ValueError("This function is only for 2D potentials.")

#     U_plot = U.copy()
#     edges_plot = [e.copy() for e in edges]

#     if crop_to_valid:
#         U_plot, edges_plot = _crop_to_valid(U_plot, edges_plot)

#     x_centers, y_centers = edges_plot
#     X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

#     if U_theoretical is not None:
#         # Resolvemos la función teórica si viene como dict
#         if isinstance(U_theoretical, dict) and 'potential_grid' in U_theoretical:
#             U_theo = U_theoretical['potential_grid'](edges).copy()
#         else:
#             U_theo = U_theoretical.copy()

#         if crop_to_valid:
#             valid = ~np.isnan(U)
#             x_mask = np.any(valid, axis=1)
#             y_mask = np.any(valid, axis=0)
#             x_start = np.argmax(x_mask); x_end = len(x_mask) - np.argmax(x_mask[::-1])
#             y_start = np.argmax(y_mask); y_end = len(y_mask) - np.argmax(y_mask[::-1])
#             U_theo = U_theo[x_start:x_end, y_start:y_end]

#         # --- Alineación de mínimos hacia Z=0 ---
#         if align_to_zero:
#             align_minima = False
#             if np.any(~np.isnan(U_plot)):
#                 U_plot = U_plot - np.nanmin(U_plot)
#             if np.any(~np.isnan(U_theo)):
#                 U_theo = U_theo - np.nanmin(U_theo)

#         # --- Alineación de mínimos (desplazamiento en Z) ---
#         if align_minima:
#             if np.any(~np.isnan(U_plot)) and np.any(~np.isnan(U_theo)):
#                 min_rec  = np.nanmin(U_plot)
#                 min_theo = np.nanmin(U_theo)
#                 offset   = min_theo - min_rec
#                 U_plot   = U_plot + offset

#         # --- Unificación de colorbar ---
#         if unify_colorbar:
#             vmin = min(np.nanmin(U_plot), np.nanmin(U_theo))
#             vmax = max(np.nanmax(U_plot), np.nanmax(U_theo))
#             contour_levels = np.linspace(vmin, vmax, levels)
#         else:
#             vmin = None
#             vmax = None
#             contour_levels = levels

#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

#         contour1 = ax1.contourf(X, Y, U_plot, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax)
#         plt.colorbar(contour1, ax=ax1, label='Potential U')
#         ax1.set_xlabel('x')
#         ax1.set_ylabel('y')
#         ax1.set_title(title_est)
#         ax1.axis('equal')

#         contour2 = ax2.contourf(X, Y, U_theo, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax)
#         plt.colorbar(contour2, ax=ax2, label='Potential U')
#         ax2.set_xlabel('x')
#         ax2.set_ylabel('y')
#         ax2.set_title(title_theo)
#         ax2.axis('equal')

#     else:
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#         contour = ax.contourf(X, Y, U_plot, levels=levels, cmap=cmap)
#         plt.colorbar(contour, ax=ax, label='Potential U')
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_title(title_est)
#         ax.axis('equal')

#     fig.tight_layout()
#     return fig

def plot_potential_2d(U, edges, U_theoretical=None, title_est="Reconstructed Potential",
                      title_theo="Theoretical Potential", levels=50, cmap='viridis',
                      figsize=(12, 5), unify_colorbar=False, align_minima=False,
                      align_to_zero=False, crop_to_valid=False, clip_percentile=None,
                      stream_function=None, reconstructed_field=None,
                      reconstructed_drift=None,
                      show_streamlines=True, stream_color='white',
                      stream_density=2.0):
    """
    Plot 2D potential. If crop_to_valid=True, automatically zooms to the
    region where U is not NaN (useful when density_threshold masks large areas).

    Parameters (nuevos)
    -------------------
    stream_function : ndarray (n1, n2) or None
        Función de corriente ψ (reservado para extensiones; no requerido para
        las streamlines).
    reconstructed_field : ndarray (2, n1, n2) or None
        Campo completo g_recon = ∇U + ∇⊥ψ. Si está disponible y
        show_streamlines=True, se superponen streamlines sobre el heatmap
        del potencial reconstruido. OJO: g = -D⁻¹f, así que estas
        streamlines ASCIENDEN el potencial (salen de los mínimos).
    reconstructed_drift : ndarray (2, n1, n2) or None
        Drift reconstruido f_recon = -D·g_recon. Si se provee, tiene
        prioridad sobre reconstructed_field para las streamlines y muestra
        la dirección FÍSICA de la dinámica (desciende U, converge a
        atractores).
    show_streamlines : bool
        Si True y hay campo disponible, mostrar streamlines.
    stream_color : str
        Color de las streamlines (default 'white').
    stream_density : float
        Densidad de streamlines (default 2.0).
    """
    if U.ndim != 2:
        raise ValueError("This function is only for 2D potentials.")

    U_plot = U.copy()
    edges_plot = [e.copy() for e in edges]

    # Bounding box de la región válida (para recortar el campo reconstruido
    # de forma consistente con el recorte de U).
    if crop_to_valid:
        x0, x1, y0, y1 = _valid_bbox(U_plot)
        U_plot, edges_plot = _crop_to_valid(U_plot, edges_plot)

    # --- Guardia: si el crop dejó menos de 2x2, contourf no puede ---
    if U_plot.shape[0] < 2 or U_plot.shape[1] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        n_valid = int(np.sum(~np.isnan(U_plot)))
        ax.text(
            0.5, 0.5,
            f"Insufficient valid data for 2D contour\n"
            f"(cropped shape: {U_plot.shape}, valid cells: {n_valid})",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="gray",
        )
        ax.set_title(title_est)
        fig.tight_layout()
        return fig

    # Campo para las streamlines: el drift (dirección física) tiene prioridad
    # sobre g_recon (dirección de ascenso, g = -D⁻¹f).
    stream_field = reconstructed_drift if reconstructed_drift is not None \
        else reconstructed_field

    # Preparar componentes del campo (si disponible)
    g1 = g2 = None
    if show_streamlines and stream_field is not None:
        g_arr = np.asarray(stream_field)
        if g_arr.shape[0] == 2 and g_arr.shape[1:] == U.shape:
            g1 = g_arr[0].copy()
            g2 = g_arr[1].copy()
            if crop_to_valid:
                g1 = g1[x0:x1, y0:y1]
                g2 = g2[x0:x1, y0:y1]

    x_centers, y_centers = edges_plot
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

    if U_theoretical is not None:
        # Resolvemos la función teórica si viene como dict
        if isinstance(U_theoretical, dict) and 'potential_grid' in U_theoretical:
            U_theo = U_theoretical['potential_grid'](edges).copy()
        else:
            U_theo = U_theoretical.copy()

        if crop_to_valid:
            valid = ~np.isnan(U)
            x_mask = np.any(valid, axis=1)
            y_mask = np.any(valid, axis=0)
            x_start = np.argmax(x_mask); x_end = len(x_mask) - np.argmax(x_mask[::-1])
            y_start = np.argmax(y_mask); y_end = len(y_mask) - np.argmax(y_mask[::-1])
            U_theo = U_theo[x_start:x_end, y_start:y_end]

        # --- Alineación de mínimos hacia Z=0 ---
        if align_to_zero:
            align_minima = False
            if np.any(~np.isnan(U_plot)):
                U_plot = U_plot - np.nanmin(U_plot)
            if np.any(~np.isnan(U_theo)):
                U_theo = U_theo - np.nanmin(U_theo)

        # --- Alineación de mínimos (desplazamiento en Z) ---
        if align_minima:
            if np.any(~np.isnan(U_plot)) and np.any(~np.isnan(U_theo)):
                min_rec  = np.nanmin(U_plot)
                min_theo = np.nanmin(U_theo)
                offset   = min_theo - min_rec
                U_plot   = U_plot + offset

        # --- Determinación de vmin/vmax para cada plot ---
        if clip_percentile is not None:
            if unify_colorbar:
                # Unificar rango recortado sobre ambos conjuntos
                valid_rec  = U_plot[~np.isnan(U_plot)]
                valid_theo = U_theo[~np.isnan(U_theo)]
                all_valid  = np.concatenate([valid_rec, valid_theo])
                if len(all_valid) > 0:
                    vmin = np.percentile(all_valid, 100 - clip_percentile)
                    vmax = np.percentile(all_valid, clip_percentile)
                else:
                    vmin = None; vmax = None
                vmin1 = vmin2 = vmin
                vmax1 = vmax2 = vmax
            else:
                # Cada plot con su propio recorte independiente
                valid_rec = U_plot[~np.isnan(U_plot)]
                if len(valid_rec) > 0:
                    vmin1 = np.percentile(valid_rec, 100 - clip_percentile)
                    vmax1 = np.percentile(valid_rec, clip_percentile)
                else:
                    vmin1 = None; vmax1 = None

                valid_theo = U_theo[~np.isnan(U_theo)]
                if len(valid_theo) > 0:
                    vmin2 = np.percentile(valid_theo, 100 - clip_percentile)
                    vmax2 = np.percentile(valid_theo, clip_percentile)
                else:
                    vmin2 = None; vmax2 = None
        elif unify_colorbar:
            vmin = min(np.nanmin(U_plot), np.nanmin(U_theo))
            vmax = max(np.nanmax(U_plot), np.nanmax(U_theo))
            vmin1 = vmin2 = vmin
            vmax1 = vmax2 = vmax
        else:
            vmin1 = vmax1 = None
            vmin2 = vmax2 = None

        # --- Construcción de levels ---
        if vmin1 is not None and vmax1 is not None:
            levels1 = np.linspace(vmin1, vmax1, levels)
        else:
            levels1 = levels
        if vmin2 is not None and vmax2 is not None:
            levels2 = np.linspace(vmin2, vmax2, levels)
        else:
            levels2 = levels

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        contour1 = ax1.contourf(X, Y, U_plot, levels=levels1, cmap=cmap, vmin=vmin1, vmax=vmax1)
        # NUEVO: streamlines del campo completo g_recon = ∇U + ∇⊥ψ (si disponible)
        if g1 is not None and g1.shape == X.shape:
            _overlay_streamlines(ax1, x_centers, y_centers, g1, g2,
                                 stream_color=stream_color,
                                 stream_density=stream_density)
        plt.colorbar(contour1, ax=ax1, label='Potential U')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(title_est)
        ax1.axis('equal')

        contour2 = ax2.contourf(X, Y, U_theo, levels=levels2, cmap=cmap, vmin=vmin2, vmax=vmax2)
        plt.colorbar(contour2, ax=ax2, label='Potential U')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(title_theo)
        ax2.axis('equal')

    else:
        # --- Caso sin teórico: clipping independiente ---
        if clip_percentile is not None:
            valid_data = U_plot[~np.isnan(U_plot)]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 100 - clip_percentile)
                vmax = np.percentile(valid_data, clip_percentile)
                contour_levels = np.linspace(vmin, vmax, levels)
            else:
                vmin = None; vmax = None; contour_levels = levels
        else:
            vmin = None; vmax = None; contour_levels = levels

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        contour = ax.contourf(X, Y, U_plot, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax)
        # NUEVO: streamlines del campo completo g_recon = ∇U + ∇⊥ψ (si disponible)
        if g1 is not None and g1.shape == X.shape:
            _overlay_streamlines(ax, x_centers, y_centers, g1, g2,
                                 stream_color=stream_color,
                                 stream_density=stream_density)
        plt.colorbar(contour, ax=ax, label='Potential U')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title_est)
        ax.axis('equal')

    fig.tight_layout()
    return fig


def plot_potential_slice(U, edges, dims=(0, 1), fixed_coords=None,
                         U_theoretical=None, title="Potencial (corte 2D)",
                         levels=50, cmap='viridis', figsize=(10, 4.5),
                         unify_colorbar=False, align_minima=False,
                         align_to_zero=False, crop_to_valid=False,
                         stream_function=None, reconstructed_field=None,
                         reconstructed_drift=None,
                         show_streamlines=True, stream_color='white',
                         stream_density=2.0):
    """
    Visualiza un corte 2D de un potencial D-dimensional.
    Si crop_to_valid=True, recorta automáticamente a la región con datos válidos.

    Parameters (nuevos)
    -------------------
    stream_function : ndarray or None
        Función de corriente ψ (reservado para extensiones).
    reconstructed_field : ndarray (D, n1, ..., nD) or None
        Campo completo g_recon = ∇U + ∇⊥ψ. Si se provee, se extrae el slice
        correspondiente (fijando las mismas coordenadas que para U) y se
        superponen streamlines de las componentes (dims[0], dims[1]).
        OJO: estas streamlines ASCIENDEN el potencial (g = -D⁻¹f).
    reconstructed_drift : ndarray (D, n1, ..., nD) or None
        Drift reconstruido f_recon = -D·g_recon. Si se provee, tiene
        prioridad sobre reconstructed_field y las streamlines siguen la
        dirección FÍSICA de la dinámica (descienden U).
    show_streamlines : bool
        Si True y hay campo disponible, mostrar streamlines.
    stream_color : str
        Color de las streamlines (default 'white').
    stream_density : float
        Densidad de streamlines (default 2.0).
    """
    D = len(edges)
    i, j = dims
    if i == j or i >= D or j >= D:
        raise ValueError(f"dims={dims} inválido para D={D}")

    if fixed_coords is None:
        fixed_coords = {}

    # --- 1. Extraer corte 2D del potencial reconstruido ---
    slice_idx = []
    for d in range(D):
        if d == i or d == j:
            slice_idx.append(slice(None))
        else:
            fixed_val = fixed_coords.get(d, 0.0)
            idx = np.argmin(np.abs(edges[d] - fixed_val))
            slice_idx.append(idx)

    U_slice = U[tuple(slice_idx)]

    # --- 1b. Extraer el mismo corte del campo para streamlines ---
    # El drift (dirección física) tiene prioridad sobre g_recon (ascenso).
    stream_field = reconstructed_drift if reconstructed_drift is not None \
        else reconstructed_field
    g1 = g2 = None
    if show_streamlines and stream_field is not None:
        g_arr = np.asarray(stream_field)
        if g_arr.shape[0] == D and g_arr.shape[1:] == U.shape:
            # Las componentes del slice 2D son las direcciones (i, j) del campo
            g1 = g_arr[i][tuple(slice_idx)]
            g2 = g_arr[j][tuple(slice_idx)]

    # --- 2. Recortar a región válida (opcional) ---
    x = edges[i].copy()
    y = edges[j].copy()
    if crop_to_valid:
        x0, x1, y0, y1 = _valid_bbox(U_slice)
        U_slice, (x, y) = _crop_to_valid(U_slice, [x, y])
        if g1 is not None:
            g1 = g1[x0:x1, y0:y1]
            g2 = g2[x0:x1, y0:y1]

    # --- Guardia: si el crop dejó menos de 2x2, contourf no puede ---
    if U_slice.shape[0] < 2 or U_slice.shape[1] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        n_valid = int(np.sum(~np.isnan(U_slice)))
        ax.text(
            0.5, 0.5,
            f"Insufficient valid data for 2D contour\n"
            f"(slice cropped shape: {U_slice.shape}, valid cells: {n_valid})",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="gray",
        )
        ax.set_title(title)
        fig.tight_layout()
        return fig

    X, Y = np.meshgrid(x, y, indexing='ij')

    # --- 3. Extraer corte 2D del teórico (si existe) ---
    U_theo_slice = None
    if U_theoretical is not None:
        if isinstance(U_theoretical, dict) and 'potential_grid' in U_theoretical:
            U_theo_full = U_theoretical['potential_grid'](edges)
            U_theo_slice = U_theo_full[tuple(slice_idx)]
        elif isinstance(U_theoretical, np.ndarray):
            U_theo_slice = U_theoretical[tuple(slice_idx)]
        elif callable(U_theoretical):
            U_theo_full = U_theoretical(edges)
            if isinstance(U_theo_full, np.ndarray):
                U_theo_slice = U_theo_full[tuple(slice_idx)]
        else:
            warnings.warn("U_theoretical no es ndarray, dict ni callable. Ignorando.", UserWarning)

        if crop_to_valid and U_theo_slice is not None:
            valid = ~np.isnan(U_slice)
            x_mask = np.any(valid, axis=1)
            y_mask = np.any(valid, axis=0)
            x_start = np.argmax(x_mask); x_end = len(x_mask) - np.argmax(x_mask[::-1])
            y_start = np.argmax(y_mask); y_end = len(y_mask) - np.argmax(y_mask[::-1])
            U_theo_slice = U_theo_slice[x_start:x_end, y_start:y_end]

    # --- 4. Alineación de potenciales ---
    U_plot = U_slice.copy()
    if U_theo_slice is not None:
        U_theo_plot = U_theo_slice.copy()

    if align_to_zero:
        align_minima = False
        if np.any(~np.isnan(U_plot)):
            U_plot = U_plot - np.nanmin(U_plot)
        if U_theo_slice is not None and np.any(~np.isnan(U_theo_plot)):
            U_theo_plot = U_theo_plot - np.nanmin(U_theo_plot)

    if align_minima and U_theo_slice is not None:
        if np.any(~np.isnan(U_plot)) and np.any(~np.isnan(U_theo_plot)):
            min_rec = np.nanmin(U_plot)
            min_theo = np.nanmin(U_theo_plot)
            offset = min_theo - min_rec
            U_plot = U_plot + offset

    # --- 5. Unificación de colorbar ---
    if unify_colorbar and U_theo_slice is not None:
        vmin = min(np.nanmin(U_plot), np.nanmin(U_theo_plot))
        vmax = max(np.nanmax(U_plot), np.nanmax(U_theo_plot))
        contour_levels = np.linspace(vmin, vmax, levels)
    else:
        vmin = None
        vmax = None
        contour_levels = levels

    # --- 6. Plotting ---
    if U_theo_slice is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        cnt1 = ax1.contourf(X, Y, U_plot, levels=contour_levels, cmap=cmap,
                            vmin=vmin, vmax=vmax)
        # NUEVO: streamlines del campo completo en el slice (si disponible)
        if g1 is not None and g1.shape == X.shape:
            _overlay_streamlines(ax1, x, y, g1, g2,
                                 stream_color=stream_color,
                                 stream_density=stream_density)
        plt.colorbar(cnt1, ax=ax1, label='U')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlabel(f'x{i+1}')
        ax1.set_ylabel(f'x{j+1}')
        ax1.set_title(title)

        cnt2 = ax2.contourf(X, Y, U_theo_plot, levels=contour_levels, cmap=cmap,
                            vmin=vmin, vmax=vmax)
        plt.colorbar(cnt2, ax=ax2, label='U')
        ax2.set_aspect('equal', adjustable='box')
        ax2.set_xlabel(f'x{i+1}')
        ax2.set_ylabel(f'x{j+1}')
        ax2.set_title("Theoretical")

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        cnt = ax.contourf(X, Y, U_plot, levels=contour_levels, cmap=cmap,
                          vmin=vmin, vmax=vmax)
        # NUEVO: streamlines del campo completo en el slice (si disponible)
        if g1 is not None and g1.shape == X.shape:
            _overlay_streamlines(ax, x, y, g1, g2,
                                 stream_color=stream_color,
                                 stream_density=stream_density)
        plt.colorbar(cnt, ax=ax, label='U')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'x{i+1}')
        ax.set_ylabel(f'x{j+1}')
        ax.set_title(title)

    fig.tight_layout()
    return fig

# def plot_potential_slice(U, edges, dims=(0, 1), fixed_coords=None,
#                          U_theoretical=None, title="Potencial (corte 2D)",
#                          levels=50, cmap='viridis', figsize=(10, 4.5)):
#     """
#     Visualiza un corte 2D de un potencial D-dimensional fijando las otras
#     dimensiones en valores constantes (default: 0 o coordenadas especificadas).

#     Parameters
#     ----------
#     U : ndarray, shape (n1, ..., nD)
#         Potencial D-dimensional.
#     edges : list of D arrays
#         Centros de bins por dimensión.
#     dims : tuple (i, j)
#         Índices de las dos dimensiones a plotear.
#     fixed_coords : dict, optional
#         {dim_index: valor} para dimensiones fijas. Si None, usa 0.
#     U_theoretical : ndarray o callable, optional
#         Potencial teórico del mismo shape, o callable(edges) -> grid.
#     """
#     D = len(edges)
#     i, j = dims
#     if i == j or i >= D or j >= D:
#         raise ValueError(f"dims={dims} inválido para D={D}")

#     if fixed_coords is None:
#         fixed_coords = {}

#     # Construir índices de slice para extraer el corte 2D
#     slice_idx = []
#     for d in range(D):
#         if d == i:
#             slice_idx.append(slice(None))
#         elif d == j:
#             slice_idx.append(slice(None))
#         else:
#             fixed_val = fixed_coords.get(d, 0.0)
#             idx = np.argmin(np.abs(edges[d] - fixed_val))
#             slice_idx.append(idx)

#     U_slice = U[tuple(slice_idx)]

#     x = edges[i]
#     y = edges[j]
#     X, Y = np.meshgrid(x, y, indexing='ij')

#     if U_theoretical is not None:
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
#         for ax, U_plot, ttl in [(ax1, U_slice, title),
#                                  (ax2, U_theoretical[tuple(slice_idx)] if isinstance(U_theoretical, np.ndarray) else U_theoretical, "Teórico")]:
#             if isinstance(U_plot, np.ndarray):
#                 #U_plot = np.nan_to_num(U_plot, nan=np.nanmedian(U_plot))
#                 cnt = ax.contourf(X, Y, U_plot, levels=levels, cmap=cmap)
#                 plt.colorbar(cnt, ax=ax, label='U')
#             ax.set_aspect('equal', adjustable='box')
#             ax.set_xlabel(f'x{i+1}')
#             ax.set_ylabel(f'x{j+1}')
#             ax.set_title(ttl)
#     else:
#         fig, ax = plt.subplots(1, 1, figsize=figsize)
#         #U_plot = np.nan_to_num(U_slice, nan=np.nanmedian(U_slice))
#         cnt = ax.contourf(X, Y, U_slice, levels=levels, cmap=cmap)
#         plt.colorbar(cnt, ax=ax, label='U')
#         ax.set_aspect('equal', adjustable='box')
#         ax.set_xlabel(f'x{i+1}')
#         ax.set_ylabel(f'x{j+1}')
#         ax.set_title(title)

#     fig.tight_layout()
#     plt.show()
#     return fig


# =============================================================================
# SECCIÓN 5b: PLOTS DE NO-EQUILIBRIO (v, ψ y campo reconstruido)
# =============================================================================

def plot_nonconservative_force_2d(v_field, edges,
                                  title='Fuerza no-conservativa v = f + D·∇U',
                                  cmap='plasma', levels=20, skip=6,
                                  scale=None, width=0.004, quiver_color='white',
                                  figsize=(8, 7), crop_to_valid=False):
    """
    Plot de la fuerza no-conservativa v (un solo panel).

    Muestra |v| como heatmap y el campo vectorial v como quiver submuestreado.
    Es el plot más informativo físicamente: indica "qué empuja al sistema
    fuera del equilibrio". Para un ring attractor se ven flechas circulares.

    Parameters
    ----------
    v_field : ndarray (2, n1, n2)
        Fuerza no-conservativa v = f + D·∇U (key 'nonconservative_force'
        del dict retornado por reconstruct_potential).
    edges : list of 2 arrays
        Centros de bins por dirección.
    skip : int
        Submuestreo del quiver (default 6).
    scale, width : float
        Parámetros de matplotlib quiver. scale=None (default) = auto-escala.
    crop_to_valid : bool
        Si True, recorta a la bounding box de celdas con |v| no-NaN.

    Returns
    -------
    fig : matplotlib Figure
    """
    v = np.asarray(v_field)
    if v.shape[0] != 2:
        raise ValueError("v_field debe tener shape (2, n1, n2)")

    v1, v2 = v[0], v[1]
    v_mag = np.sqrt(v1**2 + v2**2)
    edges_plot = [e.copy() for e in edges]

    if crop_to_valid:
        v_mag, edges_plot = _crop_to_valid(v_mag, edges_plot)
        x0, x1, y0, y1 = _valid_bbox(np.sqrt(np.asarray(v_field[0])**2 +
                                             np.asarray(v_field[1])**2))
        v1 = v1[x0:x1, y0:y1]
        v2 = v2[x0:x1, y0:y1]

    # --- Guardia: si el crop dejó menos de 2x2, contourf no puede ---
    if v_mag.shape[0] < 2 or v_mag.shape[1] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            f"Insufficient valid data for 2D contour\n"
            f"(v_mag cropped shape: {v_mag.shape})",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="gray",
        )
        ax.set_title(title)
        fig.tight_layout()
        return fig

    X, Y = np.meshgrid(edges_plot[0], edges_plot[1], indexing='ij')

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.contourf(X, Y, np.nan_to_num(v_mag, nan=0.0), levels=levels, cmap=cmap)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              np.nan_to_num(v1)[::skip, ::skip], np.nan_to_num(v2)[::skip, ::skip],
              color=quiver_color, alpha=0.8, scale=scale, width=width)
    plt.colorbar(im, ax=ax, label='|v|')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.axis('equal')
    fig.tight_layout()
    return fig


def plot_potential_combined_2d(U, edges, stream_function=None,
                               nonconservative_force=None,
                               reconstructed_field=None,
                               reconstructed_drift=None,
                               title='U (fondo) + v (flechas rojas) + streamlines (blanco)',
                               levels=25, cmap='viridis', skip=8,
                               scale=None, width=0.004,
                               stream_color='white', stream_density=1.5,
                               psi_levels=10, figsize=(9, 8),
                               crop_to_valid=False,
                               clip_percentile=None):
    """
    Plot combinado del mecanismo físico completo (un solo panel).

    - Fondo: heatmap del potencial U (el "riel").
    - Superposición opcional: isolíneas de ψ (gris).
    - Superposición opcional: fuerza no-conservativa v como quiver (rojo),
      el "empuje tangencial".
    - Superposición opcional: streamlines del campo completo reconstruido
      g_recon = ∇U + ∇⊥ψ (blanco), las trayectorias reales.

    Parameters
    ----------
    U : ndarray (n1, n2)
    edges : list of 2 arrays
    stream_function : ndarray (n1, n2) or None
    nonconservative_force : ndarray (2, n1, n2) or None
    reconstructed_field : ndarray (2, n1, n2) or None
        Campo g_recon = ∇U + ∇⊥ψ para las streamlines. OJO: asciende U
        (g = -D⁻¹f). Se usa solo si reconstructed_drift es None.
    reconstructed_drift : ndarray (2, n1, n2) or None
        Drift reconstruido f_recon = -D·g_recon. Si se provee, tiene
        prioridad para las streamlines y muestra la dirección FÍSICA
        (desciende U, converge a atractores).
    skip : int
        Submuestreo del quiver de v (default 8).
    crop_to_valid : bool
        Si True, recorta todo a la bounding box de celdas no-NaN de U.

    Returns
    -------
    fig : matplotlib Figure
    """
    if U.ndim != 2:
        raise ValueError("This function is only for 2D potentials.")

    U_plot = U.copy()
    edges_plot = [e.copy() for e in edges]
    psi_plot = None if stream_function is None else np.asarray(stream_function).copy()
    v1 = v2 = g1 = g2 = None
    if nonconservative_force is not None:
        v_arr = np.asarray(nonconservative_force)
        if v_arr.shape[0] == 2 and v_arr.shape[1:] == U.shape:
            v1, v2 = v_arr[0].copy(), v_arr[1].copy()
    # El drift (dirección física) tiene prioridad sobre g_recon (ascenso)
    stream_field = reconstructed_drift if reconstructed_drift is not None \
        else reconstructed_field
    if stream_field is not None:
        g_arr = np.asarray(stream_field)
        if g_arr.shape[0] == 2 and g_arr.shape[1:] == U.shape:
            g1, g2 = g_arr[0].copy(), g_arr[1].copy()

    if crop_to_valid:
        x0, x1, y0, y1 = _valid_bbox(U_plot)
        U_plot, edges_plot = _crop_to_valid(U_plot, edges_plot)
        if psi_plot is not None:
            psi_plot = psi_plot[x0:x1, y0:y1]
        if v1 is not None:
            v1 = v1[x0:x1, y0:y1]
            v2 = v2[x0:x1, y0:y1]
        if g1 is not None:
            g1 = g1[x0:x1, y0:y1]
            g2 = g2[x0:x1, y0:y1]

    # --- Guardia: si el crop dejó menos de 2x2, contourf no puede ---
    if U_plot.shape[0] < 2 or U_plot.shape[1] < 2:
        fig, ax = plt.subplots(figsize=figsize)
        n_valid = int(np.sum(~np.isnan(U_plot)))
        ax.text(
            0.5, 0.5,
            f"Insufficient valid data for 2D contour\n"
            f"(combined cropped shape: {U_plot.shape}, valid cells: {n_valid})",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="gray",
        )
        ax.set_title(title)
        fig.tight_layout()
        return fig

    if clip_percentile is not None:
        valid_data = U_plot[~np.isnan(U_plot)]
        if len(valid_data) > 0:
            vmin = np.percentile(valid_data, 100 - clip_percentile)
            vmax = np.percentile(valid_data, clip_percentile)
            levels = np.linspace(vmin, vmax, levels)
        else:
            vmin = None; vmax = None; levels = levels
    else:
        vmin = None; vmax = None; levels = levels

    X, Y = np.meshgrid(edges_plot[0], edges_plot[1], indexing='ij')

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.contourf(X, Y, U_plot, levels=levels, cmap=cmap, alpha=0.7, vmin=vmin, vmax=vmax)

    # Isolíneas de ψ (opcional, en gris)
    if psi_plot is not None and psi_plot.shape == X.shape and np.any(~np.isnan(psi_plot)):
        ax.contour(X, Y, psi_plot, levels=psi_levels, colors='gray',
                   linewidths=0.8, alpha=0.5)

    # Fuerza no-conservativa v como quiver submuestreado (rojo)
    if v1 is not None and v1.shape == X.shape:
        ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                  np.nan_to_num(v1)[::skip, ::skip], np.nan_to_num(v2)[::skip, ::skip],
                  color='red', alpha=0.7, scale=scale, width=width)

    # Streamlines del campo completo reconstruido (blanco)
    if g1 is not None and g1.shape == X.shape:
        _overlay_streamlines(ax, edges_plot[0], edges_plot[1], g1, g2,
                             stream_color=stream_color,
                             stream_density=stream_density,
                             linewidth=1.0)

    plt.colorbar(im, ax=ax, label='U', shrink=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.axis('equal')
    fig.tight_layout()
    return fig

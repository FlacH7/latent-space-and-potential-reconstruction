"""
iga_reconstructor.py
===================
Módulo de Análisis Isogeométrico (IgA) para reconstrucción de potenciales
estocásticos vía Galerkin con B-splines tensoriales.

Resuelve los puntos de dolor:
  1. Descomposición de Helmholtz explícita (proyección ortogonal sobre gradientes).
  2. Integración global variacional (Galerkin) en lugar de BFS secuencial.
  3. Pseudo-inversa SVD adaptativa para inversión robusta de D(x).

Compatible con D = 1..5, bins no equidistantes, y datos sin ground truth.
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy import sparse, ndimage
from scipy.sparse.linalg import spsolve, cg, spilu, LinearOperator
import itertools
import warnings

# =============================================================================
# PUNTO DE DOLOR 3: Inversión Robusta de D via SVD truncado adaptativo
# =============================================================================

def compute_g_robust(drift, diffusion, density_mask=None,
                     rtol=1e-6, atol=1e-12, return_diagnostics=True):
    """
    Calcula el campo efectivo g = -D(x)^+ f(x) usando la pseudo-inversa de
    Moore-Penrose con truncamiento adaptativo de autovalores (SVD).

    Parameters
    ----------
    drift : ndarray, shape (D, n1, ..., nD)
    diffusion : ndarray, shape (D, D, n1, ..., nD)
    density_mask : ndarray bool, optional
    rtol, atol : tolerancias para truncamiento SVD

    Returns
    -------
    g : ndarray, shape (D, n1, ..., nD)
    rank_D : ndarray int (si return_diagnostics=True)
    condition_D : ndarray float (si return_diagnostics=True)
    """
    D = drift.shape[0]
    grid_shape = drift.shape[1:]
    g = np.full(drift.shape, np.nan, dtype=float)
    rank_D = np.zeros(grid_shape, dtype=int)
    condition_D = np.full(grid_shape, np.inf, dtype=float)

    for idx in np.ndindex(grid_shape):
        if density_mask is not None and not density_mask[idx]:
            continue

        Dmat = diffusion[(slice(None), slice(None)) + idx]
        fvec = drift[(slice(None),) + idx]

        # SVD de Dmat (simétrica, pero usamos SVD general para robustez)
        try:
            U_svd, s, Vt = np.linalg.svd(Dmat, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        sigma_max = s[0] if len(s) > 0 else 0.0
        threshold = max(atol, rtol * sigma_max)

        s_inv = np.zeros_like(s)
        valid = s > threshold
        s_inv[valid] = 1.0 / s[valid]

        D_pinv = Vt.T @ np.diag(s_inv) @ U_svd.T
        gvec = -D_pinv @ fvec

        g[(slice(None),) + idx] = gvec
        rank_D[idx] = np.sum(valid)
        if np.any(valid):
            condition_D[idx] = s[valid][0] / s[valid][-1]

    if return_diagnostics:
        return g, rank_D, condition_D
    return g


# =============================================================================
# PUNTO DE DOLOR 2: Espacio B-spline Tensorial + Galerkin
# =============================================================================

class BSplineTensorSpace:
    """
    Espacio de B-splines tensoriales para Galerkin en D dimensiones.
    """

    def __init__(self, edges, degree=3):
        """
        Parameters
        ----------
        edges : list of D arrays
            Centros de los bins por dirección. Pueden ser no equidistantes.
        degree : int
            Grado de los B-splines (1, 2 o 3 recomendado).
        """
        self.edges = [np.asarray(e, dtype=float) for e in edges]
        self.D = len(self.edges)
        self.degree = int(degree)
        self.n = [len(e) for e in self.edges]          # puntos de evaluación por dim
        self.knots = self._build_knot_vectors()
        self._build_bases()

    def _build_knot_vectors(self):
        """Construye vectores de nodos abiertos con multiplicidad p+1 en extremos."""
        knots = []
        for d in range(self.D):
            x = self.edges[d]
            a, b = x[0], x[-1]
            n = len(x)
            p = self.degree
            if n < p + 2:
                raise ValueError(
                    f"Dimensión {d}: se requieren al menos {p+2} puntos para grado {p}"
                )
            # Nodos interiores necesarios para tener exactamente n bases:
            # n_bases = len(t) - p - 1  =>  len(t) = n + p + 1
            # Multiplicidad en extremos = p+1 cada uno => 2(p+1) nodos fijos
            # Nodos interiores = (n + p + 1) - 2(p+1) = n - p - 1
            n_internal = n - p - 1
            if n_internal > 0:
                # Distribución uniforme en [a, b] (independiente de equidistancia de bins)
                internal = np.linspace(a, b, n_internal)
            else:
                internal = np.array([])
            t = np.concatenate([[a] * (p + 1), internal, [b] * (p + 1)])
            knots.append(t)
        return knots

    def _build_bases(self):
        """Precomputa matrices de evaluación N y derivadas dN en los centros de bins."""
        self.N_vals = []
        self.dN_vals = []
        self.n_bases = []

        for d in range(self.D):
            t = self.knots[d]
            x = self.edges[d]
            p = self.degree
            n_bases = len(t) - p - 1
            self.n_bases.append(n_bases)
            n_eval = len(x)

            N = np.zeros((n_eval, n_bases), dtype=float)
            dN = np.zeros((n_eval, n_bases), dtype=float)

            for idx_pt, xv in enumerate(x):
                span = self._find_span(t, xv, p)
                if span < p or span >= len(t) - p - 1:
                    continue

                ders = self._basis_funs_and_derivatives(t, span, xv, p, n=1)

                for j_local in range(p + 1):
                    j_global = span - p + j_local
                    if 0 <= j_global < n_bases:
                        N[idx_pt, j_global] = ders[0, j_local]
                        dN[idx_pt, j_global] = ders[1, j_local]

            self.N_vals.append(N)
            self.dN_vals.append(dN)

        self.N_dof = int(np.prod(self.n_bases))

    def _find_span(self, knots, x, p):
        """Find span: knots[span] <= x < knots[span+1]. (Piegl & Tiller, Alg. A2.1)"""
        n = len(knots) - p - 1
        if x >= knots[n]:
            return n - 1
        if x <= knots[p]:
            return p
        low, high = p, n
        mid = (low + high) // 2
        while x < knots[mid] or x >= knots[mid + 1]:
            if x < knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2
        return mid

    def _basis_funs(self, knots, span, x, p):
        """Cox-de Boor: evaluate N_{span-p,p} ... N_{span,p} at x. (Piegl & Tiller, Alg. A2.2)"""
        left = np.zeros(p, dtype=float)
        right = np.zeros(p, dtype=float)
        N = np.zeros(p + 1, dtype=float)
        N[0] = 1.0
        for j in range(1, p + 1):
            left[j - 1] = x - knots[span + 1 - j]
            right[j - 1] = knots[span + j] - x
            saved = 0.0
            for r in range(j):
                if right[r] + left[j - 1 - r] == 0:
                    temp = 0.0
                else:
                    temp = N[r] / (right[r] + left[j - 1 - r])
                N[r] = saved + right[r] * temp
                saved = left[j - 1 - r] * temp
            N[j] = saved
        return N

    def _basis_funs_and_derivatives(self, knots, span, x, p, n=1):
        """Evaluate basis functions and derivatives up to order n. (Piegl & Tiller, Alg. A2.3)"""
        ndu = np.zeros((p + 1, p + 1), dtype=float)
        ndu[0, 0] = 1.0
        left = np.zeros(p + 1, dtype=float)
        right = np.zeros(p + 1, dtype=float)

        for j in range(1, p + 1):
            left[j] = x - knots[span + 1 - j]
            right[j] = knots[span + j] - x
            saved = 0.0
            for r in range(j):
                denom = right[r + 1] + left[j - r]
                if denom == 0:
                    temp = 0.0
                else:
                    temp = ndu[r, j - 1] / denom
                ndu[r, j] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            ndu[j, j] = saved

        ders = np.zeros((n + 1, p + 1), dtype=float)
        for j in range(p + 1):
            ders[0, j] = ndu[j, p]

        a = np.zeros((2, p + 1), dtype=float)
        for r in range(p + 1):
            s1, s2 = 0, 1
            a[0, 0] = 1.0
            for k in range(1, n + 1):
                d = 0.0
                rk = r - k
                pk = p - k
                if r >= k:
                    denom = knots[span + r + 1] - knots[span + r + 1 - k]
                    a[s2, 0] = a[s1, 0] / denom if denom != 0 else 0.0
                    d = a[s2, 0] * ndu[rk, pk]
                if r >= k + 1:
                    for j in range(1, k):
                        denom = knots[span + r + j + 1] - knots[span + r + j + 1 - k]
                        a[s2, j] = (a[s1, j] - a[s1, j - 1]) / denom if denom != 0 else 0.0
                        d += a[s2, j] * ndu[rk + j, pk]
                if r <= p - 1:
                    denom = knots[span + r + k + 1] - knots[span + r + 1]
                    a[s2, k] = -a[s1, k - 1] / denom if denom != 0 else 0.0
                    d += a[s2, k] * ndu[r, pk]
                ders[k, r] = d
                j = s1
                s1 = s2
                s2 = j
                if p <= k:
                    for j_idx in range(r + 1, p + 1):
                        a[s1, j_idx] = 0.0
                        a[s2, j_idx] = 0.0

        r = p
        for k in range(1, n + 1):
            for j in range(p + 1):
                ders[k, j] *= r
            r *= (p - k)

        return ders

    def _get_bin_volumes(self):
        """Volumen de cada celda a partir de centros de bins (midpoint rule)."""
        vol = np.ones(tuple(self.n), dtype=float)
        for d in range(self.D):
            x = self.edges[d]
            if len(x) == 1:
                dx = np.array([1.0])
            else:
                # Límites de bins por diferencias entre centros consecutivos
                left = np.concatenate([
                    [x[0] - 0.5 * (x[1] - x[0])],
                    0.5 * (x[:-1] + x[1:]),
                    [x[-1] + 0.5 * (x[-1] - x[-2])]
                ])
                dx = np.diff(left)
            shape = [1] * self.D
            shape[d] = len(dx)
            vol *= dx.reshape(shape)
        return vol

    def evaluate(self, coeffs, points=None):
        """
        Evalúa U^h = sum_j c_j N_j en los centros de bins.

        Parameters
        ----------
        coeffs : ndarray
            Coeficientes (aplanados o tensor de shape n_bases).
        points : None
            (Reservado para futura extensión a puntos arbitrarios).

        Returns
        -------
        U : ndarray, shape (n1, ..., nD)
        """
        if points is not None:
            raise NotImplementedError("Evaluación arbitraria no implementada en esta versión")
        c = np.asarray(coeffs, dtype=float).reshape(self.n_bases)
        result = c
        for d in range(self.D):
            # Contraer el eje d de result con el eje 1 (bases) de N_vals[d]
            result = np.tensordot(result, self.N_vals[d], axes=(d, 1))
            result = np.moveaxis(result, -1, d)
        return result

    def evaluate_gradient(self, coeffs, points=None):
        """
        Evalúa ∇U en los centros de bins.

        Returns
        -------
        grad_U : ndarray, shape (D, n1, ..., nD)
        """
        if points is not None:
            raise NotImplementedError("Evaluación arbitraria no implementada")
        c = np.asarray(coeffs, dtype=float).reshape(self.n_bases)
        grad = np.zeros((self.D,) + tuple(self.n), dtype=float)
        for d in range(self.D):
            temp = c
            for d2 in range(self.D):
                if d2 == d:
                    temp = np.tensordot(temp, self.dN_vals[d2], axes=(d2, 1))
                else:
                    temp = np.tensordot(temp, self.N_vals[d2], axes=(d2, 1))
                temp = np.moveaxis(temp, -1, d2)
            grad[d] = temp
        return grad

    def _get_active_bases_per_cell(self):
        """Precomputa índices de bases activas por dirección y por punto de evaluación."""
        active = []
        for d in range(self.D):
            act_d = []
            for k in range(self.n[d]):
                idx = np.where(np.abs(self.N_vals[d][k, :]) > 1e-14)[0]
                act_d.append(idx)
            active.append(act_d)
        return active

    def assemble_galerkin_system(self, g_field, density, density_mask=None,
                                  lambda_reg=0.0, weights=None):
        """
        Ensambla la matriz de rigidez K (sparse) y el vector de carga F.

        Parameters
        ----------
        g_field : ndarray (D, n1, ..., nD)
        density : ndarray (n1, ..., nD)
        density_mask : bool ndarray, optional
        lambda_reg : float
            Regularización Tikhonov del Hessiano (requiere degree >= 3).
        weights : ndarray, optional
            Pesos de cuadratura personalizados.

        Returns
        -------
        K : scipy.sparse.csr_matrix
        F : ndarray
        info : dict
        """
        D = self.D
        grid_shape = tuple(self.n)

        # Peso por celda: w(x) = min(1, rho / (0.05 * rho_max))
        if weights is not None:
            w = np.asarray(weights)
        else:
            rho_max = np.max(density)
            if rho_max > 0:
                w = np.minimum(1.0, density / (0.05 * rho_max))
            else:
                w = np.zeros_like(density)

        if density_mask is not None:
            w = np.where(density_mask, w, 0.0)

        vol = self._get_bin_volumes()
        w_vol = w * vol

        active_per_dim = self._get_active_bases_per_cell()

        # Strides para índice lineal global
        strides = [1] * D
        for d in range(D - 2, -1, -1):
            strides[d] = strides[d + 1] * self.n_bases[d + 1]

        # Acumuladores COO
        row_idx = []
        col_idx = []
        data_K = []
        F = np.zeros(self.N_dof, dtype=float)

        # Iterar solo celdas con peso positivo
        valid_cells = np.argwhere(w_vol > 0)

        for k_multi in valid_cells:
            k = tuple(k_multi)
            active_lists = [active_per_dim[d][k[d]] for d in range(D)]
            if any(len(a) == 0 for a in active_lists):
                continue

            # Producto cartesiano de índices de bases activas
            idx_grids = np.meshgrid(*active_lists, indexing='ij')
            idx_flat = [g.ravel() for g in idx_grids]
            n_active = idx_flat[0].size

            # Índices globales de las bases activas
            j_global = np.zeros(n_active, dtype=int)
            for d in range(D):
                j_global += idx_flat[d] * strides[d]

            # Valor multivariante N_j(x_k) = prod_d N_{j_d}(x_{k_d})
            N_mult = np.ones(n_active, dtype=float)
            for d in range(D):
                N_mult *= self.N_vals[d][k[d], idx_flat[d]]

            # Derivadas multivariantes dN_j/dx_d
            dN_mult = np.zeros((n_active, D), dtype=float)
            for d in range(D):
                # Producto de N en todas las direcciones
                prod = np.ones(n_active, dtype=float)
                for d2 in range(D):
                    prod *= self.N_vals[d2][k[d2], idx_flat[d2]]
                # Sustituir factor d por derivada
                N_d = self.N_vals[d][k[d], idx_flat[d]]
                dN_d = self.dN_vals[d][k[d], idx_flat[d]]
                safe = np.abs(N_d) > 1e-15
                dN_mult[:, d] = np.where(safe, prod * dN_d / N_d, 0.0)

            wk = w_vol[k]
            gk = g_field[(slice(None),) + k]

            # Vector de carga
            F[j_global] += wk * np.dot(dN_mult, gk)

            # Matriz de rigidez (triangular superior por simetría)
            for ii in range(n_active):
                i = j_global[ii]
                for jj in range(ii, n_active):
                    j = j_global[jj]
                    val = wk * np.dot(dN_mult[ii], dN_mult[jj])
                    row_idx.append(i)
                    col_idx.append(j)
                    data_K.append(val)

        # Construir K simétrica
        K_upper = sparse.coo_matrix(
            (data_K, (row_idx, col_idx)),
            shape=(self.N_dof, self.N_dof)
        )
        K = K_upper + K_upper.T - sparse.diags(K_upper.diagonal())
        K = K.tocsr()

        # Regularización Hessiano (opcional, no implementada en esta versión)
        if lambda_reg > 0.0:
            if self.degree < 3:
                warnings.warn(
                    "lambda_reg > 0 requiere degree >= 3. Ignorando regularización.",
                    UserWarning
                )
            else:
                warnings.warn(
                    "Regularización Hessiano aún no implementada. Ignorando.",
                    UserWarning
                )

        return K, F, {'strides': strides, 'w_vol': w_vol}

    def assemble_galerkin_system_psi(self, g_field, density, density_mask=None,
                                     weights=None):
        """
        Ensambla K (idéntica a la de U) y el vector de carga F_psi para la
        función de corriente ψ (solo D == 2).

        La única diferencia respecto a assemble_galerkin_system es el vector
        de carga: en vez de proyectar g sobre ∇N_i se proyecta sobre
        ∇⊥N_i = (-∂N_i/∂y, ∂N_i/∂x), lo que equivale a alimentar el
        ensamblador estándar con el campo permutado g_perp = (g2, -g1):

            F_psi[j] = ∫ w(x) [ g1·(-∂N_j/∂y) + g2·(∂N_j/∂x) ] dV

        Parameters
        ----------
        g_field : ndarray (2, n1, n2)
        density : ndarray (n1, n2)
        density_mask : bool ndarray, optional
        weights : ndarray, optional

        Returns
        -------
        K : scipy.sparse.csr_matrix
            Idéntica a la de assemble_galerkin_system.
        F_psi : ndarray
        info : dict
        """
        if self.D != 2:
            raise ValueError(
                f"assemble_galerkin_system_psi solo está definido en 2D (D={self.D})"
            )
        g_perp = np.stack([np.asarray(g_field[1], dtype=float),
                           -np.asarray(g_field[0], dtype=float)], axis=0)
        return self.assemble_galerkin_system(
            g_perp, density, density_mask=density_mask,
            lambda_reg=0.0, weights=weights
        )


# =============================================================================
# PUNTO DE DOLOR 1: Descomposición de Helmholtz explícita
# =============================================================================

def compute_helmholtz_residual(g_field, coeffs, bspline_space,
                              density_mask, density):
    """
    Calcula la descomposición de Helmholtz: g = ∇U + r,
    y la métrica de no-equilibrio η = ||r|| / ||g||.

    Returns
    -------
    grad_U : ndarray (D, n1, ..., nD)
    residual : ndarray (D, n1, ..., nD)
    eta : float
    """
    D = bspline_space.D
    grad_U = bspline_space.evaluate_gradient(coeffs)
    residual = g_field - grad_U

    if density_mask is not None:
        residual = np.where(density_mask, residual, np.nan)

    vol = bspline_space._get_bin_volumes()
    w = density * vol
    w_total = np.nansum(w[density_mask] if density_mask is not None else w)
    if w_total > 0:
        w_norm = w / w_total
    else:
        w_norm = np.zeros_like(w)

    norm_g_sq = np.nansum(w_norm * np.nansum(g_field ** 2, axis=0))
    norm_r_sq = np.nansum(w_norm * np.nansum(residual ** 2, axis=0))

    eta = np.sqrt(norm_r_sq / norm_g_sq) if norm_g_sq > 0 else np.nan
    return grad_U, residual, eta


# =============================================================================
# FUNCIÓN DE CORRIENTE ψ (Helmholtz desacoplado 2D)
# =============================================================================

def compute_stream_function(bspline_space, g_field, density, density_mask=None,
                            solver='auto', dirichlet_penalty=1e12):
    """
    Resuelve -∇²ψ = -∇×g para obtener la función de corriente ψ.

    Usa exactamente la misma matriz de rigidez K que para U; solo cambia el
    vector de carga F → F_psi (proyección de g sobre ∇⊥N_i). Aplica la
    condición de Dirichlet homogénea ψ = 0 en ∂Ω penalizando fuertemente
    todos los DOFs de funciones base con soporte en la frontera (para
    vectores de nodos abiertos: índice 0 o n_bases-1 en alguna dirección).

    Parameters
    ----------
    bspline_space : BSplineTensorSpace
    g_field : ndarray (2, n1, n2)
    density : ndarray (n1, n2)
    density_mask : bool ndarray, optional
    solver : str
        'auto', 'direct', 'cg'.
    dirichlet_penalty : float
        Penalización para la condición de Dirichlet homogénea en el borde.

    Returns
    -------
    psi : ndarray (n1, n2)
    coeffs_psi : ndarray (N_dof,)
    info : dict
    """
    if bspline_space.D != 2:
        raise ValueError(
            f"compute_stream_function solo está definido en 2D (D={bspline_space.D})"
        )

    # 1. Ensamblar K (reutilizada) y F_psi
    K, F_psi, info = bspline_space.assemble_galerkin_system_psi(
        g_field, density, density_mask=density_mask
    )
    F_psi = np.nan_to_num(F_psi, nan=0.0)
    N_dof = bspline_space.N_dof
    strides = info['strides']

    # 2. Identificar DOFs de borde (bases con soporte en ∂Ω)
    boundary = np.zeros(N_dof, dtype=bool)
    for d in range(bspline_space.D):
        n_b = bspline_space.n_bases[d]
        md = (np.arange(N_dof) // strides[d]) % n_b
        boundary |= (md == 0) | (md == n_b - 1)

    # 3. Penalizar fuertemente los DOFs de borde: K[j,j] += penalty, F[j] = 0
    K = K.tocsr()
    K = K + sparse.diags(boundary.astype(float) * dirichlet_penalty)
    F_psi[boundary] = 0.0

    # Fijar también DOFs sin ninguna contribución (filas nulas) para evitar
    # singularidad: K[j,j] = 1, F[j] = 0 (quedan en coeficiente cero).
    row_norms = np.array(K.multiply(K).sum(axis=1)).ravel()
    inactive = row_norms <= 1e-20
    if np.any(inactive):
        K = K + sparse.diags(inactive.astype(float))
        F_psi[inactive] = 0.0
    K = K.tocsr()

    # 4. Resolver K·d = F_psi con el mismo solver robusto que para U
    if N_dof <= 1:
        coeffs_psi = np.zeros(N_dof, dtype=float)
    else:
        if solver == 'auto':
            solver = 'direct' if N_dof < 5000 else 'cg'

        if solver == 'direct' or N_dof < 5000:
            try:
                coeffs_psi = spsolve(K, F_psi)
                if coeffs_psi is None or np.any(np.isnan(coeffs_psi)):
                    raise ValueError("spsolve retornó None/NaN")
            except Exception:
                K_dense = K.toarray()
                try:
                    coeffs_psi = np.linalg.solve(K_dense, F_psi)
                except np.linalg.LinAlgError:
                    coeffs_psi, _, _, _ = np.linalg.lstsq(K_dense, F_psi, rcond=None)
        else:
            try:
                M = spilu(K)
                M_op = LinearOperator(K.shape, M.solve)
                coeffs_psi, info_cg = cg(K, F_psi, M=M_op, tol=1e-10, maxiter=20000)
                if info_cg != 0:
                    raise ValueError(f"cg no convergió (info={info_cg})")
            except Exception:
                K_dense = K.toarray()
                try:
                    coeffs_psi = np.linalg.solve(K_dense, F_psi)
                except np.linalg.LinAlgError:
                    coeffs_psi, _, _, _ = np.linalg.lstsq(K_dense, F_psi, rcond=None)

    # 5. Evaluar ψ sobre la grilla
    psi = bspline_space.evaluate(coeffs_psi)

    info_psi = {
        'n_boundary_dofs': int(np.sum(boundary)),
        'n_inactive_dofs': int(np.sum(inactive)),
        'dirichlet_penalty': dirichlet_penalty,
        'solver': solver,
    }
    return psi, coeffs_psi, info_psi


def compute_reconstructed_field(coeffs_U, coeffs_psi, bspline_space):
    """
    Reconstruye el campo completo g_recon = ∇U + ∇⊥ψ.

    Parameters
    ----------
    coeffs_U : ndarray (N_dof,)
    coeffs_psi : ndarray (N_dof,) or None
    bspline_space : BSplineTensorSpace

    Returns
    -------
    g_recon : ndarray (D, n1, ..., nD)
        Si coeffs_psi es None, retorna solo ∇U (backward compat).
    """
    grad_U = bspline_space.evaluate_gradient(coeffs_U)
    if coeffs_psi is None:
        return grad_U
    grad_psi = bspline_space.evaluate_gradient(coeffs_psi)
    g_recon = grad_U.copy()
    g_recon[0] = grad_U[0] - grad_psi[1]   # dU/dx - dpsi/dy
    g_recon[1] = grad_U[1] + grad_psi[0]   # dU/dy + dpsi/dx
    return g_recon


def recover_nonconservative_force(f_field, D_field, grad_U_recon):
    """
    Recupera el componente no-conservativo v = f + D·∇U_recon.

    Parameters
    ----------
    f_field : ndarray (D, n1, ..., nD)
        Drift estimado por Kramers-Moyal.
    D_field : ndarray (D, D, n1, ..., nD)
        Matriz de difusión (la misma escala usada para calcular g = -D⁻¹f).
    grad_U_recon : ndarray (D, n1, ..., nD)
        Gradiente del potencial reconstruido.

    Returns
    -------
    v : ndarray (D, n1, ..., nD)
        Fuerza no-conservativa. Exacta siempre: en equilibrio detallado
        ∇U_recon = -D⁻¹f y por tanto v = 0; fuera de equilibrio, v ≠ 0
        captura la componente rotacional.
    """
    D_dot_gradU = np.einsum('ij...,j...->i...', D_field, grad_U_recon)
    v = f_field + D_dot_gradU
    return v


# Alias interno: evita el shadowing del parámetro compute_stream_function
# dentro de reconstruct_potential_iga.
_solve_stream_function = compute_stream_function


# =============================================================================
# Utilidades
# =============================================================================

def _connected_density_mask(density, threshold):
    """Retorna máscara de la componente conexa (scipy.ndimage) que contiene el máximo."""
    mask = density > threshold
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask
    max_idx = np.unravel_index(np.argmax(density), density.shape)
    target = labeled[max_idx]
    if target == 0:
        # Máximo aislado: tomar componente más grande
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        target = int(np.argmax(sizes)) + 1
    return labeled == target


# =============================================================================
# PIPELINE UNIFICADO
# =============================================================================

def reconstruct_potential_iga(drift, diffusion, edges, density,
                               degree=2, lambda_reg=0.0,
                               rtol=1e-6, atol=1e-12,
                               decompose_helmholtz=True,
                               density_threshold=0.01,
                               dirichlet_penalty=1e12,
                               solver='auto',
                               use_connected_component=True,
                               diffusion_factor = 0.5,
                               compute_stream_function=False):
    """
    Pipeline completo: robusto + Galerkin-B-spline + Helmholtz.

    Parameters
    ----------
    drift : ndarray (D, n1, ..., nD)
    diffusion : ndarray (D, D, n1, ..., nD)
    edges : list of D arrays
    density : ndarray (n1, ..., nD)
    degree : int
        Grado B-spline. Recomendado: 2 para D<=3, 1 para D>=4.
    lambda_reg : float
    rtol, atol : float
        Tolerancias SVD para pseudo-inversa de D.
    decompose_helmholtz : bool
        Si True, calcula residuo rotacional y métrica η.
    density_threshold : float
        Fracción de rho_max para máscara de densidad.
    dirichlet_penalty : float
        Penalización para anclaje U(x_max) ≈ 0.
    solver : str
        'auto', 'direct', 'cg'.
    use_connected_component : bool
        Si True, usa solo la componente conexa de la máscara que contiene el máximo.
    compute_stream_function : bool
        Si True y D==2, calcula la función de corriente ψ via la ecuación de
        Poisson desacoplada -∇²ψ = -∇×g con condición de Dirichlet homogénea
        ψ=0 en la frontera, el campo reconstruido completo
        g_recon = ∇U + ∇⊥ψ y la fuerza no-conservativa v = f + D·∇U.
        Default False para backward compatibility.

    Returns
    -------
    dict con:
        potential, coefficients, g_field, rank_D, condition_D,
        density_mask, gradient, residual, eta, is_equilibrium
        y, si compute_stream_function=True y D==2:
        stream_function, stream_function_coefficients, reconstructed_field,
        reconstructed_drift, nonconservative_force (None en caso contrario)

        Nota sobre dirección: 'reconstructed_field' es g = ∇U + ∇⊥ψ y
        sus streamlines ASCIENDEN el potencial (g = -D⁻¹f por definición).
        'reconstructed_drift' es f_recon = -D·g_recon y sus streamlines
        siguen la dinámica física (descienden U, convergen a atractores).
    """
    D = drift.shape[0]
    grid_shape = tuple(len(e) for e in edges)

    # 1. Máscara de densidad
    rho_max = np.max(density)
    if rho_max <= 0:
        raise ValueError("Densidad empírica es cero en todo el dominio.")

    mask = density > (density_threshold * rho_max)
    if use_connected_component:
        density_mask = _connected_density_mask(density, density_threshold * rho_max)
    else:
        density_mask = mask

    # 2. Campo g robusto (Punto de Dolor 3)
    diffusion = diffusion_factor * diffusion
    g, rank_D, condition_D = compute_g_robust(
        drift, diffusion, density_mask, rtol=rtol, atol=atol
    )

    # 3. Espacio B-spline
    bspline_space = BSplineTensorSpace(edges, degree=degree)
    if tuple(bspline_space.n) != grid_shape:
        raise ValueError(
            f"Incompatibilidad de shapes: grid {grid_shape} vs B-spline {bspline_space.n}"
        )

    # 4. Ensamblar sistema Galerkin
    K, F, info = bspline_space.assemble_galerkin_system(
        g, density, density_mask=density_mask, lambda_reg=lambda_reg
    )

    # 5. Anclaje de Dirichlet + eliminación de DOF inactivos
    max_idx = np.unravel_index(np.argmax(density), grid_shape)
    strides = info['strides']

    # Encontrar la base más dominante en el punto de máxima densidad
    best_j = None
    best_val = -1.0
    candidates_per_dim = []
    for d in range(D):
        n_b = bspline_space.n_bases[d]
        k_d = max_idx[d]
        cand = np.arange(max(0, k_d - degree), min(n_b, k_d + degree + 1))
        candidates_per_dim.append(cand)

    for local in itertools.product(*candidates_per_dim):
        val = 1.0
        for d in range(D):
            val *= bspline_space.N_vals[d][max_idx[d], local[d]]
        if val > best_val:
            best_val = val
            best_j = local

    i0 = sum(best_j[d] * strides[d] for d in range(D))
    i0 = int(np.clip(i0, 0, bspline_space.N_dof - 1))

    # 6. Resolver sistema lineal robusto
    N_dof = bspline_space.N_dof
    if N_dof <= 1:
        c = np.zeros(N_dof, dtype=float)
    else:
        # Identificar DOF activos (filas con al menos una contribución no nula)
        row_norms = np.array(K.multiply(K).sum(axis=1)).ravel()
        active_dof = row_norms > 1e-20
        # El anclaje i0 debe estar entre los activos; si no, forzarlo
        if not active_dof[i0]:
            active_dof[i0] = True
        # Eliminar i0 del conjunto activo (lo fijamos a cero)
        active_dof[i0] = False
        idx_reduced = np.where(active_dof)[0]

        if len(idx_reduced) == 0:
            # Caso degenerado: solo i0 es activo
            c = np.zeros(N_dof, dtype=float)
            c[i0] = 0.0
        else:
            K_reduced = K[idx_reduced][:, idx_reduced]
            F_reduced = F[idx_reduced]

            if solver == 'auto':
                solver = 'direct' if len(idx_reduced) < 5000 else 'cg'

            if solver == 'direct' or len(idx_reduced) < 5000:
                try:
                    c_reduced = spsolve(K_reduced, F_reduced)
                    if c_reduced is None or np.any(np.isnan(c_reduced)):
                        raise ValueError("spsolve retornó None/NaN")
                except Exception:
                    K_dense = K_reduced.toarray()
                    # Si sigue singular, usar mínimos cuadrados
                    try:
                        c_reduced = np.linalg.solve(K_dense, F_reduced)
                    except np.linalg.LinAlgError:
                        c_reduced, _, _, _ = np.linalg.lstsq(K_dense, F_reduced, rcond=None)
            else:
                K_reduced = K_reduced.tocsr()
                try:
                    M = spilu(K_reduced)
                    M_op = LinearOperator(K_reduced.shape, M.solve)
                    c_reduced, info_cg = cg(K_reduced, F_reduced, M=M_op, tol=1e-10, maxiter=20000)
                    if info_cg != 0:
                        K_dense = K_reduced.toarray()
                        c_reduced = np.linalg.solve(K_dense, F_reduced)
                except Exception:
                    K_dense = K_reduced.toarray()
                    c_reduced = np.linalg.solve(K_dense, F_reduced)

            c = np.zeros(N_dof, dtype=float)
            c[i0] = 0.0
            c[idx_reduced] = c_reduced

    # 7. Evaluar potencial en la grilla
    U = bspline_space.evaluate(c)
    if density_mask is not None:
        U = np.where(density_mask, U, np.nan)

    # ==========================================
    # 7b. Stream function ψ (solo D==2, opcional)
    # ==========================================
    psi = None
    coeffs_psi = None
    g_recon = None
    f_recon = None
    v_force = None
    if compute_stream_function and D == 2:
        # Resolver -∇²ψ = -∇×g con Dirichlet homogénea penalizada en ∂Ω
        psi, coeffs_psi, info_psi = _solve_stream_function(
            bspline_space, g, density,
            density_mask=density_mask,
            solver=solver,
            dirichlet_penalty=dirichlet_penalty
        )
        if density_mask is not None and psi is not None:
            psi = np.where(density_mask, psi, np.nan)

        # Campo completo reconstruido: g_recon = ∇U + ∇⊥ψ
        g_recon = compute_reconstructed_field(c, coeffs_psi, bspline_space)

        # Drift reconstruido: f_recon = -D·g_recon. Es la dirección FÍSICA
        # de la dinámica (desciende U, converge a atractores). Ojo: g_recon
        # apunta en dirección de ASCENSO de U, pues g = -D⁻¹f por definición.
        # Si g_recon = g, entonces f_recon = f exactamente (no solo con D
        # isótropo). Se calcula antes de enmascarar g_recon.
        f_recon = -np.einsum('ij...,j...->i...', diffusion, g_recon)

        if density_mask is not None and g_recon is not None:
            g_recon = np.where(density_mask, g_recon, np.nan)
        if density_mask is not None and f_recon is not None:
            f_recon = np.where(density_mask, f_recon, np.nan)

        # Fuerza no-conservativa exacta: v = f + D·∇U (residual de Galerkin).
        # Nota: 'diffusion' ya está escalada por diffusion_factor, de modo
        # que es coherente con g = -D⁻¹f calculado arriba.
        grad_U_full = bspline_space.evaluate_gradient(c)
        v_force = recover_nonconservative_force(drift, diffusion, grad_U_full)
        if density_mask is not None and v_force is not None:
            v_force = np.where(density_mask, v_force, np.nan)
    elif compute_stream_function and D != 2:
        warnings.warn(
            f"compute_stream_function solo está disponible en D==2 (D={D}). "
            "Ignorando.",
            UserWarning
        )

    # 8. Empaquetar resultado
    result = {
        'potential': U,
        'coefficients': c,
        'g_field': g,
        'rank_D': rank_D,
        'condition_D': condition_D,
        'density_mask': density_mask,
        'bspline_space': bspline_space,
        'stream_function': psi,
        'stream_function_coefficients': coeffs_psi,
        'reconstructed_field': g_recon,
        'reconstructed_drift': f_recon,
        'nonconservative_force': v_force,
    }

    if decompose_helmholtz:
        grad_U, residual, eta = compute_helmholtz_residual(
            g, c, bspline_space, density_mask, density
        )
        result['gradient'] = grad_U
        result['residual'] = residual
        result['eta'] = eta
        result['is_equilibrium'] = (eta < 0.1) if not np.isnan(eta) else False

    return result

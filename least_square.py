from __future__ import division, print_function, absolute_import

from warnings import warn

import numpy as np
from numpy.linalg import norm

from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import _minpack, OptimizeResult
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy._lib.six import string_types

from scipy.optimize._lsq.trf import trf
from scipy.optimize._lsq.dogbox import dogbox
from scipy.optimize._lsq.common import EPS, in_bounds, make_strictly_feasible


TERMINATION_MESSAGES = {
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied."
}


FROM_MINPACK_TO_COMMON = {
    0: -1,  # Improper input parameters from MINPACK.
    1: 2,
    2: 3,
    3: 4,
    4: 1,
    5: 0
    # There are 6, 7, 8 for too small tolerance parameters,
    # but we guard against it by checking ftol, xtol, gtol beforehand.
}


def call_minpack(fun, x0, jac, ftol, xtol, gtol, max_nfev, x_scale, diff_step):
    n = x0.size

    if diff_step is None:
        epsfcn = EPS
    else:
        epsfcn = diff_step**2

    # Compute MINPACK's `diag`, which is inverse of our `x_scale` and
    # ``x_scale='jac'`` corresponds to ``diag=None``.
    if isinstance(x_scale, string_types) and x_scale == 'jac':
        diag = None
    else:
        diag = 1 / x_scale

    full_output = True
    col_deriv = False
    factor = 100.0

    if jac is None:
        if max_nfev is None:
            # n squared to account for Jacobian evaluations.
            max_nfev = 100 * n * (n + 1)
        x, info, status = _minpack._lmdif(
            fun, x0, (), full_output, ftol, xtol, gtol,
            max_nfev, epsfcn, factor, diag)
    else:
        if max_nfev is None:
            max_nfev = 100 * n
        x, info, status = _minpack._lmder(
            fun, jac, x0, (), full_output, col_deriv,
            ftol, xtol, gtol, max_nfev, factor, diag)

    f = info['fvec']

    if callable(jac):
        J = jac(x)
    else:
        J = np.atleast_2d(approx_derivative(fun, x))

    cost = 0.5 * np.dot(f, f)
    g = J.T.dot(f)
    g_norm = norm(g, ord=np.inf)

    nfev = info['nfev']
    njev = info.get('njev', None)

    status = FROM_MINPACK_TO_COMMON[status]
    active_mask = np.zeros_like(x0, dtype=int)

    return OptimizeResult(
        x=x, cost=cost, fun=f, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev, status=status)


def prepare_bounds(bounds, n):
    lb, ub = [np.asarray(b, dtype=float) for b in bounds]
    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


def check_tolerance(ftol, xtol, gtol):
    message = "{} is too low, setting to machine epsilon {}."
    if ftol < EPS:
        warn(message.format("`ftol`", EPS))
        ftol = EPS
    if xtol < EPS:
        warn(message.format("`xtol`", EPS))
        xtol = EPS
    if gtol < EPS:
        warn(message.format("`gtol`", EPS))
        gtol = EPS

    return ftol, xtol, gtol


def check_x_scale(x_scale, x0):
    if isinstance(x_scale, string_types) and x_scale == 'jac':
        return x_scale

    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False

    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with "
                         "positive numbers.")

    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)

    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    return x_scale


def check_jac_sparsity(jac_sparsity, m, n):
    if jac_sparsity is None:
        return None

    if not issparse(jac_sparsity):
        jac_sparsity = np.atleast_2d(jac_sparsity)

    if jac_sparsity.shape != (m, n):
        raise ValueError("`jac_sparsity` has wrong shape.")

    return jac_sparsity, group_columns(jac_sparsity)


# Loss functions.


def huber(z, rho, cost_only):
    mask = z <= 1
    rho[0, mask] = z[mask]
    rho[0, ~mask] = 2 * z[~mask]**0.5 - 1
    if cost_only:
        return
    rho[1, mask] = 1
    rho[1, ~mask] = z[~mask]**-0.5
    rho[2, mask] = 0
    rho[2, ~mask] = -0.5 * z[~mask]**-1.5


def soft_l1(z, rho, cost_only):
    t = 1 + z
    rho[0] = 2 * (t**0.5 - 1)
    if cost_only:
        return
    rho[1] = t**-0.5
    rho[2] = -0.5 * t**-1.5


def cauchy(z, rho, cost_only):
    rho[0] = np.log1p(z)
    if cost_only:
        return
    t = 1 + z
    rho[1] = 1 / t
    rho[2] = -1 / t**2


def arctan(z, rho, cost_only):
    rho[0] = np.arctan(z)
    if cost_only:
        return
    t = 1 + z**2
    rho[1] = 1 / t
    rho[2] = -2 * z / t**2


IMPLEMENTED_LOSSES = dict(linear=None, huber=huber, soft_l1=soft_l1,
                          cauchy=cauchy, arctan=arctan)


def construct_loss_function(m, loss, f_scale):
    if loss == 'linear':
        return None

    if not callable(loss):
        loss = IMPLEMENTED_LOSSES[loss]
        rho = np.empty((3, m))

        def loss_function(f, cost_only=False):
            z = (f / f_scale) ** 2
            loss(z, rho, cost_only=cost_only)
            if cost_only:
                return 0.5 * f_scale ** 2 * np.sum(rho[0])
            rho[0] *= f_scale ** 2
            rho[2] /= f_scale ** 2
            return rho
    else:
        def loss_function(f, cost_only=False):
            z = (f / f_scale) ** 2
            rho = loss(z)
            if cost_only:
                return 0.5 * f_scale ** 2 * np.sum(rho[0])
            rho[0] *= f_scale ** 2
            rho[2] /= f_scale ** 2
            return rho

    return loss_function

def least_squares(
        fun, x0, jac='2-point', bounds=(-np.inf, np.inf), method='trf',
        ftol=1e-8, xtol=1e-8, gtol=1e-8, x_scale=1.0, loss='linear',
        f_scale=1.0, diff_step=None, tr_solver=None, tr_options={},
        jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={}):
    
    if method not in ['trf', 'dogbox', 'lm']:
        raise ValueError("`method` must be 'trf', 'dogbox' or 'lm'.")

    if jac not in ['2-point', '3-point', 'cs'] and not callable(jac):
        raise ValueError("`jac` must be '2-point', '3-point', 'cs' or "
                         "callable.")

    if tr_solver not in [None, 'exact', 'lsmr']:
        raise ValueError("`tr_solver` must be None, 'exact' or 'lsmr'.")

    if loss not in IMPLEMENTED_LOSSES and not callable(loss):
        raise ValueError("`loss` must be one of {0} or a callable."
                         .format(IMPLEMENTED_LOSSES.keys()))

    if method == 'lm' and loss != 'linear':
        raise ValueError("method='lm' supports only 'linear' loss function.")

    if verbose not in [0, 1, 2]:
        raise ValueError("`verbose` must be in [0, 1, 2].")

    if len(bounds) != 2:
        raise ValueError("`bounds` must contain 2 elements.")

    if max_nfev is not None and max_nfev <= 0:
        raise ValueError("`max_nfev` must be None or positive integer.")

    if np.iscomplexobj(x0):
        raise ValueError("`x0` must be real.")

    x0 = np.atleast_1d(x0).astype(float)

    if x0.ndim > 1:
        raise ValueError("`x0` must have at most 1 dimension.")

    lb, ub = prepare_bounds(bounds, x0.shape[0])

    if method == 'lm' and not np.all((lb == -np.inf) & (ub == np.inf)):
        raise ValueError("Method 'lm' doesn't support bounds.")

    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")

    if np.any(lb >= ub):
        raise ValueError("Each lower bound must be strictly less than each "
                         "upper bound.")

    if not in_bounds(x0, lb, ub):
        raise ValueError("`x0` is infeasible.")

    x_scale = check_x_scale(x_scale, x0)

    ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol)

    def fun_wrapped(x):
        return np.atleast_1d(fun(x, *args, **kwargs))

    if method == 'trf':
        x0 = make_strictly_feasible(x0, lb, ub)

    f0 = fun_wrapped(x0)

    if f0.ndim != 1:
        raise ValueError("`fun` must return at most 1-d array_like.")

    if not np.all(np.isfinite(f0)):
        raise ValueError("Residuals are not finite in the initial point.")

    n = x0.size
    m = f0.size

    if method == 'lm' and m < n:
        raise ValueError("Method 'lm' doesn't work when the number of "
                         "residuals is less than the number of variables.")

    loss_function = construct_loss_function(m, loss, f_scale)
    if callable(loss):
        rho = loss_function(f0)
        if rho.shape != (3, m):
            raise ValueError("The return value of `loss` callable has wrong "
                             "shape.")
        initial_cost = 0.5 * np.sum(rho[0])
    elif loss_function is not None:
        initial_cost = loss_function(f0, cost_only=True)
    else:
        initial_cost = 0.5 * np.dot(f0, f0)

    if callable(jac):
        J0 = jac(x0, *args, **kwargs)

        if issparse(J0):
            J0 = csr_matrix(J0)

            def jac_wrapped(x, _=None):
                return csr_matrix(jac(x, *args, **kwargs))

        elif isinstance(J0, LinearOperator):
            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs)

        else:
            J0 = np.atleast_2d(J0)

            def jac_wrapped(x, _=None):
                return np.atleast_2d(jac(x, *args, **kwargs))

    else:  # Estimate Jacobian by finite differences.
        if method == 'lm':
            if jac_sparsity is not None:
                raise ValueError("method='lm' does not support "
                                 "`jac_sparsity`.")

            if jac != '2-point':
                warn("jac='{0}' works equivalently to '2-point' "
                     "for method='lm'.".format(jac))

            J0 = jac_wrapped = None
        else:
            if jac_sparsity is not None and tr_solver == 'exact':
                raise ValueError("tr_solver='exact' is incompatible "
                                 "with `jac_sparsity`.")

            jac_sparsity = check_jac_sparsity(jac_sparsity, m, n)

            def jac_wrapped(x, f):
                J = approx_derivative(fun, x, rel_step=diff_step, method=jac,
                                      f0=f, bounds=bounds, args=args,
                                      kwargs=kwargs, sparsity=jac_sparsity)
                if J.ndim != 2:  # J is guaranteed not sparse.
                    J = np.atleast_2d(J)

                return J

            J0 = jac_wrapped(x0, f0)

    if J0 is not None:
        if J0.shape != (m, n):
            raise ValueError(
                "The return value of `jac` has wrong shape: expected {0}, "
                "actual {1}.".format((m, n), J0.shape))

        if not isinstance(J0, np.ndarray):
            if method == 'lm':
                raise ValueError("method='lm' works only with dense "
                                 "Jacobian matrices.")

            if tr_solver == 'exact':
                raise ValueError(
                    "tr_solver='exact' works only with dense "
                    "Jacobian matrices.")

        jac_scale = isinstance(x_scale, string_types) and x_scale == 'jac'
        if isinstance(J0, LinearOperator) and jac_scale:
            raise ValueError("x_scale='jac' can't be used when `jac` "
                             "returns LinearOperator.")

        if tr_solver is None:
            if isinstance(J0, np.ndarray):
                tr_solver = 'exact'
            else:
                tr_solver = 'lsmr'

    if method == 'lm':
        result = call_minpack(fun_wrapped, x0, jac_wrapped, ftol, xtol, gtol,
                              max_nfev, x_scale, diff_step)

    elif method == 'trf':
        result = trf(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol, xtol,
                     gtol, max_nfev, x_scale, loss_function, tr_solver,
                     tr_options.copy(), verbose)

    elif method == 'dogbox':
        if tr_solver == 'lsmr' and 'regularize' in tr_options:
            warn("The keyword 'regularize' in `tr_options` is not relevant "
                 "for 'dogbox' method.")
            tr_options = tr_options.copy()
            del tr_options['regularize']

        result = dogbox(fun_wrapped, jac_wrapped, x0, f0, J0, lb, ub, ftol,
                        xtol, gtol, max_nfev, x_scale, loss_function,
                        tr_solver, tr_options, verbose)

    result.message = TERMINATION_MESSAGES[result.status]
    result.success = result.status > 0

    if verbose >= 1:
        print(result.message)
        print("Function evaluations {0}, initial cost {1:.4e}, final cost "
              "{2:.4e}, first-order optimality {3:.2e}."
              .format(result.nfev, initial_cost, result.cost,
                      result.optimality))

    return result